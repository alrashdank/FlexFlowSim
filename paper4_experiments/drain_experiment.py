"""Drain-mode experiment (reviewer concern 1) + pure-throughput PPO (concern 3).

Part A: retrain 5 baseline seeds (wc=0.10, CostFocus) with model saving, then
evaluate each policy and the four rules under two accountings:
  (a) standard: truncate at t=480 (reproduces published protocol)
  (b) drain: arrivals stop at t=480, simulation runs until the system is
      empty (cap 3000), all costs charged, all departures counted.
Part B: pure-throughput PPO, weights (0, 1, 0), 5 seeds, standard eval.

Usage: python3 drain_experiment.py [A|B|BOTH]
Output: drain_results.json / puretp_results.json
"""
import json, sys, time
import numpy as np
from stable_baselines3 import PPO
from env import FlexFlowSimEnv, load_config
import baselines

SEEDS = [42, 123, 256, 512, 1024]
EVAL_SEED = 9999
N_EVAL = 50
RULES = ["RoundRobin", "Random", "ShortestQueue", "LeastUtilised"]
COSTFOCUS = (0.8, 0.1, 0.1)
PURETP = (0.0, 1.0, 0.0)


class DrainEnv(FlexFlowSimEnv):
    """Arrivals stop at arrival_end; episode truncates at max_time only."""
    def __init__(self, cfg, weights, arrival_end):
        self._arrival_end = arrival_end
        super().__init__(cfg, weights=weights)

    def _arrival_process(self):
        while True:
            iat = self._arrival_sampler()
            yield self._simpy_env.timeout(iat)
            if self._simpy_env.now >= self._arrival_end:
                return
            route = self._action_tuples[self._current_action]
            self._simpy_env.process(self._entity_process(route))


def jobs_in_system(env):
    return int(np.sum(env._queue_len) + np.sum(env._in_service))


def run_episode(env, act_fn, ep_seed, drain=False):
    obs, info = env.reset(seed=ep_seed)
    done = False
    t480 = None
    while not done:
        obs, r, term, trunc, info = env.step(act_fn(obs))
        now = env._simpy_env.now
        if drain and t480 is None and now >= 480:
            t480 = dict(cost=float(info["total_cost"]),
                        dep=int(info["total_departed"]),
                        wip=jobs_in_system(env),
                        processing=float(info["processing_cost"]),
                        idle=float(info["idle_cost"]),
                        waiting=float(info["waiting_cost"]))
        if drain and now >= 480 and jobs_in_system(env) == 0:
            break
        done = term or trunc
    out = dict(total_cost=float(info["total_cost"]),
               throughput=int(info["total_departed"]),
               cpu=float(info["total_cost"]) / max(int(info["total_departed"]), 1),
               processing=float(info["processing_cost"]),
               idle=float(info["idle_cost"]),
               waiting=float(info["waiting_cost"]),
               end_time=float(env._simpy_env.now))
    if drain and t480:
        out["at480"] = t480
    return out


def evaluate(act_fn, cfg, weights, drain, policy=None):
    if drain:
        c = dict(cfg); c["max_time"] = float(cfg.get("drain_max_time", 8000.0))
        env = DrainEnv(c, weights, arrival_end=480.0)
    else:
        env = FlexFlowSimEnv(cfg, weights=weights)
    eps = []
    for i in range(N_EVAL):
        if policy is not None and hasattr(policy, "reset"):
            policy.reset()
        eps.append(run_episode(env, act_fn, EVAL_SEED + i, drain))
    env.close()
    return eps


def rule_fn(name, cfg):
    env = FlexFlowSimEnv(cfg, weights=COSTFOCUS)
    cls = getattr(baselines, name + "Policy")
    try:
        pol = cls(env=env, seed=EVAL_SEED)
    except TypeError:
        pol = cls(env=env)
    env.close()
    return pol, (lambda obs: int(pol.predict(obs)))


def model_fn(model):
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


def train(cfg, weights, seed, tag):
    env = FlexFlowSimEnv(cfg, weights=weights)
    spe = int(cfg["max_time"] / cfg["dt"])
    m = PPO("MlpPolicy", env, seed=seed, verbose=0,
            learning_rate=3e-4, n_steps=min(2048, spe), batch_size=64,
            n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs={"net_arch": [64, 64]})
    m.learn(total_timesteps=500 * spe)
    m.save(f"model_{tag}_s{seed}")
    env.close()
    return m


def part_a():
    cfg = load_config("configs/bakery_bk50.json")
    res = {"ppo": {}, "rules": {}}
    try:
        res = json.load(open("drain_results.json"))
    except Exception:
        pass
    for rule in RULES:
        if rule in res["rules"]:
            print(f"[A] rule {rule} banked, skipping", flush=True)
            continue
        print(f"[A] rule {rule}: standard + drain eval", flush=True)
        pol, fn = rule_fn(rule, cfg)
        res["rules"][rule] = {
            "standard": evaluate(fn, cfg, COSTFOCUS, drain=False, policy=pol),
            "drain": evaluate(fn, cfg, COSTFOCUS, drain=True, policy=pol)}
        json.dump(res, open("drain_results.json", "w"))
    import os
    try:
        res = json.load(open("drain_results.json"))
    except Exception:
        pass
    for seed in SEEDS:
        if str(seed) in res["ppo"]:
            print(f"[A] seed {seed} already banked, skipping", flush=True)
            continue
        t0 = time.time()
        mp = f"model_wc010_s{seed}.zip"
        if os.path.exists(mp):
            print(f"[A] loading existing model seed {seed}", flush=True)
            m = PPO.load(mp)
        else:
            print(f"[A] training seed {seed}...", flush=True)
            m = train(cfg, COSTFOCUS, seed, "wc010")
        fn = model_fn(m)
        res["ppo"][str(seed)] = {
            "standard": evaluate(fn, cfg, COSTFOCUS, drain=False),
            "drain": evaluate(fn, cfg, COSTFOCUS, drain=True)}
        json.dump(res, open("drain_results.json", "w"))
        cpu_s = np.mean([e["cpu"] for e in res["ppo"][str(seed)]["standard"]])
        cpu_d = np.mean([e["cpu"] for e in res["ppo"][str(seed)]["drain"]])
        print(f"    {time.time()-t0:.0f}s  std CPU {cpu_s:.1f} | drain CPU {cpu_d:.1f}",
              flush=True)


def part_b():
    cfg = load_config("configs/bakery_bk50.json")
    res = {"ppo": {}}
    import os
    try:
        res = json.load(open("puretp_results.json"))
    except Exception:
        pass
    for seed in SEEDS:
        if str(seed) in res["ppo"]:
            print(f"[B] seed {seed} already banked, skipping", flush=True)
            continue
        t0 = time.time()
        mp = f"model_puretp_s{seed}.zip"
        if os.path.exists(mp):
            print(f"[B] loading existing model seed {seed}", flush=True)
            m = PPO.load(mp)
        else:
            print(f"[B] pure-throughput training seed {seed}...", flush=True)
            m = train(cfg, PURETP, seed, "puretp")
        fn = model_fn(m)
        res["ppo"][str(seed)] = {"standard": evaluate(fn, cfg, COSTFOCUS, drain=False)}
        json.dump(res, open("puretp_results.json", "w"))
        e = res["ppo"][str(seed)]["standard"]
        print(f"    {time.time()-t0:.0f}s  TP {np.mean([x['throughput'] for x in e]):.1f}"
              f"  CPU {np.mean([x['cpu'] for x in e]):.1f}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "BOTH"
    if which in ("A", "BOTH"):
        part_a()
    if which in ("B", "BOTH"):
        part_b()
    print("DONE", which)
