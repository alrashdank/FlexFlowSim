"""Drain extension to a further environment (reviewer priority 2).

Trains PPO-Default (gamma = 0.95, CostFocus, 500 episodes, 5 seeds) on the
given config, then evaluates PPO and all four dispatching rules under BOTH
accountings on identical arrival streams (dedicated arrival RNG), exactly as
in redrain3.py for Bakery.

Usage: python3 env_drain.py configs/electronics_3stage.json electronics
"""
import json
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO

import drain_experiment as D
from env import FlexFlowSimEnv, load_config, _make_sampler


class SplitRNGDrainEnv(D.DrainEnv):
    def reset(self, seed=None, options=None):
        r = super().reset(seed=seed, options=options)
        arr_rng = np.random.default_rng(10_000_019 * (seed or 0) + 7)
        self._arrival_sampler = _make_sampler(self._arrival_cfg, arr_rng)
        return r


D.DrainEnv = SplitRNGDrainEnv

CFG_PATH, TAG = sys.argv[1], sys.argv[2]
SEEDS = [42, 123, 256, 512, 1024]
RULES = ["RoundRobin", "Random", "ShortestQueue", "LeastUtilised"]
OUT = f"drain_{TAG}.json"


def main():
    cfg = load_config(CFG_PATH)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"ppo": {}, "rules": {}}

    for rule in RULES:
        if rule in res["rules"]:
            print(f"rule {rule} banked, skipping", flush=True)
            continue
        pol, fn = D.rule_fn(rule, cfg)
        res["rules"][rule] = {
            "standard": D.evaluate(fn, cfg, D.COSTFOCUS, drain=False, policy=pol),
            "drain": D.evaluate(fn, cfg, D.COSTFOCUS, drain=True, policy=pol)}
        json.dump(res, open(OUT, "w"))
        d = res["rules"][rule]
        print(f"rule {rule}: fixed {np.mean([e['total_cost'] for e in d['standard']]):.0f} "
              f"| drain {np.mean([e['total_cost'] for e in d['drain']]):.0f}", flush=True)

    spe = int(cfg["max_time"] / cfg["dt"])
    for seed in SEEDS:
        if str(seed) in res["ppo"]:
            print(f"seed {seed} banked, skipping", flush=True)
            continue
        t0 = time.time()
        mp = f"model_{TAG}_s{seed}.zip"
        if os.path.exists(mp):
            m = PPO.load(mp)
            print(f"loaded model seed {seed}", flush=True)
        else:
            print(f"training {TAG} seed {seed}...", flush=True)
            env = FlexFlowSimEnv(cfg, weights=D.COSTFOCUS)
            m = PPO("MlpPolicy", env, seed=seed, verbose=0,
                    learning_rate=3e-4, n_steps=min(2048, spe), batch_size=64,
                    n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs={"net_arch": [64, 64]})
            m.learn(total_timesteps=500 * spe)
            m.save(f"model_{TAG}_s{seed}")
            env.close()
        fn = lambda o: int(m.predict(o, deterministic=True)[0])
        res["ppo"][str(seed)] = {
            "standard": D.evaluate(fn, cfg, D.COSTFOCUS, drain=False),
            "drain": D.evaluate(fn, cfg, D.COSTFOCUS, drain=True)}
        json.dump(res, open(OUT, "w"))
        d = res["ppo"][str(seed)]
        fully = all(e["throughput"] == e["at480"]["dep"] + e["at480"]["wip"]
                    for e in d["drain"] if "at480" in e)
        print(f"  seed {seed} [{time.time()-t0:.0f}s] "
              f"fixed {np.mean([e['total_cost'] for e in d['standard']]):.0f} "
              f"TP {np.mean([e['throughput'] for e in d['standard']]):.1f} "
              f"| drain {np.mean([e['total_cost'] for e in d['drain']]):.0f} "
              f"| fully drained {fully}", flush=True)
    print(f"{TAG.upper()} DRAIN COMPLETE")


if __name__ == "__main__":
    main()
