"""Gamma robustness (reviewer priority 1).

Trains PPO-Default at a specified discount factor on Bakery (wc = 0.10,
CostFocus weights), identical five-seed 500-episode protocol, then evaluates
each policy under BOTH accountings on identical arrival streams (dedicated
arrival RNG, as in redrain3.py) so results are directly comparable with the
canonical gamma = 0.95 dataset in drain_results.json.

Records: total cost, throughput, CPU, WIP at 480, drain time, cost
decomposition, and routing action counts (concentration).

Usage: python3 gamma_experiment.py 0.99
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
    """Arrival times from a dedicated stream: policy-invariant by construction."""

    def reset(self, seed=None, options=None):
        r = super().reset(seed=seed, options=options)
        arr_rng = np.random.default_rng(10_000_019 * (seed or 0) + 7)
        self._arrival_sampler = _make_sampler(self._arrival_cfg, arr_rng)
        return r


D.DrainEnv = SplitRNGDrainEnv

GAMMA = float(sys.argv[1])
SEEDS = [42, 123, 256, 512, 1024]
TAG = "g" + str(GAMMA).replace(".", "")
OUT = f"gamma_results_{TAG}.json"


def train_gamma(cfg, seed, gamma):
    env = FlexFlowSimEnv(cfg, weights=D.COSTFOCUS)
    spe = int(cfg["max_time"] / cfg["dt"])
    m = PPO("MlpPolicy", env, seed=seed, verbose=0,
            learning_rate=3e-4, n_steps=min(2048, spe), batch_size=64,
            n_epochs=10, gamma=gamma, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs={"net_arch": [64, 64]})
    m.learn(total_timesteps=500 * spe)
    m.save(f"model_{TAG}_s{seed}")
    env.close()
    return m


def counting_fn(model, counter):
    def f(obs):
        a = int(model.predict(obs, deterministic=True)[0])
        counter[a] = counter.get(a, 0) + 1
        return a
    return f


def main():
    cfg = load_config("configs/bakery_bk50.json")
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        "gamma": GAMMA, "ppo": {}}
    for seed in SEEDS:
        if str(seed) in res["ppo"]:
            print(f"seed {seed} banked, skipping", flush=True)
            continue
        t0 = time.time()
        mp = f"model_{TAG}_s{seed}.zip"
        if os.path.exists(mp):
            print(f"loading model seed {seed}", flush=True)
            m = PPO.load(mp)
        else:
            print(f"training gamma={GAMMA} seed {seed}...", flush=True)
            m = train_gamma(cfg, seed, GAMMA)
        cs, cd = {}, {}
        std = D.evaluate(counting_fn(m, cs), cfg, D.COSTFOCUS, drain=False)
        drn = D.evaluate(counting_fn(m, cd), cfg, D.COSTFOCUS, drain=True)
        res["ppo"][str(seed)] = {"standard": std, "drain": drn,
                                 "actions_standard": cs, "actions_drain": cd}
        json.dump(res, open(OUT, "w"))
        tot = sum(cs.values())
        top = max(cs.values()) / tot * 100 if tot else 0
        fully = all(e["throughput"] == e["at480"]["dep"] + e["at480"]["wip"]
                    for e in drn if "at480" in e)
        print(f"  seed {seed} [{time.time()-t0:.0f}s] "
              f"TP {np.mean([e['throughput'] for e in std]):.1f} "
              f"CPU {np.mean([e['cpu'] for e in std]):.1f} "
              f"| drain {np.mean([e['total_cost'] for e in drn]):.0f} "
              f"| top-route {top:.0f}% | fully drained {fully}", flush=True)
    print(f"GAMMA {GAMMA} COMPLETE")


if __name__ == "__main__":
    main()
