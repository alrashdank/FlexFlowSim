"""Optuna-Static under canonical evaluation seeds (v7 spec section 5).

Trains PPO with the Optuna-selected hyperparameters (from the original
10-trial search recorded in paper_tables.json), 5 seeds x 500 episodes on
Bakery, then evaluates under the canonical protocol (seeds 9999+i) under
both accountings with the dedicated arrival stream.
"""
import json
import os
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

BP = json.load(open("/mnt/project/paper_tables.json"))["optuna"]["best_params"]
SEEDS = [42, 123, 256, 512, 1024]
OUT = "optuna_canonical.json"


def main():
    cfg = load_config("configs/bakery_bk50.json")
    spe = int(cfg["max_time"] / cfg["dt"])
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        "best_params": BP, "ppo": {}}
    for seed in SEEDS:
        if str(seed) in res["ppo"]:
            print(f"seed {seed} banked", flush=True)
            continue
        t0 = time.time()
        mp = f"model_optuna_s{seed}.zip"
        if os.path.exists(mp):
            m = PPO.load(mp)
        else:
            print(f"training optuna-static seed {seed}...", flush=True)
            env = FlexFlowSimEnv(cfg, weights=D.COSTFOCUS)
            m = PPO("MlpPolicy", env, seed=seed, verbose=0,
                    learning_rate=BP["lr"], n_steps=BP["n_steps"],
                    batch_size=64, n_epochs=BP["n_epochs"],
                    gamma=BP["gamma"], gae_lambda=0.95,
                    clip_range=BP["clip_range"], ent_coef=BP["ent_coef"],
                    vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs={"net_arch": [64, 64]})
            m.learn(total_timesteps=500 * spe)
            m.save(f"model_optuna_s{seed}")
            env.close()
        fn = lambda o: int(m.predict(o, deterministic=True)[0])
        res["ppo"][str(seed)] = {
            "standard": D.evaluate(fn, cfg, D.COSTFOCUS, drain=False),
            "drain": D.evaluate(fn, cfg, D.COSTFOCUS, drain=True)}
        json.dump(res, open(OUT, "w"))
        d = res["ppo"][str(seed)]
        print(f"  seed {seed} [{time.time()-t0:.0f}s] "
              f"total {np.mean([e['total_cost'] for e in d['standard']]):.0f} "
              f"TP {np.mean([e['throughput'] for e in d['standard']]):.1f} "
              f"CPU {np.mean([e['cpu'] for e in d['standard']]):.1f} "
              f"| drain {np.mean([e['total_cost'] for e in d['drain']]):.0f}",
              flush=True)
    print("OPTUNA CANONICAL COMPLETE")


if __name__ == "__main__":
    main()
