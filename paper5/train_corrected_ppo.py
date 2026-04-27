"""
Paper 5 — Corrected PPO Training Grid
=====================================

Final experimental run that produced ``paper5/results/ppo_fixed_eval_sweep.csv``.
Trains PPO across all CPU-aligned electronics configs (stationary + 5
breakdown severities), 5 seeds each, using the three protocol corrections
described in the manuscript:

  - gamma = 0.99 (longer-horizon credit assignment, ~100 effective steps)
  - CPU-aligned reward (set ``cpu_aligned_reward: true`` in the config)
  - Eval-based checkpointing (save the policy with the lowest mean
    cost-per-unit on a held-out 10-seed eval set, evaluated every 25
    training episodes)

Pure cost-minimisation reward weighting (1.0, 0.0, 0.0) throughout.

Total: 30 training runs (6 configs x 5 seeds). Expected ~2.5 hr on a
recent laptop CPU.

Usage (run from the repository root):
    python paper5/train_corrected_ppo.py
    python paper5/train_corrected_ppo.py --skip-existing
    python paper5/train_corrected_ppo.py --seeds 42 --episodes 300   # smoke test
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Ensure the repository root is on sys.path when invoked as
# ``python paper5/train_corrected_ppo.py``.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Anchor CWD so all relative paths (configs/, results/) resolve inside the repo.
os.chdir(REPO_ROOT)

from env import FlexFlowSimEnv


WEIGHTS = (1.0, 0.0, 0.0)
TIMESTEPS_PER_EP = 480

PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 480,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [64, 64]},
}

EVAL_INTERVAL = 25          # eval every 25 training episodes
EVAL_SEEDS = 10             # per intermediate eval
FINAL_EVAL_SEEDS = 50       # per final eval
DEFAULT_SEEDS = [42, 123, 256, 512, 1024]

TRAIN_EVAL_SEED_OFFSET = 999000   # eval seeds during training
FINAL_EVAL_SEED_OFFSET = 500000   # eval seeds after training (disjoint)


def get_configs():
    return sorted(glob.glob("configs/paper5_electronics_cpu/*.json"))


def extract_severity(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("_metadata", {}).get("availability_target", 1.0)


def eval_policy(model, config_path, seed_offset, n_seeds):
    cpus = []
    tps = []
    for s in range(n_seeds):
        env = FlexFlowSimEnv(config=config_path, weights=WEIGHTS,
                             seed=seed_offset + s)
        obs, _ = env.reset(seed=seed_offset + s)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(a))
            done = term or trunc
        dep = info["total_departed"]
        cpus.append(info["total_cost"] / max(dep, 1))
        tps.append(dep)
    return cpus, tps


class EvalCheckpointCallback(BaseCallback):
    def __init__(self, config_path, save_path, verbose=1):
        super().__init__(verbose=verbose)
        self.config_path = config_path
        self.save_path = save_path
        self.best_eval_cpu = np.inf
        self.best_eval_episode = 0
        self.episode_count = 0
        self.history = []
        self._ep_reward = 0.0
        self.episode_rewards = []

    def _on_step(self):
        self._ep_reward += self.locals.get("rewards", [0.0])[0]
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
            if self.episode_count % EVAL_INTERVAL == 0:
                cpus, tps = eval_policy(
                    self.model, self.config_path,
                    TRAIN_EVAL_SEED_OFFSET, EVAL_SEEDS
                )
                mean_cpu = float(np.mean(cpus))
                mean_tp = float(np.mean(tps))
                self.history.append({
                    "episode": self.episode_count,
                    "mean_cpu": mean_cpu,
                    "mean_tp": mean_tp,
                })
                if mean_cpu < self.best_eval_cpu:
                    self.best_eval_cpu = mean_cpu
                    self.best_eval_episode = self.episode_count
                    self.model.save(self.save_path)
                if self.verbose >= 1:
                    print(f"      [Ep {self.episode_count:4d}] "
                          f"eval CPU={mean_cpu:.2f} TP={mean_tp:.1f} "
                          f"(best={self.best_eval_cpu:.2f}@ep{self.best_eval_episode})")
        return True


def train_one(config_path, seed, episodes, save_dir, verbose=1):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"PPO_fixed_seed{seed}_best.zip")

    env = FlexFlowSimEnv(config=config_path, weights=WEIGHTS, seed=seed)
    env.reset(seed=seed)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, **PPO_KWARGS)

    cb = EvalCheckpointCallback(config_path, save_path, verbose=verbose)
    t0 = time.time()
    model.learn(total_timesteps=episodes * TIMESTEPS_PER_EP, callback=cb)
    train_time = time.time() - t0

    return {
        "seed": seed,
        "best_eval_cpu": cb.best_eval_cpu,
        "best_eval_episode": cb.best_eval_episode,
        "total_episodes": cb.episode_count,
        "train_time_s": train_time,
        "eval_history": cb.history,
        "save_path": save_path,
    }


def final_eval(model_path, config_path, train_seed):
    cpus = []
    tps = []
    lts = []
    bd_counts = []
    for s in range(FINAL_EVAL_SEEDS):
        eval_seed = FINAL_EVAL_SEED_OFFSET + train_seed * 100 + s
        env = FlexFlowSimEnv(config=config_path, weights=WEIGHTS, seed=eval_seed)
        model = PPO.load(model_path, env=env)
        obs, _ = env.reset(seed=eval_seed)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(a))
            done = term or trunc
        dep = info["total_departed"]
        cpus.append(info["total_cost"] / max(dep, 1))
        tps.append(dep)
        lts.append(info["avg_lead_time"])
        if "breakdown_count" in info:
            bd_counts.append(sum(info["breakdown_count"]))
        else:
            bd_counts.append(0)
    return cpus, tps, lts, bd_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="results/paper5_phase12_full_fixed_grid")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    configs = get_configs()

    total_runs = len(configs) * len(args.seeds)
    print("=" * 70)
    print("  PHASE 12: FULL FIXED GRID")
    print("  (gamma=0.99, CPU-aligned reward, eval-based checkpointing)")
    print("=" * 70)
    print(f"  Configs: {len(configs)}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Episodes per run: {args.episodes}")
    print(f"  Total training runs: {total_runs}")
    print(f"  Estimated time: {total_runs * 5 / 60:.1f} hr")
    print()

    all_train_logs = []
    all_eval_rows = []
    t_total = time.time()

    for ci, cfg_path in enumerate(configs):
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        sev = extract_severity(cfg_path)
        cfg_out = os.path.join(args.output, cfg_name)
        os.makedirs(cfg_out, exist_ok=True)

        print(f"\n  [{ci+1}/{len(configs)}] {cfg_name} (severity={sev})")
        print(f"  {'-' * 60}")

        for seed in args.seeds:
            save_path = os.path.join(cfg_out, f"PPO_fixed_seed{seed}_best.zip")

            if args.skip_existing and os.path.exists(save_path):
                print(f"    seed={seed}: skipping (exists)")
            else:
                print(f"    seed={seed}: training...")
                log = train_one(cfg_path, seed, args.episodes, cfg_out, verbose=1)
                log.update({"config": cfg_name, "severity": sev})
                all_train_logs.append(log)
                print(f"    seed={seed}: trained in {log['train_time_s']:.0f}s, "
                      f"best CPU={log['best_eval_cpu']:.2f}")

            if os.path.exists(save_path):
                cpus, tps, lts, bds = final_eval(save_path, cfg_path, seed)
                for ei, (cpu, tp, lt, bd) in enumerate(zip(cpus, tps, lts, bds)):
                    all_eval_rows.append({
                        "config": cfg_name,
                        "severity": sev,
                        "method": "PPO_fixed",
                        "trainSeed": seed,
                        "evalSeed": FINAL_EVAL_SEED_OFFSET + seed * 100 + ei,
                        "costPerUnit": cpu,
                        "totalDeparted": tp,
                        "avgLeadTime": lt,
                        "breakdownCount": bd,
                    })
                mean_cpu = float(np.mean(cpus))
                std_cpu = float(np.std(cpus))
                mean_tp = float(np.mean(tps))
                print(f"    seed={seed}: final eval CPU={mean_cpu:.2f}"
                      f" +/- {std_cpu:.2f}, TP={mean_tp:.1f}")

    elapsed = time.time() - t_total

    if all_train_logs:
        log_path = os.path.join(args.output, "ppo_fixed_training_log.json")
        with open(log_path, "w") as f:
            json.dump(all_train_logs, f, indent=2, default=str)

    if all_eval_rows:
        df = pd.DataFrame(all_eval_rows)
        eval_path = os.path.join(args.output, "ppo_fixed_eval_sweep.csv")
        df.to_csv(eval_path, index=False)
        summary = df.groupby(["config", "severity"])["costPerUnit"].agg(
            ["mean", "std", "count"]).round(2)
        summary_path = os.path.join(args.output, "ppo_fixed_eval_summary.csv")
        summary.to_csv(summary_path)

    print()
    print("=" * 70)
    print(f"  DONE — {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    print("=" * 70)

    if all_eval_rows:
        df = pd.DataFrame(all_eval_rows)
        print()
        print("  SUMMARY — PPO (fixed) CPU by severity")
        print(f"  {'Config':<42} {'Sev':>6} {'Mean CPU':>9} {'Std':>7} {'TP':>6}")
        print("  " + "-" * 75)
        for cfg in sorted(df["config"].unique()):
            sub = df[df["config"] == cfg]
            sev = sub["severity"].iloc[0]
            print(f"  {cfg:<42} {sev:>6.2f} "
                  f"{sub['costPerUnit'].mean():>9.2f} "
                  f"{sub['costPerUnit'].std():>7.2f} "
                  f"{sub['totalDeparted'].mean():>6.1f}")


if __name__ == "__main__":
    main()
