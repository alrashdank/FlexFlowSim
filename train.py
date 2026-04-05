"""
FlexFlowSim — Multi-Seed Training Script
==========================================

Trains RL agents (DQN and/or PPO) across multiple seeds and weight scenarios.
Implements the Paper 3 multi-seed protocol.

Usage:
    python train.py --config configs/bakery_bk50.json
    python train.py --config configs/bakery_bk50.json --algo PPO --episodes 500
    python train.py --config configs/bakery_bk50.json --scenario CostFocus --seeds 42 123 256
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from env import FlexFlowSimEnv, load_config


# ═══════════════════════════════════════════════════════════════════
# SCENARIOS
# ═══════════════════════════════════════════════════════════════════

SCENARIOS = {
    "CostFocus":       {"weights": (0.8, 0.1, 0.1),   "label": "Cost-Dominant"},
    "ThroughputFocus": {"weights": (0.1, 0.8, 0.1),   "label": "Throughput-Dominant"},
    "LeadTimeFocus":   {"weights": (0.1, 0.1, 0.8),   "label": "Lead Time-Dominant"},
    "Balanced":        {"weights": (0.33, 0.33, 0.34), "label": "Balanced"},
}

DEFAULT_SEEDS = [42, 123, 256, 512, 1024]

ALGO_MAP = {
    "DQN": DQN,
    "PPO": PPO,
}


# ═══════════════════════════════════════════════════════════════════
# CALLBACK
# ═══════════════════════════════════════════════════════════════════

class EpisodeTracker(BaseCallback):
    """Tracks episode rewards and saves best agent."""

    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_reward = -np.inf
        self.best_episode = 0
        self.episode_count = 0
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals.get("rewards", [0])[0]
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            self.episode_rewards.append(self._current_reward)
            if self._current_reward > self.best_reward:
                self.best_reward = self._current_reward
                self.best_episode = self.episode_count
                self.model.save(self.save_path)
            if self.verbose and self.episode_count % 25 == 0:
                avg = np.mean(self.episode_rewards[-20:])
                print(f"    [Ep {self.episode_count:4d}] Avg(20)={avg:.3f}  "
                      f"Best={self.best_reward:.3f} (Ep {self.best_episode})")
            self._current_reward = 0.0
        return True


# ═══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════

def get_hyperparams(algo_name, steps_per_episode):
    """Return tuned hyperparameters for each algorithm."""
    if algo_name == "DQN":
        return {
            "learning_rate": 5e-4,
            "buffer_size": 100_000,
            "learning_starts": max(1000, steps_per_episode * 2),
            "batch_size": 256,
            "gamma": 0.95,
            "target_update_interval": 500,
            "exploration_fraction": 0.5,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "train_freq": 4,
            "gradient_steps": 1,
            "policy_kwargs": {"net_arch": [64, 64]},
        }
    elif algo_name == "PPO":
        return {
            "learning_rate": 3e-4,
            "n_steps": min(2048, steps_per_episode),
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {"net_arch": [64, 64]},
        }
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_single(config_path, algo_name, scenario_name, weights,
                 seed, total_episodes, output_dir):
    """Train a single agent (one algo × one scenario × one seed)."""
    tag = f"{algo_name}_{scenario_name}_seed{seed}"
    save_path = os.path.join(output_dir, f"{tag}_best.zip")

    env = FlexFlowSimEnv(config=config_path, weights=weights, seed=seed)
    steps_per_ep = int(env._max_time / env._dt)
    total_timesteps = total_episodes * steps_per_ep

    AlgoClass = ALGO_MAP[algo_name]
    hparams = get_hyperparams(algo_name, steps_per_ep)

    model = AlgoClass("MlpPolicy", env, verbose=0, seed=seed, **hparams)
    callback = EpisodeTracker(save_path=save_path, verbose=1)

    print(f"\n  Training {tag} ({total_episodes} eps × {steps_per_ep} steps = "
          f"{total_timesteps} timesteps)")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0

    # Save final model too
    final_path = os.path.join(output_dir, f"{tag}_final.zip")
    model.save(final_path)

    # Learning curve
    fig, ax = plt.subplots(figsize=(8, 4))
    eps = np.arange(1, len(callback.episode_rewards) + 1)
    rewards = np.array(callback.episode_rewards)
    ma = pd.Series(rewards).rolling(20, min_periods=1).mean().values
    ax.plot(eps, rewards, alpha=0.3, label="Episode")
    ax.plot(eps, ma, linewidth=2, label="MA(20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(tag)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{tag}_curve.png"), dpi=150)
    plt.close(fig)

    log = {
        "algo": algo_name,
        "scenario": scenario_name,
        "weights": list(weights),
        "seed": seed,
        "total_episodes": callback.episode_count,
        "best_episode": callback.best_episode,
        "best_reward": float(callback.best_reward),
        "training_time_s": elapsed,
        "episode_rewards": [float(r) for r in callback.episode_rewards],
    }

    print(f"    Done in {elapsed:.0f}s — Best: Ep {callback.best_episode} "
          f"(reward={callback.best_reward:.3f})")
    return log


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FlexFlowSim Multi-Seed Training")
    parser.add_argument("--config", type=str, required=True, help="Path to env config JSON")
    parser.add_argument("--algo", type=str, nargs="+", default=["DQN", "PPO"],
                        choices=["DQN", "PPO"], help="Algorithm(s) to train")
    parser.add_argument("--scenario", type=str, nargs="+", default=None,
                        help="Scenario(s) to train (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds for multi-seed protocol")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Training episodes per run")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    # Validate config
    cfg = load_config(args.config)
    scenario_names = args.scenario or list(SCENARIOS.keys())
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("  FlexFlowSim — Multi-Seed Training")
    print("=" * 70)
    print(f"  Config: {args.config}")
    print(f"  Algorithms: {args.algo}")
    print(f"  Scenarios: {scenario_names}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Output: {args.output}")

    all_logs = []
    t_total = time.time()

    for algo_name in args.algo:
        for scenario_name in scenario_names:
            weights = SCENARIOS[scenario_name]["weights"]
            for seed in args.seeds:
                log = train_single(
                    config_path=args.config,
                    algo_name=algo_name,
                    scenario_name=scenario_name,
                    weights=weights,
                    seed=seed,
                    total_episodes=args.episodes,
                    output_dir=args.output,
                )
                all_logs.append(log)

    # Save consolidated log
    log_path = os.path.join(args.output, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2, default=str)

    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  ALL COMPLETE — {elapsed_total / 60:.1f} min, {len(all_logs)} runs")
    print(f"  Log: {log_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
