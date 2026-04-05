"""
FlexFlowSim — Normalisation Constant Calibration
==================================================

Runs random-policy episodes with norm_constants=[1,1,1] to measure raw
reward component magnitudes. Use the output to set norm_constants in
your JSON config before training.

Usage:
    python calibrate.py --config configs/bakery_bk50.json
    python calibrate.py --config configs/bakery_bk50.json --episodes 10
"""

import argparse
import json
import numpy as np
from env import FlexFlowSimEnv, load_config


def calibrate(config_path, num_episodes=5, seed=42):
    """Run random-policy episodes to measure raw reward scales."""
    # Load config and override norm_constants
    cfg = load_config(config_path)
    cfg["norm_constants"] = [1.0, 1.0, 1.0]

    env = FlexFlowSimEnv(config=cfg, weights=(0.33, 0.33, 0.34), seed=seed)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=num_episodes)

    costs, deps, lead_times = [], [], []

    print("=" * 65)
    print("  FlexFlowSim Calibration — Measuring Reward Component Scales")
    print("=" * 65)
    print(env.describe())
    print(f"\nRunning {num_episodes} random-policy episodes...\n")

    for i in range(num_episodes):
        obs, info = env.reset(seed=int(seeds[i]))
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        costs.append(info["total_cost"])
        deps.append(info["total_departed"])
        lead_times.append(info["avg_lead_time"])
        print(f"  Episode {i+1}/{num_episodes} (seed={seeds[i]}): "
              f"Cost={info['total_cost']:.0f}, Dep={info['total_departed']}, "
              f"AvgLT={info['avg_lead_time']:.1f}")

    mean_cost = np.mean(costs)
    mean_dep = np.mean(deps)
    mean_lt = np.mean(lead_times)

    # NormConstants: round to clean values
    ref_cost = max(round(mean_cost, -1), 1.0)
    ref_tp = max(round(mean_dep, 0), 1.0)
    est_wip_integral = mean_dep * mean_lt
    ref_wip = max(round(est_wip_integral, -1), 1.0)

    print(f"\n{'=' * 65}")
    print(f"  RESULTS (averaged over {num_episodes} episodes)")
    print(f"{'=' * 65}")
    print(f"  Mean Total Cost:      {mean_cost:.1f}")
    print(f"  Mean Total Departed:  {mean_dep:.1f}")
    print(f"  Mean Avg Lead Time:   {mean_lt:.1f}")
    print(f"  Est. WIP Integral:    {est_wip_integral:.1f}")

    print(f"\n  RECOMMENDED norm_constants: [{ref_cost:.0f}, {ref_tp:.0f}, {ref_wip:.0f}]")
    print(f"\n  Verification (each should be ~1.0):")
    print(f"    |cumCost / Ref_Cost|   = {mean_cost / ref_cost:.3f}")
    print(f"    |cumDep / Ref_Tp|      = {mean_dep / ref_tp:.3f}")
    print(f"    |cumWIP_dt / Ref_WIP|  = {est_wip_integral / ref_wip:.3f}")

    # Optionally update the config file
    print(f"\n  To update your config:")
    print(f'    "norm_constants": [{ref_cost:.0f}, {ref_tp:.0f}, {ref_wip:.0f}]')
    print(f"{'=' * 65}")

    return ref_cost, ref_tp, ref_wip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate FlexFlowSim norm constants")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    calibrate(args.config, args.episodes, args.seed)
