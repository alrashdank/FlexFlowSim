"""
FlexFlowSim — Evaluation & Statistical Analysis
==================================================

Evaluates trained agents and baselines with multi-seed replication,
Kruskal-Wallis testing, Dunn's post-hoc, and Cliff's Delta effect sizes.

Usage:
    python evaluate.py --config configs/bakery_bk50.json --agents-dir results/
    python evaluate.py --config configs/bakery_bk50.json --baselines-only --reps 50
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from env import FlexFlowSimEnv, load_config
from baselines import BASELINE_POLICIES, run_episode


def cliffs_delta(x, y):
    """Cliff's Delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def cliffs_delta_interpretation(d):
    d = abs(d)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


def evaluate_baselines(config_path, weights, num_reps=50, seed=42):
    """Run all baselines for num_reps replications."""
    env = FlexFlowSimEnv(config=config_path, weights=weights, seed=seed)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=num_reps)

    all_results = {}
    for name, PolicyClass in BASELINE_POLICIES.items():
        policy = PolicyClass(env=env)
        results = [run_episode(policy, env, int(s)) for s in seeds]
        all_results[name] = pd.DataFrame(results)

    return all_results


def evaluate_agent(config_path, agent_path, algo_name, weights, num_reps=50, seed=42):
    """Evaluate a trained RL agent."""
    from stable_baselines3 import DQN, PPO
    ALGOS = {"DQN": DQN, "PPO": PPO}

    env = FlexFlowSimEnv(config=config_path, weights=weights, seed=seed)
    model = ALGOS[algo_name].load(agent_path)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=num_reps)

    results = []
    for s in seeds:
        obs, info = env.reset(seed=int(s))
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        dep = info["total_departed"]
        row = {
            "totalCost": info["total_cost"],
            "totalDeparted": dep,
            "costPerUnit": info["total_cost"] / max(dep, 1),
            "avgLeadTime": info["avg_lead_time"],
            "processingCost": info["processing_cost"],
            "idleCost": info["idle_cost"],
            "waitingCost": info["waiting_cost"],
            "totalReward": total_reward,
        }
        for i, u in enumerate(info["utilisation"]):
            row[f"util_{i}"] = u
        results.append(row)

    return pd.DataFrame(results)


def statistical_comparison(all_results, metric="totalCost"):
    """Kruskal-Wallis + Dunn's post-hoc with Holm correction."""
    names = list(all_results.keys())
    groups = [all_results[n][metric].values for n in names]

    # Normality tests
    print(f"\n  Normality tests (Shapiro-Wilk) for '{metric}':")
    for name, g in zip(names, groups):
        if len(g) >= 3:
            w, p = stats.shapiro(g)
            print(f"    {name:20s}: W={w:.4f}, p={p:.4f} "
                  f"{'✓' if p > 0.05 else '✗'}")

    # Kruskal-Wallis
    if len(groups) >= 2:
        h, p_kw = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis: H={h:.2f}, p={p_kw:.6f}")

        if p_kw < 0.05:
            print(f"\n  Pairwise Cliff's Delta ({metric}):")
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    d = cliffs_delta(groups[i], groups[j])
                    interp = cliffs_delta_interpretation(d)
                    print(f"    {names[i]:20s} vs {names[j]:20s}: "
                          f"d={d:+.3f} ({interp})")
        else:
            print("  No significant difference (p >= 0.05)")

    return names, groups


def summary_table(all_results):
    """Print a summary table of mean ± std for key metrics."""
    metrics = ["totalCost", "totalDeparted", "costPerUnit", "avgLeadTime", "totalReward"]
    names = list(all_results.keys())

    print(f"\n{'Algorithm':20s}", end="")
    for m in metrics:
        label = m[:12]
        print(f"  {label:>22s}", end="")
    print()
    print("-" * (20 + 24 * len(metrics)))

    for name in names:
        df = all_results[name]
        print(f"{name:20s}", end="")
        for m in metrics:
            mean = df[m].mean()
            std = df[m].std()
            print(f"  {mean:10.1f} ± {std:7.1f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="FlexFlowSim Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agents-dir", type=str, default=None,
                        help="Directory with trained agent .zip files")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--scenario", type=str, default="Balanced")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    SCENARIOS = {
        "CostFocus": (0.8, 0.1, 0.1),
        "ThroughputFocus": (0.1, 0.8, 0.1),
        "LeadTimeFocus": (0.1, 0.1, 0.8),
        "Balanced": (0.33, 0.33, 0.34),
    }
    weights = SCENARIOS.get(args.scenario, (0.33, 0.33, 0.34))

    print("=" * 70)
    print(f"  FlexFlowSim Evaluation — {args.scenario} (w={weights})")
    print("=" * 70)

    # Baselines
    print("\n  Running baselines...")
    all_results = evaluate_baselines(args.config, weights, args.reps, args.seed)
    print(f"  {len(all_results)} baselines × {args.reps} reps done")

    # Agents
    if args.agents_dir and not args.baselines_only:
        for fname in sorted(os.listdir(args.agents_dir)):
            if fname.endswith("_best.zip") and args.scenario in fname:
                algo = fname.split("_")[0]
                seed_str = fname.split("seed")[1].split("_")[0]
                label = f"{algo}_seed{seed_str}"
                path = os.path.join(args.agents_dir, fname)
                print(f"  Evaluating {label}...")
                df = evaluate_agent(args.config, path, algo, weights, args.reps, args.seed)
                all_results[label] = df

    # Summary
    summary_table(all_results)

    # Statistical tests
    for metric in ["totalCost", "totalDeparted", "avgLeadTime"]:
        print(f"\n{'=' * 70}")
        print(f"  Statistical Analysis: {metric}")
        print(f"{'=' * 70}")
        statistical_comparison(all_results, metric)

    # Save
    output_path = f"eval_{args.scenario}.csv"
    rows = []
    for name, df in all_results.items():
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict["algorithm"] = name
            rows.append(row_dict)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
