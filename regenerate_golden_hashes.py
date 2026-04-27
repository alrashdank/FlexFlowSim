"""
Paper 5 — Master Sweep Script
==============================

Runs the full experimental grid for Paper 5:

  PHASE 1 (fast, ~10 min):
    - Baseline sweep on electronics (5 rules + Vanilla TS + HBQ)
    - All 6 configs (stationary + 5 breakdown levels) x 50 seeds

  PHASE 2 (slow, ~3 hr):
    - Vanilla PPO on bakery breakdowns
    - 5 configs x 5 seeds x 500 episodes

  PHASE 3 (slow, ~3 hr):
    - Vanilla PPO on electronics breakdowns
    - 5 configs x 5 seeds x 500 episodes

  PHASE 4 (slow, ~3 hr):
    - NS-aware PPO on electronics breakdowns
    - 5 configs x 5 seeds x 500 episodes

Total: ~10 hr. Run overnight. Resumable: skip phases that have output already.

Usage (run from the repository root):
    python paper5/run_sweep.py                 # all phases
    python paper5/run_sweep.py --phase 1       # only baselines
    python paper5/run_sweep.py --phase 2 3 4   # only PPO phases
    python paper5/run_sweep.py --skip-existing # resume
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure the repository root is on sys.path so that `import env` and
# `import baselines` work when this script is invoked as
# ``python paper5/run_sweep.py`` from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Anchor CWD so that all relative paths below (configs/, results/) resolve
# inside the repo regardless of the user's working directory.
os.chdir(REPO_ROOT)

from env import FlexFlowSimEnv
from baselines import (BASELINE_POLICIES, run_episode,
                       VanillaThompsonSampling, HybridBanditQueue)


def run_bandit_episode(cfg_path, bandit_cls, seed, **kw):
    env = FlexFlowSimEnv(config=cfg_path, weights=(1.0, 0.0, 0.0), seed=seed)
    b = bandit_cls(env=env, seed=seed, **kw)
    b.reset()
    obs, info = env.reset(seed=seed)
    while True:
        a = b.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        route = env._action_tuples[a]
        for si, sj in enumerate(route):
            fi = env._flat_idx[(si, sj)]
            c = (min(env._in_service[fi], 1.0)
                 * env._processing_cost[fi] * env._dt)
            b.update(si, sj, -c)
        if term or trunc:
            break
    dep = info["total_departed"]
    result = {
        "totalCost": info["total_cost"],
        "totalDeparted": dep,
        "costPerUnit": info["total_cost"] / max(dep, 1),
        "avgLeadTime": info["avg_lead_time"],
        "processingCost": info["processing_cost"],
        "idleCost": info["idle_cost"],
        "waitingCost": info["waiting_cost"],
    }
    if "breakdown_count" in info:
        result["breakdownCount"] = sum(info["breakdown_count"])
        result["breakdownTime"] = sum(info["breakdown_time"])
    return result


# ═══════════════════════════════════════════════════════════════════
# CONFIG DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def get_bakery_breakdown_configs():
    base = [os.path.join("configs", "bakery_bk50.json")]
    ns = sorted(glob.glob("configs/paper5_ns/bakery_breakdowns_*.json"))
    return base + ns


def get_electronics_breakdown_configs():
    base = [os.path.join("configs", "electronics_3stage.json")]
    ns = sorted(glob.glob("configs/paper5_electronics/electronics_breakdowns_*.json"))
    return base + ns


def get_electronics_nsaware_configs():
    return sorted(glob.glob("configs/paper5_electronics_nsaware/*.json"))


def extract_severity(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    meta = cfg.get("_metadata", {})
    return meta.get("availability_target", 1.0)


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: BASELINES
# ═══════════════════════════════════════════════════════════════════

def phase1_baselines(args):
    print("\n" + "=" * 70)
    print("  PHASE 1: BASELINES (rules + Vanilla TS + HBQ)")
    print("=" * 70)
    out_dir = "results/paper5_phase1_baselines"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "baseline_sweep.csv")

    if args.skip_existing and os.path.exists(csv_path):
        print(f"  Skipping: {csv_path} exists")
        return

    rule_names = ["RoundRobin", "Random", "ShortestQueue",
                  "LeastUtilised", "CostMinimising"]
    all_configs = (
        [("bakery", p) for p in get_bakery_breakdown_configs()]
        + [("electronics", p) for p in get_electronics_breakdown_configs()]
    )

    rows = []
    t0 = time.time()
    for ci, (env_name, cfg_path) in enumerate(all_configs):
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        sev = extract_severity(cfg_path)
        print(f"  [{ci+1}/{len(all_configs)}] {env_name}: {cfg_name}")

        # Rules
        for rule in rule_names:
            for seed in range(args.seeds):
                env = FlexFlowSimEnv(config=cfg_path,
                                     weights=(1.0, 0.0, 0.0), seed=seed)
                pol = BASELINE_POLICIES[rule](env=env)
                res = run_episode(pol, env, seed)
                res.update({"env": env_name, "config": cfg_name,
                            "severity": sev, "method": rule,
                            "category": "rule", "seed": seed})
                rows.append(res)

        # Vanilla TS
        for seed in range(args.seeds):
            res = run_bandit_episode(cfg_path, VanillaThompsonSampling, seed)
            res.update({"env": env_name, "config": cfg_name,
                        "severity": sev, "method": "VanillaTS",
                        "category": "bandit", "seed": seed})
            rows.append(res)

        # HBQ
        for seed in range(args.seeds):
            res = run_bandit_episode(cfg_path, HybridBanditQueue,
                                     seed, load_weight=20.0)
            res.update({"env": env_name, "config": cfg_name,
                        "severity": sev, "method": "HBQ",
                        "category": "hybrid", "seed": seed})
            rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    summary = df.groupby(["env", "severity", "method"])["costPerUnit"].agg(
        ["mean", "std", "count"]).round(2)
    summary.to_csv(os.path.join(out_dir, "baseline_summary.csv"))

    elapsed = time.time() - t0
    print(f"  Phase 1 done: {elapsed/60:.1f} min, {len(rows)} episodes")
    print(f"  CSV: {csv_path}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 2-4: PPO TRAINING
# ═══════════════════════════════════════════════════════════════════

def run_ppo_phase(phase_label, configs, output_root, args):
    """Train PPO on a list of configs with multiple seeds."""
    from train import train_single, SCENARIOS

    print("\n" + "=" * 70)
    print(f"  {phase_label}")
    print("=" * 70)

    weights = SCENARIOS["CostFocus"]["weights"]
    all_train = []
    all_eval = []
    t0 = time.time()

    for ci, cfg_path in enumerate(configs):
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        sev = extract_severity(cfg_path)
        cfg_out = os.path.join(output_root, cfg_name)
        os.makedirs(cfg_out, exist_ok=True)

        # Skip if already trained
        seeds_to_run = []
        for seed in args.ppo_seeds:
            best_path = os.path.join(
                cfg_out, f"PPO_CostFocus_seed{seed}_best.zip"
            )
            if args.skip_existing and os.path.exists(best_path):
                print(f"  [{ci+1}/{len(configs)}] {cfg_name} seed={seed}: skipping (exists)")
            else:
                seeds_to_run.append(seed)

        for seed in seeds_to_run:
            print(f"  [{ci+1}/{len(configs)}] {cfg_name} seed={seed}")
            log = train_single(
                config_path=cfg_path,
                algo_name="PPO",
                scenario_name="CostFocus",
                weights=weights,
                seed=seed,
                total_episodes=args.ppo_episodes,
                output_dir=cfg_out,
            )
            log["config"] = cfg_name
            log["severity"] = sev
            all_train.append(log)

        # Evaluate all available models
        from stable_baselines3 import PPO
        for seed in args.ppo_seeds:
            best_path = os.path.join(
                cfg_out, f"PPO_CostFocus_seed{seed}_best.zip"
            )
            if not os.path.exists(best_path):
                continue
            for eval_seed in range(args.eval_episodes):
                env = FlexFlowSimEnv(config=cfg_path, weights=weights,
                                     seed=eval_seed + seed * 1000)
                model = PPO.load(best_path, env=env)
                obs, info = env.reset(seed=eval_seed + seed * 1000)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = env.step(int(action))
                    done = term or trunc
                dep = info["total_departed"]
                row = {
                    "config": cfg_name, "severity": sev,
                    "method": "PPO", "trainSeed": seed,
                    "evalSeed": eval_seed,
                    "totalCost": info["total_cost"],
                    "totalDeparted": dep,
                    "costPerUnit": info["total_cost"] / max(dep, 1),
                    "avgLeadTime": info["avg_lead_time"],
                }
                if "breakdown_count" in info:
                    row["breakdownCount"] = sum(info["breakdown_count"])
                all_eval.append(row)

    train_path = os.path.join(output_root, "ppo_training_log.json")
    with open(train_path, "w") as f:
        json.dump(all_train, f, indent=2, default=str)

    if all_eval:
        df = pd.DataFrame(all_eval)
        eval_path = os.path.join(output_root, "ppo_eval_sweep.csv")
        df.to_csv(eval_path, index=False)
        summary = df.groupby("config")["costPerUnit"].agg(
            ["mean", "std", "count"]).round(2)
        summary.to_csv(os.path.join(output_root, "ppo_eval_summary.csv"))
        print(f"  Eval CSV: {eval_path}")

    elapsed = time.time() - t0
    print(f"  {phase_label} done: {elapsed/60:.1f} min, "
          f"{len(all_train)} new training runs")


def phase2_ppo_bakery(args):
    configs = get_bakery_breakdown_configs()
    run_ppo_phase("PHASE 2: VANILLA PPO ON BAKERY BREAKDOWNS",
                  configs, "results/paper5_phase2_ppo_bakery", args)


def phase3_ppo_electronics(args):
    configs = get_electronics_breakdown_configs()
    run_ppo_phase("PHASE 3: VANILLA PPO ON ELECTRONICS BREAKDOWNS",
                  configs, "results/paper5_phase3_ppo_electronics", args)


def phase4_ppo_nsaware_electronics(args):
    configs = get_electronics_nsaware_configs()
    if not configs:
        print("\n  PHASE 4 SKIPPED: no configs in configs/paper5_electronics_nsaware/")
        print("  Run: python -c \"from step6_full_paper5_sweep import generate_nsaware_electronics; generate_nsaware_electronics()\"")
        return
    run_ppo_phase("PHASE 4: NS-AWARE PPO ON ELECTRONICS BREAKDOWNS",
                  configs, "results/paper5_phase4_ppo_nsaware_electronics", args)


def generate_nsaware_electronics():
    """Generate ns_aware variants of electronics breakdown configs."""
    src_dir = "configs/paper5_electronics"
    dst_dir = "configs/paper5_electronics_nsaware"
    os.makedirs(dst_dir, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        with open(src) as f:
            cfg = json.load(f)
        cfg["ns_features"] = {"enabled": True, "ema_window": 30}
        name = os.path.basename(src).replace(".json", "_nsaware.json")
        with open(os.path.join(dst_dir, name), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Wrote {name}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Paper 5 master sweep")
    parser.add_argument("--phase", type=int, nargs="+",
                        default=[1, 2, 3, 4],
                        help="Phases to run (default: all)")
    parser.add_argument("--seeds", type=int, default=50,
                        help="Eval seeds for baselines (default: 50)")
    parser.add_argument("--ppo-seeds", type=int, nargs="+",
                        default=[42, 123, 256, 512, 1024],
                        help="Training seeds for PPO")
    parser.add_argument("--ppo-episodes", type=int, default=500,
                        help="Episodes per PPO run")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Eval episodes per trained PPO seed")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip phases/runs with existing output")
    args = parser.parse_args()

    print("=" * 70)
    print("  PAPER 5 — MASTER SWEEP")
    print("=" * 70)
    print(f"  Phases: {args.phase}")
    print(f"  Eval seeds (baselines): {args.seeds}")
    print(f"  PPO seeds: {args.ppo_seeds}")
    print(f"  PPO episodes: {args.ppo_episodes}")
    print(f"  PPO eval episodes: {args.eval_episodes}")
    print(f"  Skip existing: {args.skip_existing}")

    # Generate ns-aware electronics configs if phase 4 requested
    if 4 in args.phase:
        if not glob.glob("configs/paper5_electronics_nsaware/*.json"):
            print("\n  Generating ns_aware electronics configs...")
            generate_nsaware_electronics()

    t_total = time.time()
    if 1 in args.phase: phase1_baselines(args)
    if 2 in args.phase: phase2_ppo_bakery(args)
    if 3 in args.phase: phase3_ppo_electronics(args)
    if 4 in args.phase: phase4_ppo_nsaware_electronics(args)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE — {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
