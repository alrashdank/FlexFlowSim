"""Regenerate the manuscript's tables from the CSV outputs in paper5/results/.

Verifies, for the reader, that the cell values reported in Tables 2, 3, 5,
6, and 7 of the paper are exactly what the CSV files contain.

Usage:
    python paper5/analyse_results.py

Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load() -> dict[str, pd.DataFrame]:
    return {
        "baselines": pd.read_csv(RESULTS / "baseline_sweep.csv"),
        "ppo_naive": pd.read_csv(RESULTS / "ppo_eval_sweep.csv"),
        "ppo_corrected": pd.read_csv(RESULTS / "ppo_fixed_eval_sweep.csv"),
    }


# ──────────────────────────────────────────────────────────────────────
def table2_electronics_cpu(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Manuscript Table 2: cost per unit by (severity × method) on electronics."""
    base = d["baselines"]
    elec = base[base["env"] == "electronics"]
    methods = ["ShortestQueue", "HBQ", "LeastUtilised", "RoundRobin",
               "VanillaTS", "CostMinimising", "Random"]
    sev_order = ["electronics_3stage",
                 "electronics_breakdowns_A99",
                 "electronics_breakdowns_A95",
                 "electronics_breakdowns_A90",
                 "electronics_breakdowns_A80",
                 "electronics_breakdowns_A70"]
    sev_label = {"electronics_3stage": "Stationary",
                 "electronics_breakdowns_A99": "A=0.99",
                 "electronics_breakdowns_A95": "A=0.95",
                 "electronics_breakdowns_A90": "A=0.90",
                 "electronics_breakdowns_A80": "A=0.80",
                 "electronics_breakdowns_A70": "A=0.70"}
    rows = []
    for cfg in sev_order:
        row = {"Severity": sev_label[cfg]}
        for m in methods:
            sub = elec[(elec["config"] == cfg) & (elec["method"] == m)]["costPerUnit"]
            row[m] = f"{sub.mean():.1f} ± {sub.std():.1f}" if len(sub) else "-"
        # PPO corrected
        ppo = d["ppo_corrected"]
        cpu_label = cfg.replace("3stage", "stationary") + "_cpu"
        sub = ppo[ppo["config"] == cpu_label]["costPerUnit"]
        row["PPO (corr.)"] = f"{sub.mean():.1f} ± {sub.std():.1f}" if len(sub) else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def table3_electronics_throughput(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = d["baselines"]
    elec = base[base["env"] == "electronics"]
    sev_order = ["electronics_3stage",
                 "electronics_breakdowns_A99",
                 "electronics_breakdowns_A95",
                 "electronics_breakdowns_A90",
                 "electronics_breakdowns_A80",
                 "electronics_breakdowns_A70"]
    sev_label = {"electronics_3stage": "Stationary",
                 "electronics_breakdowns_A99": "A=0.99",
                 "electronics_breakdowns_A95": "A=0.95",
                 "electronics_breakdowns_A90": "A=0.90",
                 "electronics_breakdowns_A80": "A=0.80",
                 "electronics_breakdowns_A70": "A=0.70"}
    rows = []
    for cfg in sev_order:
        sq = elec[(elec["config"] == cfg) & (elec["method"] == "ShortestQueue")]["totalDeparted"].mean()
        hbq = elec[(elec["config"] == cfg) & (elec["method"] == "HBQ")]["totalDeparted"].mean()
        cpu_label = cfg.replace("3stage", "stationary") + "_cpu"
        ppo = d["ppo_corrected"][d["ppo_corrected"]["config"] == cpu_label]["totalDeparted"].mean()
        gap = (ppo - sq) / sq * 100 if sq else float("nan")
        rows.append({"Severity": sev_label[cfg],
                     "ShortestQueue": f"{sq:.1f}",
                     "HBQ": f"{hbq:.1f}",
                     "PPO (corr.)": f"{ppo:.1f}",
                     "PPO gap": f"{gap:+.1f}%"})
    return pd.DataFrame(rows)


def table4_per_seed_ppo(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-seed mean CPU under corrected PPO, electronics testbed.

    Display rule matches Table 4 in the manuscript: any cell whose unrounded
    mean is below 200 but rounds to 200.0 at 1dp is shown with two decimals
    and a trailing asterisk, so the reader does not mistake it for a failure
    (the failure threshold is strictly > 200).
    """
    ppo = d["ppo_corrected"]
    elec = ppo[ppo["config"].str.startswith("electronics_") & ppo["config"].str.endswith("_cpu")]
    sev_order = ["electronics_stationary_cpu",
                 "electronics_breakdowns_A99_cpu",
                 "electronics_breakdowns_A95_cpu",
                 "electronics_breakdowns_A90_cpu",
                 "electronics_breakdowns_A80_cpu",
                 "electronics_breakdowns_A70_cpu"]
    sev_label = {"electronics_stationary_cpu": "Stationary",
                 "electronics_breakdowns_A99_cpu": "A=0.99",
                 "electronics_breakdowns_A95_cpu": "A=0.95",
                 "electronics_breakdowns_A90_cpu": "A=0.90",
                 "electronics_breakdowns_A80_cpu": "A=0.80",
                 "electronics_breakdowns_A70_cpu": "A=0.70"}
    seeds = [42, 123, 256, 512, 1024]

    def fmt(m):
        # If the unrounded value is below 200 but would round up to 200.0 at
        # 1dp, show two decimals and flag with asterisk.
        if m < 200.0 and round(m, 1) >= 200.0:
            return f"{m:.2f}*"
        return f"{m:.1f}"

    rows = []
    for cfg in sev_order:
        row = {"Severity": sev_label[cfg]}
        seed_means = []
        for s in seeds:
            sub = elec[(elec["config"] == cfg) & (elec["trainSeed"] == s)]["costPerUnit"]
            m = sub.mean()
            seed_means.append(m)
            row[f"Seed {s}"] = fmt(m)
        row["Median"] = fmt(float(np.median(seed_means)))
        rows.append(row)
    return pd.DataFrame(rows)


def table5_bakery(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = d["baselines"]
    bk = base[base["env"] == "bakery"]
    sev_order = ["bakery_bk50",
                 "bakery_breakdowns_A99",
                 "bakery_breakdowns_A95",
                 "bakery_breakdowns_A90",
                 "bakery_breakdowns_A80",
                 "bakery_breakdowns_A70"]
    sev_label = {"bakery_bk50": "Stationary",
                 "bakery_breakdowns_A99": "A=0.99",
                 "bakery_breakdowns_A95": "A=0.95",
                 "bakery_breakdowns_A90": "A=0.90",
                 "bakery_breakdowns_A80": "A=0.80",
                 "bakery_breakdowns_A70": "A=0.70"}
    methods = ["ShortestQueue", "HBQ", "RoundRobin", "VanillaTS"]
    rows = []
    for cfg in sev_order:
        row = {"Severity": sev_label[cfg]}
        for m in methods:
            sub = bk[(bk["config"] == cfg) & (bk["method"] == m)]["costPerUnit"]
            row[m] = f"{sub.mean():.1f} ± {sub.std():.1f}" if len(sub) else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def table6_protocol_sensitivity(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = d["baselines"]
    elec = base[base["env"] == "electronics"]
    naive = d["ppo_naive"]
    corrected = d["ppo_corrected"]
    sev_order = [("A=0.99", "electronics_breakdowns_A99"),
                 ("A=0.95", "electronics_breakdowns_A95"),
                 ("A=0.90", "electronics_breakdowns_A90"),
                 ("A=0.80", "electronics_breakdowns_A80"),
                 ("A=0.70", "electronics_breakdowns_A70")]
    rows = []
    for label, cfg in sev_order:
        sq = elec[(elec["config"] == cfg) & (elec["method"] == "ShortestQueue")]["costPerUnit"].mean()
        n = naive[naive["config"] == cfg + "_nsaware"]["costPerUnit"].mean()
        c = corrected[corrected["config"] == cfg + "_cpu"]["costPerUnit"].mean()
        ng = (n - sq) / sq * 100
        cg = (c - sq) / sq * 100
        rows.append({"Severity": label,
                     "Naive PPO": f"{n:.1f}",
                     "Corrected PPO": f"{c:.1f}",
                     "ShortestQueue": f"{sq:.1f}",
                     "Naive gap": f"{ng:+.1f}%",
                     "Corrected gap": f"{cg:+.1f}%",
                     "Reduction": f"{ng - cg:.1f} pp"})
    return pd.DataFrame(rows)


def table7_training_failures(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-protocol training-failure counts (per-seed mean CPU > 200)."""
    naive = d["ppo_naive"]
    corrected = d["ppo_corrected"]
    sev_order = [("A=0.99", "electronics_breakdowns_A99"),
                 ("A=0.95", "electronics_breakdowns_A95"),
                 ("A=0.90", "electronics_breakdowns_A90"),
                 ("A=0.80", "electronics_breakdowns_A80"),
                 ("A=0.70", "electronics_breakdowns_A70")]
    rows = []
    for label, cfg in sev_order:
        n_fail = (naive[naive["config"] == cfg + "_nsaware"]
                  .groupby("trainSeed")["costPerUnit"].mean() > 200).sum()
        c_fail = (corrected[corrected["config"] == cfg + "_cpu"]
                  .groupby("trainSeed")["costPerUnit"].mean() > 200).sum()
        rows.append({"Severity": label,
                     "Naive failures": f"{n_fail} / 5",
                     "Corrected failures": f"{c_fail} / 5"})
    rows.append({"Severity": "Total",
                 "Naive failures": f"{sum(int(r['Naive failures'].split()[0]) for r in rows)} / 25",
                 "Corrected failures": f"{sum(int(r['Corrected failures'].split()[0]) for r in rows)} / 25"})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.expand_frame_repr", False)

    print("=" * 80)
    print("TABLE 2 — Cost per unit, electronics testbed")
    print("=" * 80)
    print(table2_electronics_cpu(d).to_string(index=False))

    print()
    print("=" * 80)
    print("TABLE 3 — Throughput, electronics testbed")
    print("=" * 80)
    print(table3_electronics_throughput(d).to_string(index=False))

    print()
    print("=" * 80)
    print("TABLE 4 — Per-seed mean CPU under corrected PPO, electronics")
    print("=" * 80)
    print(table4_per_seed_ppo(d).to_string(index=False))

    print()
    print("=" * 80)
    print("TABLE 5 — Cost per unit, bakery testbed (leading methods)")
    print("=" * 80)
    print(table5_bakery(d).to_string(index=False))

    print()
    print("=" * 80)
    print("TABLE 6 — Protocol sensitivity: PPO gap to ShortestQueue")
    print("=" * 80)
    print(table6_protocol_sensitivity(d).to_string(index=False))

    print()
    print("=" * 80)
    print("TABLE 7 — Training failures by protocol")
    print("=" * 80)
    print(table7_training_failures(d).to_string(index=False))


if __name__ == "__main__":
    main()
