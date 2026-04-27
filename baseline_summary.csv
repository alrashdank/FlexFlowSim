# Reproducing Paper 5

This folder contains the code and data artefacts underlying

> **Alrashdan, K. R. (2026). Routing under machine breakdowns: a benchmark
> of dispatching rules, bandits, and reinforcement learning for
> multi-server flow shops.**

The paper extends FlexFlowSim with three opt-in non-stationarity
mechanisms (machine breakdowns, demand surges, processing-cost drift)
and benchmarks five dispatching rules, two online bandits, and two PPO
configurations on a multi-server flow shop subject to machine
breakdowns at five severity levels.

## What's in this folder

```
paper5/
├── README.md                          # this file
├── analyse_results.py                 # regenerates Tables 2-7 from the CSVs
└── results/
    ├── baseline_sweep.csv             # 4,200 rows: 7 baseline methods × 12 configs × 50 seeds
    ├── baseline_summary.csv           # aggregated baselines
    ├── ppo_eval_sweep.csv             # 1,250 rows: naive PPO × 5 breakdown configs × 5 seeds × 50 evals
    ├── ppo_eval_summary.csv           # aggregated naive PPO
    ├── ppo_fixed_eval_sweep.csv       # 1,500 rows: corrected PPO × 6 configs × 5 seeds × 50 evals
    └── ppo_fixed_eval_summary.csv     # aggregated corrected PPO
```

## Mapping CSV → manuscript tables

| Manuscript table | CSV file(s) | Filter |
|---|---|---|
| Table 1 (env. parameters) | `../configs/*.json` | bakery_bk50.json, electronics_3stage.json |
| Table 2 (CPU comparison, electronics) | `baseline_sweep.csv` + `ppo_fixed_eval_sweep.csv` | `env == 'electronics'` |
| Table 3 (throughput, electronics) | same | same |
| Table 4 (per-seed PPO, electronics) | `ppo_fixed_eval_sweep.csv` | `config.startswith('electronics_')` |
| Table 5 (bakery cost-per-unit) | `baseline_sweep.csv` | `env == 'bakery'` |
| Table 6 (protocol sensitivity) | `ppo_eval_sweep.csv` (naive) and `ppo_fixed_eval_sweep.csv` (corrected) | matched on severity |
| Table 7 (training failures) | `ppo_eval_sweep.csv` and `ppo_fixed_eval_sweep.csv` | per-seed mean CPU |

Configuration naming convention used in the CSVs:

- `<testbed>_<config>` (e.g. `electronics_3stage`, `bakery_breakdowns_A90`) for rules and bandits
- `<testbed>_<config>_nsaware` for naive PPO (NS-aware observation features, default training protocol)
- `<testbed>_<config>_cpu` for corrected PPO (NS-aware features + CPU-aligned reward + raised γ + eval-cost checkpointing)

## How to verify the byte-identical claim (§3.1)

```bash
pytest tests/test_byte_identical.py -v
```

40 cases, runs in ~3 seconds. Verifies that the patched `env.py` reproduces
the unpatched Paper 3 / Paper 4 simulator's outputs exactly when run on a
stationary configuration.

## How to regenerate the table numbers

```bash
python paper5/analyse_results.py
```

Reads the CSVs and prints the cell values for Tables 2, 3, 5, and 6. Useful
as a sanity check that the CSVs match the manuscript's reported numbers.

## What's NOT in this folder

The orchestration scripts that drove the original sweep (`run_baselines_sweep.py`,
`run_ppo_naive_sweep.py`, `run_ppo_corrected_sweep.py`) were used on the
author's local machine and are not bundled here. They iterate over the 12
configurations and call `train.py` and `evaluate.py` for each. A reviewer
can re-run any single (config, method) combination directly:

```bash
# Baseline rules and bandits on a single config
python evaluate.py --config configs/electronics_breakdowns_A90.json --baselines-only --reps 50

# Train PPO with the corrected protocol on a single config
# (requires adding cpu_aligned_reward: true and ns_features.enabled: true to the config)
python train.py --config configs/electronics_breakdowns_A90.json --algo PPO --episodes 500 --seeds 42 123 256 512 1024
```

If a reviewer wants the full sweep scripts, please open an issue on the
GitHub repository or contact the author directly.

## Reference

If you use these artefacts, please cite the paper. The simulator itself
predates Paper 5 and is also citeable independently as:

> Alrashdan, K. R. (2026). FlexFlowSim: An open-source configurable
> benchmark for multi-objective reinforcement learning in flow shop
> routing [Computer software]. <https://github.com/alrashdank/FlexFlowSim>
