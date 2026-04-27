# FlexFlowSim

**An Open-Source Configurable Benchmark for Reinforcement Learning in Flow Shop Routing**

FlexFlowSim is a Python framework that combines discrete-event simulation
(SimPy) with a Gymnasium-compatible reinforcement learning interface for
benchmarking routing policies in manufacturing flow shops. It includes
eight dispatching-rule baselines, two online bandit baselines, DQN and
PPO integration via Stable-Baselines3, optional non-stationarity
mechanisms (machine breakdowns, demand surges, processing-cost drift),
and an interactive Streamlit dashboard.

## Features

- **JSON-driven configuration**: Define arbitrary N-stage, M-server flow shop topologies without writing code
- **Four service-time distributions**: Normal (with minimum clamp), exponential, lognormal, uniform
- **Eight built-in dispatching rules**: RoundRobin, Random, ShortestQueue, LeastUtilised, FastServerFirst, SPT, LPT, CostMinimising
- **Two online bandit baselines**: Vanilla Thompson Sampling, HBQ (Hybrid Bandit-Queue)
- **DQN and PPO agents** via Stable-Baselines3 with multi-seed training
- **Three opt-in non-stationarity mechanisms**: per-server machine breakdowns (preempt-resume, configurable TTF/TTR), piecewise-constant arrival-rate surges via Lewis-Shedler thinning, and piecewise-constant processing-cost drift
- **Multi-objective reward**: Weighted scalarisation of cost rate, throughput rate, and WIP-based congestion penalty
- **Statistical evaluation**: Kruskal-Wallis tests, Cliff's delta effect sizes, multi-seed aggregation
- **Interactive dashboard**: Six-tab Streamlit interface (Configure, Train, Evaluate, Simulate, Sensitivity, Compare)
- **Reproducible**: Seeded at environment, algorithm, and evaluation levels; byte-identical regression test guarantees stationary configurations match earlier paper releases exactly; all logs exportable

## Installation

### Requirements

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/alrashdank/FlexFlowSim.git
cd FlexFlowSim
pip install -r requirements.txt
```

### Quick Start (Dashboard)

```bash
streamlit run app.py
```

Or on Windows, double-click `FlexFlowSim.bat`.

## Repository Structure

```
FlexFlowSim/
├── env.py                 # FlexFlowSimEnv: SimPy + Gymnasium wrapper
├── baselines.py           # Dispatching-rule and bandit policies
├── train.py               # Multi-seed training harness (DQN, PPO)
├── evaluate.py            # Statistical evaluation (Kruskal-Wallis, Cliff's delta)
├── calibrate.py           # Reward normalisation constant calibration
├── app.py                 # Streamlit dashboard
├── configs/
│   ├── bakery_bk50.json                       # 2-stage bakery (real-data calibrated)
│   ├── electronics_3stage.json                # 3-stage electronics assembly (synthetic)
│   ├── bakery_breakdowns_A{99,95,90,80,70}.json       # Bakery breakdown variants
│   └── electronics_breakdowns_A{99,95,90,80,70}.json  # Electronics breakdown variants
├── tests/
│   ├── test_byte_identical.py        # Stationary regression test (42 cases)
│   ├── golden_hashes.json            # 20-seed Paper 3/4 reference digests
│   └── regenerate_golden_hashes.py   # Helper to refresh the reference
├── paper5/
│   ├── README.md                  # Paper 5 reproduction guide
│   ├── analyse_results.py         # Regenerate Paper 5 tables from CSVs
│   └── results/                   # CSV outputs underlying Paper 5 Tables 2-7
├── requirements.txt
├── FlexFlowSim.bat        # Windows one-click launcher
├── LICENSE                # MIT
└── README.md
```

## Usage

### Command-Line Training

```bash
# Train DQN and PPO on the bakery configuration, 500 episodes, 5 seeds
python train.py --config configs/bakery_bk50.json --algo DQN PPO --episodes 500 --seeds 42 123 256 512 1024

# Train on the electronics configuration
python train.py --config configs/electronics_3stage.json --algo DQN PPO --episodes 500 --seeds 42 123 256 512 1024
```

### Command-Line Evaluation

```bash
# Evaluate all methods (baselines + trained RL agents)
python evaluate.py --config configs/bakery_bk50.json --reps 50
```

### Custom Configuration

Create a JSON file specifying your flow shop topology:

```json
{
  "arrival": { "distribution": "exponential", "mean": 9.6 },
  "shift_length": 480,
  "dt": 1.0,
  "max_queue": 50,
  "waiting_cost": 0.1,
  "stages": [
    {
      "name": "Stage 1",
      "servers": [
        {
          "name": "Server A",
          "service_time": { "distribution": "normal", "mean": 14.2, "std": 5.8 },
          "processing_cost": 1.5,
          "idle_cost": 0.5
        },
        {
          "name": "Server B",
          "service_time": { "distribution": "normal", "mean": 16.7, "std": 6.5 },
          "processing_cost": 1.0,
          "idle_cost": 0.5
        }
      ]
    }
  ]
}
```

## Non-Stationarity Mechanisms

Three opt-in mechanisms are available, all configurable via JSON. Each uses a
separate random number stream so toggling any one mechanism does not perturb
the realisations of the others.

### Per-server breakdowns

```json
"breakdowns": {
  "enabled": true,
  "default": {
    "ttf": { "distribution": "exponential", "mean": 180.0 },
    "ttr": { "distribution": "lognormal", "mean": 20.0, "std": 6.667 }
  }
}
```

Servers alternate between operational and broken states. Time-to-failure (TTF)
and time-to-repair (TTR) are sampled from configurable distributions
(exponential, lognormal, Weibull, or normal). Long-run availability per
server is `A = MTBF / (MTBF + MTTR)`. Breakdowns follow the **preempt-resume**
convention (Pinedo, 2016): a job in service when a server fails is paused, its
remaining service time is preserved, and processing resumes from the same
point when the server is repaired. No work is lost.

### Demand surges

```json
"arrival": {
  "distribution": "exponential",
  "mean": 6.0,
  "schedule": [
    { "t": 0,   "rate": 0.166 },
    { "t": 100, "rate": 0.500 },
    { "t": 200, "rate": 0.166 }
  ]
}
```

Piecewise-constant arrival-rate schedule implemented via Lewis-Shedler
thinning (Lewis & Shedler, 1979). Preserves the Poisson character within each
segment.

### Processing-cost drift

```json
"cost_schedule": {
  "enabled": true,
  "processing_multiplier": [
    { "t": 0,   "multiplier": 1.0 },
    { "t": 240, "multiplier": 2.5 }
  ]
}
```

Piecewise-constant multiplier on processing cost. Idle and waiting costs
are unchanged.

When all three mechanisms are absent or disabled, the simulator takes the
original code path and produces traces that are byte-identical to the
unmodified Paper 3 / Paper 4 release. This is verified by the regression
test in `tests/`.

## Tests

```bash
pytest tests/
```

The byte-identical regression test runs 40 stationary parametrised cases
(2 testbeds x 20 seeds x 480 steps) against a stored Paper 3 / Paper 4
reference, plus 2 sanity checks (the golden file is complete; enabling
breakdowns actually changes the trace). Total runtime is ~3 seconds.

## Reproducing the Paper 5 Benchmark

The folder `paper5/` contains the CSV outputs underlying

> Alrashdan, K. R. (2026). Routing under machine breakdowns: a benchmark
> of dispatching rules, bandits, and reinforcement learning for
> multi-server flow shops.

See `paper5/README.md` for the full reproduction guide and the mapping from
CSVs to manuscript tables.

```bash
# Regenerate Tables 2-7 from the bundled CSVs
python paper5/analyse_results.py
```

## Objective Weights (Paper 4)

Four predefined scenarios are included for the multi-objective reward
formulation used in Paper 4 (the MORL benchmark):

| Scenario | Cost | Throughput | WIP |
|---|---|---|---|
| CostFocus | 0.8 | 0.1 | 0.1 |
| ThroughputFocus | 0.1 | 0.8 | 0.1 |
| LeadTimeFocus | 0.1 | 0.1 | 0.8 |
| Balanced | 0.33 | 0.33 | 0.34 |

Custom weight scenarios can be defined through the dashboard or passed
programmatically. Paper 5 uses single-objective cost-per-unit minimisation
and does not use these scenarios.

## Citation

For the simulator itself:

```bibtex
@software{flexflowsim2026,
  title={FlexFlowSim: An Open-Source Configurable Benchmark for
         Reinforcement Learning in Flow Shop Routing},
  author={Alrashdan, Khaled R.},
  year={2026},
  url={https://github.com/alrashdank/FlexFlowSim}
}
```

For the Paper 5 benchmark:

```bibtex
@article{alrashdan2026routing,
  title={Routing under machine breakdowns: a benchmark of dispatching rules,
         bandits, and reinforcement learning for multi-server flow shops},
  author={Alrashdan, Khaled R.},
  year={2026},
  note={Under review}
}
```

## Acknowledgements

The bakery case study uses the BK50 dataset from Babor and Hitzmann (2022),
available at [https://doi.org/10.17632/dhgbssb8ns.2](https://doi.org/10.17632/dhgbssb8ns.2)
under CC BY 4.0.

## Licence

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
