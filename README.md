# FlexFlowSim

**An Open-Source Configurable Benchmark for Multi-Objective Reinforcement Learning in Flow Shop Routing**

FlexFlowSim is a Python framework that combines discrete-event simulation (SimPy) with a Gymnasium-compatible reinforcement learning interface for benchmarking multi-objective routing policies in manufacturing flow shops. It includes eight dispatching-rule baselines, DQN and PPO integration via Stable-Baselines3, and an interactive Streamlit dashboard.

## Features

- **JSON-driven configuration**: Define arbitrary N-stage, M-server flow shop topologies without writing code
- **Four service-time distributions**: Normal (with minimum clamp), exponential, lognormal, uniform
- **Eight built-in dispatching rules**: RoundRobin, Random, ShortestQueue, LeastUtilised, FastServerFirst, SPT, LPT, CostMinimising
- **DQN and PPO agents** via Stable-Baselines3 with multi-seed training
- **Multi-objective reward**: Weighted scalarisation of cost rate, throughput rate, and WIP-based congestion penalty
- **Statistical evaluation**: Kruskal-Wallis tests, Cliff's delta effect sizes, multi-seed aggregation
- **Interactive dashboard**: Six-tab Streamlit interface (Configure, Train, Evaluate, Simulate, Sensitivity, Compare)
- **Reproducible**: Seeded at environment, algorithm, and evaluation levels; all logs exportable

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
├── env.py              # FlexFlowSimEnv: SimPy + Gymnasium wrapper
├── baselines.py        # Eight dispatching-rule policies
├── train.py            # Multi-seed training harness (DQN, PPO)
├── evaluate.py         # Statistical evaluation (Kruskal-Wallis, Cliff's delta)
├── calibrate.py        # Reward normalisation constant calibration
├── app.py              # Streamlit dashboard (~1100 lines, 6 tabs)
├── configs/
│   ├── bakery_bk50.json          # 2-stage bakery (real-data calibrated)
│   └── electronics_3stage.json   # 3-stage electronics assembly (synthetic)
├── requirements.txt
├── FlexFlowSim.bat     # Windows one-click launcher
├── LICENSE             # MIT
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
python evaluate.py --config configs/bakery_bk50.json --episodes 50
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

## Case Studies

### Bakery (2x2, 4 actions)
Calibrated from the BK50 dataset (Babor & Hitzmann, 2022). Two stages (Kneading, Baking), two servers each. Service times from real production data; cost ratios from energy-consumption literature.

### Electronics Assembly (3x(2,3,2), 12 actions)
Synthetic three-stage line (PCB Mounting, Soldering, Quality Testing) with seven servers. Designed to test scaling behaviour with heterogeneous cost-speed trade-offs across stages.

## Objective Weights

Four predefined scenarios are included:

| Scenario | Cost | Throughput | WIP |
|---|---|---|---|
| CostFocus | 0.8 | 0.1 | 0.1 |
| ThroughputFocus | 0.1 | 0.8 | 0.1 |
| LeadTimeFocus | 0.1 | 0.1 | 0.8 |
| Balanced | 0.33 | 0.33 | 0.34 |

Custom weight scenarios can be defined through the dashboard or passed programmatically.


## Acknowledgements

The bakery case study uses the BK50 dataset from Babor and Hitzmann (2022), available at [https://doi.org/10.17632/dhgbssb8ns.2](https://doi.org/10.17632/dhgbssb8ns.2) under CC BY 4.0.

## Licence

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
