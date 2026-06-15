# Eco-Scale — Predictive Kubernetes Autoscaling with Deep RL

Reinforcement-learning autoscaler for Kubernetes, trained on **real Alibaba 2018 cluster traces**
and benchmarked against the production **Horizontal Pod Autoscaler (HPA)**. The agent learns to
right-size a namespace — observing CPU load, pod count, request queue, and time-of-day — to balance
latency against energy use.

Three algorithms are compared (**DQN**, **PPO**, **REINFORCE**); **PPO** is the deployed agent,
selected empirically as the only one to significantly beat the HPA baseline on held-out traces.

## Problem

Standard Kubernetes autoscaling (HPA) is reactive — it scales only after thresholds are breached,
causing latency spikes and over-provisioning. Eco-Scale learns a scaling policy from historical
traffic to manage the latency-vs-energy trade-off proactively.

## Setup

```bash
git clone https://github.com/Jean-de-Dieu-Muhirwa/eco-scale-rl-summative
cd eco-scale-rl-summative
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data

Real Alibaba 2018 cluster-trace CPU series, sliced into 13 daily traces (288 steps each),
split into **8 train / 5 test** (stratified by difficulty).

```bash
python data/make_split.py        # regenerate data/split.json from data/traces/
```

## Usage

```bash
# Train (10-run hyperparameter sweep, ~11 min/algorithm on CPU)
python training/dqn_training.py
python training/ppo_training.py
python training/reinforce_training.py

# Validate a candidate reward offline BEFORE training (no GPU)
python training/reward_design.py

# Select the champion from a sweep + render the mean-reward figure
python training/select_champion.py

# Behavioral diagnostic — does the champion track demand (vs over-provision)?
python training/diagnose_champion.py

# HEADLINE: RL vs HPA on held-out test traces + paired t-test
python evaluation/evaluate_vs_hpa.py

# Regenerate cross-algorithm comparison figures
python generate_plots.py
```

## Project Structure

```
eco-scale/
├── environment/
│   └── custom_env.py            # Trace-driven Gymnasium env (reward, dynamics)
├── data/
│   ├── traces/                  # 13 real Alibaba daily traces (.npy)
│   ├── make_split.py            # stratified train/test split generator
│   └── split.json               # 8 train / 5 test assignment
├── training/
│   ├── dqn_training.py          # DQN sweep (10 runs)
│   ├── ppo_training.py          # PPO sweep (10 runs)
│   ├── reinforce_training.py    # REINFORCE sweep (10 runs)
│   ├── reward_design.py         # offline reward validation (no training)
│   ├── select_champion.py       # champion selection + headline figure
│   └── diagnose_champion.py     # behavioral diagnostic (demand-tracking)
├── baselines/
│   └── hpa_controller.py        # realistic Kubernetes HPA baseline
├── evaluation/
│   └── evaluate_vs_hpa.py       # RL-vs-HPA comparison + paired t-test
├── models/
│   ├── eco_scale_ppo_best.zip   # deployed champion (PPO)
│   ├── eco_scale_dqn_best.zip   # DQN champion (comparison)
│   └── *_champion_metadata.json
├── outputs/                     # results, grouped by phase
│   ├── data/                    # trace characterization
│   ├── training/                # sweep results + champion + comparison figures
│   ├── hpa_comparison/          # RL-vs-HPA, energy, cost-benefit
│   └── realcluster/             # Stage 2 real-cluster validation
├── docs/                        # technical report, system architecture
├── generate_plots.py            # cross-algorithm comparison figures
└── requirements.txt
```
*(`logs/`, `models/dqn/`, `models/pg/` hold training churn and are gitignored.)*

## Results

Trained on real Alibaba traces with an energy-aware reward; evaluated on 5 held-out test traces.
Reward combines latency, absolute energy (pod count), SLA breaches, and scaling cost — so values
are **not comparable** to the earlier synthetic-env figures.

| Algorithm | Mean reward (test) | vs HPA |
|-----------|--------------------|--------|
| **PPO** (deployed) | **−340.2 ± 10.3** | **+4.5, p<0.0001 — significantly beats HPA** |
| HPA (baseline)     | −344.6 ± 8.6       | — |
| DQN                | −347.5 ± 9.3       | loses (over-provisions) |
| REINFORCE          | −348.9 ± 12.0      | — |
| random             | ~−460              | floor |

**Headline:** PPO significantly outperforms HPA (paired t-test, p<0.0001) by maintaining lower
p95 latency at comparable reliability. This is a latency/quality win at a modest pod-usage
increase — not an absolute energy reduction versus HPA.

See the [Technical Report](docs/REINFORCEMENT_LEARNING_REPORT.md) for the full analysis.
