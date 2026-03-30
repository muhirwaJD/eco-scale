# Eco-Scale RL Summative

Reinforcement learning agent for autonomous Kubernetes pod scaling.
Compares DQN, REINFORCE, and PPO on a custom Gymnasium environment.

## Problem

Cloud infrastructure wastes energy through over-provisioning. Eco-Scale learns to scale Kubernetes pods to minimize idle pod-hours while keeping request latency within SLA bounds.

## Setup

```bash
git clone https://github.com/yourusername/your_name_rl_summative
cd your_name_rl_summative
pip install -r requirements.txt
```

## Run random agent (visualization demo — no model)
```bash
python3.14 environment/rendering.py
```

## Train models
```bash
python3.14 training/dqn_training.py
python3.14 training/ppo_training.py
python3.14 training/reinforce_training.py
```

## Run best agent (main demo)
```bash
python3.14 main.py
```

## Evaluate and generate plots
```bash
python3.14 evaluation/compare_agents.py
```

## Project Structure

```
eco-scale/
├── environment/
│   ├── __init__.py              # Package init
│   ├── custom_env.py            # KubernetesEnv — custom Gymnasium environment
│   └── rendering.py             # Pygame visualization
├── training/
│   ├── dqn_training.py          # DQN training (10 hyperparameter runs)
│   ├── ppo_training.py          # PPO training (10 hyperparameter runs)
│   └── reinforce_training.py    # REINFORCE training (10 hyperparameter runs)
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved PPO/REINFORCE models
├── evaluation/
│   └── compare_agents.py        # Evaluation + plot generation
├── outputs/                     # Generated plots (PNG)
├── main.py                      # Entry point — best performing agent + GUI
├── requirements.txt             # Dependencies
└── README.md                    # This file
```
