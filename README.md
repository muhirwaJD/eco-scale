# Eco-Scale RL Summative

Autonomous Reinforcement Learning agent for Horizontal Pod Autoscaling in Kubernetes.
This project compares **DQN**, **REINFORCE**, and **PPO** on a custom Gymnasium environment simulating cyclical and burst traffic patterns.

## Problem Statement

Cloud infrastructure suffers from energy waste due to over-provisioning and performance degradation due to under-provisioning. **Eco-Scale** is an RL-based agent that learns to 'right-size' a Kubernetes namespace by observing CPU load, request queues, and time-of-day, executing scaling actions to balance latency against energy consumption.

## Setup

```bash
# Clone the repository
git clone https://github.com/Jean-de-Dieu-Muhirwa/eco-scale-rl-summative
cd eco-scale-rl-summative

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 🚀 Run the Best Agent Demo (with GUI)
To see the best-performing DQN agent in action with the Pygame visualization:
```bash
python3 main.py
```

### 🎮 Run Random Agent (Visualization Demo)
To see the environment visualization without any training (random actions):
```bash
python3 environment/rendering.py
```

### 📊 Evaluate & Generate Plots
To evaluate all models and regenerate the comparison plots:
```bash
python3 generate_plots.py
```

## Project Structure

```
eco-scale/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment
│   └── rendering.py             # Pygame dashboard and visualization
├── training/
│   ├── dqn_training.py          # DQN training (10 runs)
│   ├── ppo_training.py          # PPO training (10 runs)
│   └── reinforce_training.py    # REINFORCE training (10 runs)
├── models/
│   └── best_model.zip           # Currently selected best performing agent
├── outputs/                     # Comparison plots and result tables
├── docs/
│   ├── REINFORCEMENT_LEARNING_REPORT.md  # Final Technical Report
│   └── video_script.md          # 3-minute presentation script
├── main.py                      # Main entry point for the demo
└── requirements.txt             # Project dependencies
```

## Results Summary

| Algorithm | Best Mean Reward | Top Configuration |
|-----------|------------------|-------------------|
| **DQN**   | **-12.52**       | Slower Target Update |
| REINFORCE | -12.57           | Larger Network (128) |
| PPO       | -12.59           | Higher Entropy Coeff |

For a full breakdown of the hyperparameter tuning and performance analysis, see the [Technical Report](docs/REINFORCEMENT_LEARNING_REPORT.md).
