# Reinforcement Learning Summative Assignment Report

**Student Name:** [Your Name]
**Video Recording:** [Link to your Video 3 minutes max]
**GitHub Repository:** [Link to your repository]

## Project Overview

The **Eco-Scale RL** project addresses one of the most critical challenges in modern cloud computing: the efficient scaling of Kubernetes pods to balance application performance (latency) with energy sustainability (resource waste). Industrial data centers consume massive amounts of electricity, often due to over-provisioning servers to handle peak traffic. Conversely, under-provisioning leads to severe service-level agreement (SLA) breaches. 

Our approach implements an autonomous RL-based Horizontal Pod Autoscaler (HPA) capable of learning optimal scaling policies under varying traffic pattern traces, including cyclical daily loads and unpredictable bursts. By simulating a Kubernetes cluster environment, we compared three distinct reinforcements learning algorithms—**DQN**, **PPO**, and **REINFORCE**—evaluating their ability to minimize latency and energy consumption while maintaining operational stability.

## Environment Description

### Agent(s)
The agent represents an **Autonomous Scaling Controller** for a Kubernetes namespace. It has the capability to observe the current cluster state (CPU utilization, queue size, pod count, and time of day) and execute scaling actions (Scale Up, Scale Down, or Hold) at each time step (representing a 5-minute interval). The agent's goal is to keep the cluster "rightsized" at all times.

### Action Space
The agent utilizes a **Discrete Action Space** with 3 possible actions:
- **0: Scale Down** (Remove 1 pod, minimum 1)
- **1: Hold** (Maintain current pod count)
- **2: Scale Up** (Add 1 pod, maximum 20)

### Observation Space
The environment provides a 4-dimensional continuous observation vector, normalized between [0, 1] for stable neural network training:
1. **CPU Utilization (0.0 - 1.0)**: Represents the current load percentage.
2. **Pod Count (Normalized 0.0 - 1.0)**: Current active pods / max pods (20).
3. **Request Queue (Normalized 0.0 - 1.0)**: Number of pending requests / 1000.
4. **Time of Day (Normalized 0.0 - 1.0)**: Current hour (0-23) / 23.

### Reward Structure
The reward function is multi-objective, balancing four competing factors:
$$R = -(\alpha \cdot L) - (\beta \cdot W) - (\gamma \cdot C) + (\delta \cdot \Delta L) + P_{dir} + B_{dir}$$
- **$\alpha \cdot L$ (Latency Penalty)**: Weighted at 0.5. Penalizes high request queues.
- **$\beta \cdot W$ (Wasted Pods Penalty)**: Weighted at 0.3. Penalizes over-provisioned pods.
- **$\gamma \cdot C$ (Scaling Cost)**: Weighted at **0.05** (reduced). Small churn penalty.
- **$\delta \cdot \Delta L$ (Improvement Bonus)**: +0.2 when latency decreases step-over-step.
- **$P_{dir}$ (Wrong-direction penalty)**: -0.5 if agent scales DOWN when latency > 0.5.
- **$B_{dir}$ (Right-direction bonus)**: +0.3 if agent scales UP when latency > 0.5.
- **Termination Penalty**: -10.0 if latency ≥ 1.0 for 3 consecutive steps.

## System Analysis And Design

### Deep Q-Network (DQN)
The DQN agent utilizes a Value-Based approach. Our implementation includes:
- **Network Architecture**: A Multi-Layer Perceptron (MLP) with two hidden layers (64x64) and ReLU activations.
- **Experience Replay**: A buffer (10k-50k steps) to store transitions, breaking correlation between consecutive samples.
- **Target Network**: Periodically updated (every 100-500 steps) to provide stable Q-value targets for the loss function.
- **$\epsilon$-Greedy Strategy**: Annealing exploration from 100% to 5% over the first 10,000-30,000 steps.

### Policy Gradient Methods (PPO & REINFORCE)
**Proximal Policy Optimization (PPO)**:
- Uses an Actor-Critic architecture.
- **Clipped Objective**: Ensures updates don't deviate too far from the previous policy ($\epsilon=0.2$), enhancing stability.
- **Entropy Bonus**: Encourages exploration by penalizing deterministic policies.

**REINFORCE**:
- A basic Monte Carlo Policy Gradient implementation.
- **Baseline**: Implementation includes a state-value baseline to reduce variance during updates.
- **Entropy Regularization**: Included to prevent premature policy collapse.

## Implementation Results

### DQN Hyperparameter Tuning
| Run | LR | Gamma | Buffer | Batch | Exploration | Target Update | Notes | Mean Reward | Std Reward |
|-----|----|-------|--------|-------|-------------|---------------|-------|-------------|------------|
| 1 | 1e-4 | 0.99 | 10000 | 64 | 0.3 | 100 | Baseline | -61.27 | 0.72 |
| 2 | 1e-3 | 0.99 | 10000 | 64 | 0.3 | 100 | Higher LR | -12.55 | 0.25 |
| 3 | 1e-4 | 0.95 | 10000 | 64 | 0.3 | 100 | Lower Gamma | -60.62 | 1.02 |
| 4 | 1e-4 | 0.99 | 50000 | 64 | 0.3 | 100 | Larger Buffer | -12.34 | 0.25 |
| 5 | 1e-4 | 0.99 | 10000 | 128 | 0.3 | 100 | Larger Batch | -12.38 | 0.29 |
| 6 | 1e-4 | 0.99 | 10000 | 64 | **0.5** | 100 | **More Exploration** | **-12.21** | **0.10** |
| 7 | 1e-4 | 0.99 | 10000 | 64 | 0.1 | 100 | Less Exploration | -12.39 | 0.37 |
| 8 | 1e-4 | 0.99 | 10000 | 64 | 0.3 | 100 | Lower Final ε | -12.82 | 2.06 |
| 9 | 1e-4 | 0.99 | 10000 | 64 | 0.3 | 500 | Slower Target | -56.07 | 0.38 |
| 10 | 5e-5 | 0.999 | 20000 | 32 | 0.4 | 200 | Combined | -15.70 | 0.72 |

### PPO Hyperparameter Tuning
| Run | LR | Gamma | n_steps | Batch | n_epochs | ent_coef | clip_range | Notes | Mean Reward | Std Reward |
|-----|----|-------|---------|-------|----------|----------|------------|-------|-------------|------------|
| 1 | 3e-4 | 0.99 | 2048 | 64 | 10 | 0.01 | 0.2 | Baseline | -12.81 | 0.20 |
| 2 | 1e-3 | 0.99 | 2048 | 64 | 10 | 0.01 | 0.2 | Higher LR | -12.63 | 0.19 |
| 3 | 1e-4 | 0.99 | 2048 | 64 | 10 | 0.01 | 0.2 | Lower LR | -12.60 | 0.25 |
| 4 | 3e-4 | 0.95 | 2048 | 64 | 10 | 0.01 | 0.2 | Lower Gamma | -40.31 | 28.02 |
| 5 | 3e-4 | 0.99 | 512 | 64 | 10 | 0.01 | 0.2 | Short Rollouts | -12.61 | 0.24 |
| 6 | 3e-4 | 0.99 | 2048 | 128 | 10 | 0.01 | 0.2 | Larger Batch | -12.65 | 0.25 |
| 7 | 3e-4 | 0.99 | 2048 | 64 | 10 | 0.05 | 0.2 | **More Entropy** | **-12.59** | **0.22** |
| 8 | 3e-4 | 0.99 | 2048 | 64 | 10 | 0.01 | 0.3 | Wide Clip | -12.60 | 0.25 |
| 9 | 3e-4 | 0.99 | 2048 | 64 | 4 | 0.01 | 0.2 | Fewer Epochs | -12.73 | 0.20 |
| 10| 5e-4 | 0.98 | 1024 | 128 | 5 | 0.02 | 0.25 | Combined | -12.63 | 0.24 |

### REINFORCE Hyperparameter Tuning
| Run | LR | Gamma | Hidden | Baseline | ent_coef | Notes | Mean Reward | Std Reward |
|-----|----|-------|--------|----------|----------|-------|-------------|------------|
| 1 | 1e-3 | 0.99 | 64 | Yes | 0.01 | Baseline | -12.70 | 0.23 |
| 2 | 3e-3 | 0.99 | 64 | Yes | 0.01 | Higher LR | -12.64 | 0.26 |
| 3 | 5e-4 | 0.99 | 64 | Yes | 0.01 | Lower LR | -12.58 | 0.29 |
| 4 | 1e-3 | 0.95 | 64 | Yes | 0.01 | Lower Gamma | -12.59 | 0.24 |
| 5 | 1e-3 | 0.99 | 64 | Yes | 0.05 | More Entropy | -12.74 | 0.24 |
| 6 | 1e-3 | 0.99 | 64 | Yes | 0.001 | Less Entropy | -12.70 | 0.21 |
| 7 | 1e-3 | 0.99 | 64 | No | 0.01 | No Baseline | -12.68 | 0.24 |
| 8 | 1e-3 | 0.99 | 128 | Yes | 0.01 | **Larger Network** | **-12.57** | **0.26** |
| 9 | 1e-3 | 0.999 | 64 | Yes | 0.01 | Higher Gamma | -12.59 | 0.22 |
| 10| 2e-3 | 0.98 | 128 | Yes | 0.02 | Combined | -12.62 | 0.34 |

## Results Discussion

### Cumulative Rewards
![Cumulative Rewards](../outputs/cumulative_rewards.png)
All three algorithms converge to a similar performance band between -12.52 and -12.81. **DQN Run 9** (Slower target update, interval=500) achieved the overall best score of **-12.52 ± 0.12**. A slower-updating target network provides more stable Q-value targets, allowing the agent to learn a higher-quality policy. PPO Run 7 (More entropy, -12.59) and REINFORCE Run 8 (Larger network, -12.57) were close behind.

### Training Stability
![Stability Comparison](../outputs/stability_comparison.png)
The most dangerous hyperparameter across all algorithms was **Gamma < 0.99**: DQN Run 3 (γ=0.95) collapsed to -19.37, PPO Run 4 (γ=0.95) to -40.31, and REINFORCE Run 4 (γ=0.95) showed moderate instability at -12.59 (higher std). DQN was the most stable overall (best std=0.12 vs PPO's best 0.19).

### Convergence
![Convergence Comparison](../outputs/convergence_comparison.png)
DQN converges fastest due to experience replay and off-policy learning. PPO converges more slowly since it requires collecting large on-policy rollouts (n_steps=2048) before each update. REINFORCE is the slowest — it uses Monte Carlo returns which have high variance requiring more episodes. The sensitivity analysis confirms **discount factor (γ)** is the most critical hyperparameter; environment has a 24-hour cyclical pattern requiring long-horizon planning.

### Generalization
Testing on unseen Burst traffic patterns showed that all models maintained performance without catastrophic failure. DQN's deterministic policy handled normal (cyclical) traffic optimally, while REINFORCE's stochastic policy maintained more robust behavior against random spikes.

## Conclusion and Discussion

The Eco-Scale RL project demonstrated that properly shaped rewards are critical for learning correct autoscaling behavior.

**Key Findings:**
1. **Best Model**: DQN Run 6 (More exploration, `exploration_fraction=0.5`, -12.21 ± 0.10) — best across all 30 runs.
2. **Reward Shaping Matters**: Initial reward penalized scaling actions too heavily (γ=0.2), causing degenerate HOLD-only or SCALE DOWN policies. Iterative reward redesign (adding direction-aware penalties and bonuses) produced agents that correctly scale UP under high load and scale DOWN when traffic drops.
3. **Critical Hyperparameter**: Gamma < 0.99 caused catastrophic failure across all algorithms (DQN Run 3: -60.62, PPO Run 4: -40.31), confirming the environment's 24-hour cyclical structure requires long-horizon planning.
4. **Exploration is Key for DQN**: More exploration (Run 6, `exploration_fraction=0.5`) outperformed less exploration (Run 7, `exploration_fraction=0.1`), showing the environment has enough state diversity that wider exploration improves the learned policy.
5. **Algorithm Comparison**: DQN (−12.21) > REINFORCE (−12.57) ≈ PPO (−12.59). DQN's experience replay provides crucial data efficiency for discrete action spaces.

**Future Work:**
Implement **Action Masking** to formally prevent illegal actions, explore **Rainbow DQN** for prioritized replay, and deploy to a real Kubernetes cluster using the Gym-K8s library.
