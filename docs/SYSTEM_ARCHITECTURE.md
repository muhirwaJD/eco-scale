# Eco-Scale RL — System Architecture Guide

This document explains the technical "under the hood" logic of your project to help you prepare for your defense.

## 1. The High-Level Architecture
The project follows the standard **Agent-Environment Loop**. In our case:
- **Environment**: A simulated Kubernetes cluster.
- **Agent**: The RL brain making scaling decisions.

```mermaid
graph LR
    subgraph Environment [Kubernetes Simulation]
        Traffic[Traffic Generator] --> State[Current State]
        State --> Reward[Reward Calculator]
    end

    subgraph Agent [RL Controller]
        NN[Neural Network] --> Decision[Action: Up/Down/Hold]
    end

    Decision --> Environment
    State --> NN
    Reward --> NN
```

---

## 2. How the Environment Works (`KubernetesEnv`)
The environment is a discrete-time simulation ($T = 5$ minutes per step).

### A. Traffic Generation (The "Input Force")
We simulate two types of traffic:
1. **Cyclical**: A sine wave representing daily usage (high during the day, low at night).
2. **Burst**: Random "spikes" added to the sine wave to test the agent's resilience.

### B. System Dynamics
When the agent chooses an action:
1. **Pod Update**: `self.pod_count` increases, decreases, or stays the same.
2. **Latency Calculation**: 
   $$\text{Latency} = \frac{\text{Request Queue}}{\text{Pod Count} \times \text{Capacity}}$$
   If pods are too few, latency hits 1.0 (100% delay).
3. **Queue Update**: Requests arrive based on traffic and are processed based on pods.

---

## 3. The "Brain" Architecture
All three algorithms share a similar **Neural Network (MLP)** structure:
- **Input Layer (4 nodes)**: Receives the 4 state variables.
- **Hidden Layers (2x64 nodes)**: Learns complex patterns (e.g., "If it's 8:00 AM and queue is growing, scale up now!").
- **Output Layer (3 nodes)**: Predicts the value/probability of [Scale Down, Hold, Scale Up].

### Algorithm-Specific Logic:
- **DQN (Value-Based)**: Predicts **Q-Values**. It asks: *"What is the total future reward if I scale up right now?"* It picks the action with the highest Q-value.
- **PPO/REINFORCE (Policy-Based)**: Predicts **Probabilities**. It learns a distribution (e.g., 80% chance to Scale Up, 15% to Hold, 5% to Scale Down).

---

## 4. The Reward Function (The Strategy)
This is how we "teach" the agent. We penalize three things:
1. **SLA Breach**: High latency (Weight: 0.5).
2. **Carbon/Resource Waste**: Running more pods than needed (Weight: 0.3).
3. **Thrashing**: Scaling up and down too fast (Weight: 0.2).

The agent learns to find the **"Goldilocks Zone"**—not too many pods (waste), not too few (latency).

---

## 5. Why did DQN perform best?
In our results, DQN slightly beat the others. Historically, DQN is excellent at **discrete action spaces** (0, 1, 2) and **low-dimensional states**. Because our environment is stable and doesn't have "noisy" high-dimensional inputs like pixels, the Q-value estimation in DQN is very precise and converges quickly.
