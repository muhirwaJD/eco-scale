# Chapter 4 — Results and Analysis

## 4.1 Experimental Setup

**Data.** All experiments use real CPU-utilization traces from the Alibaba 2018
cluster trace. The raw series (~95M rows) was resampled to 5-minute steps,
normalized to [0, 1], and sliced into **13 daily traces of 288 steps each**
(CPU range 0.165–0.783). Figure `peak_info.png` shows the daily structure and
peaks. The 13 traces were split **8 train / 5 test**, stratified by difficulty
(number of stressed steps) so both sets span easy-to-hard conditions. The 5 test
traces are held out and never seen during training.

**Environment.** A custom Gymnasium environment (`KubernetesEnv`) replays one
trace per episode (288 fixed-length steps; multi-trace training samples a random
trace each episode). State is 4-D — `[cpu_util, pods/max, queue/scale,
day_progress]`. Actions are discrete: scale down (−1 pod), maintain, scale up
(+1 pod), bounded to 1–20 pods.

**Reward.** The reward balances service quality against energy:

```
reward = −W_LAT·latency − W_ENERGY·(pods/MAX_PODS) − W_SLA·breach − W_SCALE·scaling
         (W_LAT=1.0,   W_ENERGY=1.5,   W_SLA=1.0,   W_SCALE=0.02)
```

Energy is charged on the **absolute** pod count, and the "right-sized" reference
is the pod count that holds utilization at a healthy 70%. This was validated
offline (`reward_design.py`) before training: a demand-tracking policy scores
−348.5 versus −500.7 for an over-provisioning ("park-high") policy, confirming
the reward rewards right-sizing rather than over-provisioning.

**Metrics.** Mean episode reward, p95 latency, SLA breach rate (% of steps at
saturation), waste (idle pods above the healthy count), and mean pod count
(energy proxy). Reward values are specific to this environment and are **not
comparable** to earlier synthetic-environment figures.

## 4.2 Hyperparameter Tuning and Algorithm Comparison

Each of DQN, PPO, and REINFORCE was trained as a **10-run hyperparameter sweep**
(150k steps per run) on the 8 train traces. Best run per algorithm:

| Algorithm | Best configuration | Mean reward | Sweep range | Stability |
|---|---|---|---|---|
| **PPO** | Larger batch | **−340.07 ± 10.24** | −351 … −340 | tightest; no collapses |
| **DQN** | Baseline | −344.58 ± 7.74 | −489 … −345 | 2 runs collapsed |
| **REINFORCE** | Larger network | −348.91 ± 12.02 | −403 … −349 | highest variance |

The ranking **PPO > DQN > REINFORCE** matches reinforcement-learning theory: PPO
is the most stable, DQN is sample-efficient but prone to instability (two
configurations collapsed to ≈ −489), and REINFORCE has the highest variance (the
no-baseline run collapsed). All three cluster near the demand-tracking ceiling
(≈ −346), indicating the recalibrated reward generalizes across algorithms.
See `convergence_comparison.png`, `stability_comparison.png`,
`sensitivity_analysis.png`, and `summary_table.png`.

PPO and DQN are statistically comparable (overlapping error bars); PPO was
retained as the deployed agent on the strength of its stability and its
performance against the HPA baseline (§4.5). REINFORCE is clearly last.

## 4.3 Champion Selection

The champion is selected automatically as the best run across all algorithms by
sweep mean reward (`agents.find_champion`) — **PPO, run 6** (`champion_mean_reward.png`).
No model path is hardcoded; the selection is read from the result tables.

## 4.4 Behavioral Validation — Does the Agent Track Demand?

A good reward number is not sufficient; the agent must right-size rather than
over-provision. On the held-out test traces (`diagnose_champion.py`):

| Policy | Reward | mean &#124;pods − required&#124; | Waste | Actions (down/hold/up) |
|---|---|---|---|---|
| **PPO champion** | −339.7 | **1.11** | 0.051 | 90 / 1248 / 102 |
| track-ideal (reference) | −345.8 | 0.54 | 0.012 | 335 / 766 / 339 |
| random (floor) | −482.2 | 3.73 | 0.060 | balanced |

The champion stays within ~1 pod of the healthy target and **beats the random
floor by +142** while matching the demand-tracking ceiling (+6). It is a genuine
autoscaler, not a degenerate constant policy.

*(Note: an earlier reward design caused the agent to over-provision — parking
near the maximum pod count, pod-gap 6.8, waste 0.34. Recalibrating the reward to
charge energy on absolute pods fixed this: pod-gap dropped to ~1–2 and waste
fell ~70%. This is reported as a methodological finding.)*

## 4.5 Comparison Against Kubernetes HPA (Headline Result)

A realistic reactive HPA baseline (`baselines/hpa_controller.py`) — standard
target-utilization formula, ±1 pod/step, scale-down stabilization — was compared
against the RL agents on the **5 held-out test traces**, each evaluated at 10
start-offsets (**50 paired episodes** per controller). Because the episodes are
matched, significance is assessed with a **paired t-test**.

| Controller | Mean reward | p95 latency | Breach % | Waste | Mean pods |
|---|---|---|---|---|---|
| **PPO** | **−340.19 ± 10.3** | **0.72** | 0.22 | 0.050 | 8.03 |
| HPA (70% target) | −344.64 ± 8.6 | 0.79 | 0.21 | 0.019 | 7.29 |
| DQN | −347.51 ± 9.3 | 0.78 | 0.47 | 0.106 | 9.06 |
| random | −460 ± 44 | 1.00 | 32 | 0.137 | 8.12 |

**Paired t-tests vs HPA:**
- **PPO vs HPA: +4.45, t = 17.1, p < 0.0001 — PPO significantly beats HPA.**
- DQN vs HPA: −2.88, t = −9.1, p < 0.0001 — DQN is significantly worse (it
  over-provisions, ~9 pods, without a service benefit).

PPO outperforms HPA on the combined objective primarily by maintaining **lower
p95 latency (0.72 vs 0.79) at comparable reliability** (`dqn_vs_hpa_comparison.png`).
This is a service-quality win; it is not, by itself, an energy reduction versus a
well-tuned HPA (HPA@70% runs fewer pods).

## 4.6 Energy Analysis — PPO vs HPA Tuning Frontier

Because HPA uses a single static utilization target, its energy/reliability
trade-off depends entirely on that target. Sweeping it (`energy_vs_hpa.py`,
`energy_frontier.png`):

| Controller | Reward | Mean pods | Waste | Breach % |
|---|---|---|---|---|
| **PPO (adaptive)** | **−340.2** | 8.03 | 0.050 | 0.2 |
| HPA @50% (conservative/safe default) | −353.2 | 9.97 | 0.142 | 0.1 |
| HPA @70% (well-tuned) | −344.6 | 7.29 | 0.019 | 0.2 |
| HPA @90% (aggressive) | −370.5 | 5.81 | 0.000 | 4.4 |

**Finding.** Against the **conservative HPA@50% that production teams deploy by
default**, PPO uses **19% fewer pods and 65% less waste at equal reliability** —
and reaches this operating point with **no threshold tuning**. PPO sits on
HPA's tuning curve near a well-tuned 65% target.

**Honest limit.** PPO does **not** beat a perfectly-tuned HPA@70% on pure energy
(7.29 vs 8.03 pods). The value of the learned policy is that it lands on a good
operating point automatically and wins on the combined objective; surpassing the
tuned frontier on energy would require a predictive (look-ahead) state feature so
the agent can pre-scale before peaks. This is identified as future work.

## 4.7 Summary of Findings

1. **RL beats the production autoscaler:** PPO significantly outperforms HPA on
   the combined latency/energy objective (p < 0.0001) on held-out real-trace data.
2. **Algorithm comparison:** PPO ≈ DQN > REINFORCE; PPO is the most stable and
   was selected as the deployed agent. DQN, as trained, over-provisions and loses
   to HPA — reported honestly.
3. **Energy:** PPO cuts pods ~19% and waste ~65% versus the conservative HPA
   default, without tuning; it matches (not beats) a perfectly-tuned HPA on pure
   energy.
4. **Methodological contribution:** an over-provisioning failure caused by reward
   misspecification was diagnosed and corrected via offline reward validation
   before training — a reusable safeguard.

**Threats to validity.** Results are from a simulated environment where latency
is modeled as clipped utilization; real systems show a non-linear latency curve.
The start-offset episodes used for statistical power are correlated samples, so
the t-test is supporting evidence alongside per-trace effect sizes. Real-cluster
validation (Kubernetes + Prometheus) is planned to confirm the simulated findings.
