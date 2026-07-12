# Chapter 4 — Results and Analysis

## 4.1 Experimental Setup

**Data.** All experiments use real CPU-utilization traces from the Alibaba 2018
cluster trace. The raw series (~95M rows) was resampled to 5-minute steps,
normalized to [0, 1], and sliced into **13 daily traces of 288 steps each**
(CPU range 0.165–0.783). Figure `outputs/data/peak_info.png` shows the daily structure and
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
See `outputs/training/learning_curves.png` (eval reward vs timesteps — PPO converges
smoothly, DQN oscillates), `outputs/training/convergence_comparison.png`,
`outputs/training/stability_comparison.png`, `outputs/training/sensitivity_analysis.png`,
and `outputs/training/summary_table.png`.

PPO and DQN are statistically comparable (overlapping error bars); PPO was
retained as the deployed agent on the strength of its stability and its
performance against the HPA baseline (§4.5). REINFORCE is clearly last.

## 4.3 Champion Selection

The champion is selected automatically as the best run across all algorithms by
sweep mean reward (`agents.find_champion`) — **PPO, run 6** (`outputs/training/champion_mean_reward.png`).
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
p95 latency (0.72 vs 0.79) at comparable reliability** (`outputs/simulation/agent_vs_hpa_comparison.png`).
This is a service-quality win; it is not, by itself, an energy reduction versus a
well-tuned HPA (HPA@70% runs fewer pods).

## 4.6 Energy Analysis — PPO vs HPA Tuning Frontier

Because HPA uses a single static utilization target, its energy/reliability
trade-off depends entirely on that target. Sweeping it (`energy_vs_hpa.py`,
`outputs/simulation/energy_frontier.png`):

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
operating point automatically and wins on the combined objective. One might expect
that surpassing the tuned frontier on energy needs a predictive (look-ahead) state
feature so the agent can pre-scale before peaks — so we tested exactly that (§4.6.1).

### 4.6.1 Testing the Predictive Hypothesis

To check whether anticipation could close the energy gap, we extended the champion
with **one extra observation** (same PPO config, so any change is attributable to the
feature, not retuning) in three variants:

- **trend** — causal slope of recent load (deployable live);
- **forecast** — causal Holt (level + trend) projection 6 steps ahead (deployable live);
- **oracle** — the *true* peak load over the next 6 steps (perfect foresight; an upper
  bound, not deployable).

Each was trained with 3 seeds (best kept) and evaluated on the 5 held-out test traces
(50 paired episodes), against the reactive champion:

| Controller | Reward | Breach % | Waste | Mean pods |
|---|---|---|---|---|
| **Champion (reactive)** | **−340.19** | 0.22 | 0.050 | 8.03 |
| Oracle (perfect foresight) | −342.66 | 0.43 | 0.043 | 7.80 |
| Trend | −342.16 | 0.24 | 0.065 | 8.35 |
| Forecast | −341.51 | 0.22 | 0.058 | 8.20 |

**Result: anticipation does not help in this environment.** All three variants are
*significantly worse* than the reactive champion on the combined objective (paired
t-tests, all **p < 0.0001**). Decisively, even the **oracle** — with perfect knowledge
of future load — loses: it runs marginally leaner (7.80 vs 8.03 pods) but roughly
*doubles* SLA breaches (0.43% vs 0.22%), a net loss under the reward. At 5-minute
control granularity with ±1-pod steps and gradually varying daily load, reacting to
current load is already near-optimal; look-ahead offers no usable advantage, and the
leaner operating point it tempts the agent toward costs reliability. The reactive
champion is therefore retained. (Figure `outputs/simulation/predictive_comparison.png`;
reproduce with `python evaluation/evaluate_predictive.py`.)

## 4.7 Real-Cluster Validation (Stage 2)

Sections 4.5–4.6 are *simulated* results: the controllers act inside the Gym
environment. To check that the learned agent actually transfers to a live system,
the champion was deployed on a real Kubernetes cluster and run head-to-head
against the cluster's own native autoscaler. This is a **deployment-feasibility
validation**, not a second statistical experiment (see caveats below).

### 4.7.1 Setup

| Component | Choice |
|---|---|
| Cluster | OrbStack single-node Kubernetes (local) |
| Workload | CPU-bound FastAPI app (`/work` burns ~50 ms CPU/request), deployed as a Deployment + Service |
| Metrics | `metrics-server` (real per-pod CPU via `kubectl top`) |
| Load | Built-in HTTP load generator, 60 worker threads, a **triangular wave** (ramp up to peak, back down) hitting the app through `kubectl port-forward` |
| Run length | 120 s per controller, decision tick every 15 s, **repeated for 3 rounds** |

The **same load wave** was applied to each controller, repeated for 3 rounds, so
the comparison is like-for-like and averaged:

- **RL controller** (`deploy/controller/rl_controller.py`): each tick reads the
  real replica count (`kubectl get`) and average pod CPU (`kubectl top`), maps
  them into the agent's normalized 4-D observation, asks the **PPO champion** for
  an action, and applies it with `kubectl scale` (±1 replica).
- **Observation calibration (sim-to-real bridge).** The agent trained on *total*
  normalized demand. The controller therefore converts the real signal as
  `cpu_util = POD_CAPACITY × (avg_pod_CPU × replicas) ÷ CPU_request`, reading the
  CPU request from the deployment. This expresses real total demand on the scale
  the agent learned, so its "pods needed" matches reality (an idle cluster maps
  to ≈0, removing the earlier idle over-provisioning). A queue *proxy* is derived
  from CPU, since the real app has no request queue.
- **Native Kubernetes HPA** (`deploy/k8s/hpa.yaml`): standard
  `HorizontalPodAutoscaler`, target CPU utilization 60 %, min 1 / max 10 — the
  cluster's built-in autoscaler, driving the *same* deployment.

### 4.7.2 Results

Per-tick replicas, CPU, and p95 latency were logged for each round
(`outputs/realcluster/`). Aggregated over **3 repeated rounds** (mean ± std,
`realcluster_repeats.csv`):

| Metric | RL (PPO) | Native HPA |
|---|---|---|
| Mean replicas | **4.00 ± 0.22** | 6.17 ± 0.31 |
| p95 latency (ms) | 1054 ± 17 | 1126 ± 72 |

**Finding.** On the real cluster the RL controller held the service with **~35 %
fewer replicas** (4.00 vs 6.17) at **comparable p95 latency** (1054 vs 1126 ms),
and the small standard deviations show the result is **reproducible across runs**,
not a one-off. With the observation calibration in place, the agent **ramps with
the load** (1 → 6 replicas as the wave climbs) and right-sizes to the correct pod
count, rather than parking arbitrarily. Its behaviour transferred to live
Kubernetes and beats the production HPA on resource use at equal service quality.

**Sim-to-real gap (reported honestly).** Before calibration the agent
over-provisioned on the real cluster — most visibly when idle, where it scaled up
despite near-zero load. The cause was an input mismatch: it was fed one pod's CPU
as a fraction of a whole core, which ignored the replica count and sat far below
the scale it trained on, so it fell back on its learned safety bias. Expressing
*total* real demand on the trained scale (the calibration above) fixes the
under-load behaviour and removes the idle over-provisioning down to the agent's
trained floor (≈4 pods rather than the maximum). The residual idle floor — the
agent does not return all the way to 1 pod — stems from training always starting
at 5 pods, and fully removing it would require on-cluster fine-tuning (future
work). Both controllers also **saturated** on the single node (~1.05–1.13 s p95),
so this regime tests resource efficiency under stress, not low-latency operation.

### 4.7.3 Caveats

The result is now averaged over repeated rounds, but its scope is still limited:

- **3 rounds, one cluster** — repeated and consistent, but a smaller sample than
  Stage 1 (50 paired episodes + t-test) and on a single machine.
- **Saturated regime** — the single node was overloaded under the wave (~1.1 s
  p95 for both), so this tests resource efficiency under stress, not a
  low-latency operating point.
- **Residual idle floor** — the agent settles near 4 pods when fully idle rather
  than 1, a leftover of its training start point (removable via on-cluster
  fine-tuning).
- **Synthetic load** — an in-process HTTP generator, not Locust (future work).
- **`metrics-server`, not Prometheus** — adequate for CPU-driven scaling, but
  without Prometheus' richer history/percentiles.
- **Queue proxy** — the observation's queue dimension is approximated from CPU
  because the real app has no request queue.

The honest takeaway: after calibrating the observation, the agent transfers to
live Kubernetes, right-sizes with the load, and **uses ~35 % fewer replicas than
the production HPA at comparable latency, reproducibly across 3 runs**. A larger,
multi-node, unsaturated study (with Prometheus and Locust) remains future work.

## 4.8 Summary of Findings

1. **RL beats the production autoscaler:** PPO significantly outperforms HPA on
   the combined latency/energy objective (p < 0.0001) on held-out real-trace data.
2. **Algorithm comparison:** PPO ≈ DQN > REINFORCE; PPO is the most stable and
   was selected as the deployed agent. DQN, as trained, over-provisions and loses
   to HPA — reported honestly.
3. **Energy:** PPO cuts pods ~19% and waste ~65% versus the conservative HPA
   default, without tuning; it matches (not beats) a perfectly-tuned HPA on pure
   energy.
4. **Real-cluster transfer:** the champion was deployed on live Kubernetes
   (OrbStack) and run head-to-head against the **real native HPA**. After
   calibrating the observation to express real total demand on the trained scale,
   the agent right-sizes with the load and uses **~35% fewer replicas at
   comparable latency (4.00 vs 6.17 pods), reproducibly over 3 runs** (Section 4.7).
   The sim-to-real gap (notably idle over-provisioning) was diagnosed and largely
   closed by the calibration; a residual idle floor is left to on-cluster
   fine-tuning.
5. **Methodological contribution:** an over-provisioning failure caused by reward
   misspecification was diagnosed and corrected via offline reward validation
   before training — a reusable safeguard.
6. **Anticipation was tested and rejected:** adding look-ahead features
   (trend/forecast/oracle) did not beat the reactive champion — even a
   perfect-foresight oracle lost (Section 4.6.1). At this control granularity,
   reaction is already near-optimal, so the reactive policy is retained.

**Threats to validity.** Results are from a simulated environment where latency
is modeled as clipped utilization; real systems show a non-linear latency curve.
The start-offset episodes used for statistical power are correlated samples, so
the t-test is supporting evidence alongside per-trace effect sizes. Real-cluster
validation (Section 4.7) confirms the agent transfers and, after observation
calibration, uses ~35% fewer replicas than the real HPA at comparable latency,
reproducibly across 3 runs — though on a single, saturated node, so a larger
multi-node study remains future work.
