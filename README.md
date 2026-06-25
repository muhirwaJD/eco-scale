---
title: Eco-Scale Console
emoji: 📈
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Eco-Scale — Predictive Kubernetes Autoscaling with Deep RL

Reinforcement-learning autoscaler for Kubernetes, trained on **real Alibaba 2018 cluster traces**
and benchmarked against the production **Horizontal Pod Autoscaler (HPA)**. The agent learns to
right-size a namespace — observing CPU load, pod count, request queue, and time-of-day — to balance
latency against energy use.

Three algorithms are compared (**DQN**, **PPO**, **REINFORCE**); **PPO** is the deployed agent.

- 🌐 **Live console:** https://muhirwa56-eco-scale-console.hf.space/ (Simulation mode — no setup needed)
- 🎬 **Demo video:** https://www.veed.io/view/8d8e28a2-8dd1-4ba3-9695-cf01fa8be08e?source=Dashboard&panel=share

---

## What the product is

Eco-Scale is a drop-in alternative to Kubernetes HPA. Instead of a static CPU threshold, it uses a
trained RL policy to decide — every interval — whether to **scale up**, **maintain**, or **scale
down** a workload. It ships in three usable forms:

| Form | What it does | How to run |
|---|---|---|
| **Web dashboard** | Browser UI: set cluster state → see the agent's decision; view RL-vs-HPA results | `streamlit run serving/dashboard.py` |
| **Decision API** | `POST /predict` returns a scaling action for a given cluster state | `uvicorn serving.api:app` |
| **Cluster controller** | Drives a real Kubernetes Deployment via `kubectl scale` | `python deploy/controller/rl_controller.py` |

## Problem

Standard Kubernetes autoscaling (HPA) is reactive — it scales only after thresholds are breached,
causing latency spikes and over-provisioning. Eco-Scale learns a scaling policy from historical
traffic to manage the latency-vs-energy trade-off.

---

## Quick start

```bash
git clone https://github.com/muhirwaJD/eco-scale.git
cd eco-scale
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Run the dashboard (the product):**
```bash
streamlit run serving/dashboard.py        # opens http://localhost:8501
```

**Run the decision API:**
```bash
uvicorn serving.api:app --reload          # docs at http://localhost:8000/docs
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cpu_util":0.9,"pods":3,"queue_depth":900,"day_progress":0.5}'
# -> {"action":2,"action_name":"scale_up", ...}
```

---

## Deployment

### Option A — Docker (one command, runs API + dashboard together)
```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Option B — Streamlit Community Cloud (public link)
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, set **Main file path** to `serving/dashboard.py`, and Deploy.
4. The model (`models/eco_scale_best.zip`) and figures (`outputs/`) are committed, so it runs as-is.
5. Paste the resulting URL at the top of this README.

### Option C — Real Kubernetes cluster (the controller)
```bash
# 1. build the sample app, then deploy it
docker build -t eco-sample-app:latest deploy/app
kubectl apply -f deploy/k8s/deployment.yaml

# 2a. baseline: let native HPA drive it
kubectl apply -f deploy/k8s/hpa.yaml
python deploy/run_experiment.py --mode hpa --duration 300

# 2b. OR: let the RL agent drive it (delete the HPA first)
kubectl delete -f deploy/k8s/hpa.yaml
python deploy/run_experiment.py --mode rl --duration 300

# 3. compare the two runs
python deploy/compare_realcluster.py        # -> outputs/realcluster/
```

---

## Testing

```bash
# Unit + integration tests (env, HPA, inference engine, API) — 43 tests
python -m pytest

# Performance on this machine (load time, latency, throughput)
python tests/benchmark_performance.py

# Same benchmark inside the container (different software spec)
docker compose run --rm api python tests/benchmark_performance.py

# Comparative test: RL vs HPA on held-out traces + paired t-test
python evaluation/evaluate_vs_hpa.py
```

| Strategy | Where |
|---|---|
| Unit | `tests/test_environment.py`, `tests/test_hpa_controller.py`, `tests/test_inference_engine.py` |
| Integration (HTTP) | `tests/test_api.py` |
| Performance / different specs | `tests/benchmark_performance.py` (local vs Docker vs cluster) |
| Comparative / statistical | `evaluation/evaluate_vs_hpa.py` (paired t-test) |

---

## Data

Real Alibaba 2018 cluster-trace CPU series, sliced into 13 daily traces (288 steps each),
split into **8 train / 5 test** (stratified by difficulty).

```bash
python data/make_split.py        # regenerate data/split.json from data/traces/
```

## Reproduce the research

```bash
# Train (10-run hyperparameter sweep per algorithm)
python training/dqn_training.py
python training/ppo_training.py
python training/reinforce_training.py

python training/reward_design.py        # validate a reward offline, before training
python training/select_champion.py      # pick the champion + render figure
python training/diagnose_champion.py    # does the champion track demand?
python evaluation/evaluate_vs_hpa.py    # HEADLINE: RL vs HPA + paired t-test
python utils/generate_plots.py          # cross-algorithm comparison figures
```

## Project structure

```
eco-scale/
├── environment/custom_env.py       # Trace-driven Gymnasium env (reward, dynamics)
├── data/                           # 13 real Alibaba traces + stratified split
├── training/                       # DQN/PPO/REINFORCE sweeps, reward design, champion selection
├── baselines/hpa_controller.py     # realistic Kubernetes HPA baseline
├── evaluation/                     # RL-vs-HPA comparison, energy frontier, cost-benefit
├── serving/                        # inference engine, FastAPI API, Streamlit dashboard
│   ├── inference_engine.py
│   ├── api.py
│   └── dashboard.py
├── deploy/                         # real-cluster validation
│   ├── app/                        # CPU-bound sample app + Dockerfile
│   ├── k8s/                        # Deployment, Service, HPA manifests
│   ├── controller/rl_controller.py # RL agent driving a live cluster
│   └── run_experiment.py           # load generator + experiment runner
├── tests/                          # unit + integration tests, performance benchmark
├── utils/                          # champion auto-detection, plotting helpers
├── models/                         # eco_scale_best.zip + champion_metadata.json (deployed agent)
├── outputs/                        # results: training / hpa_comparison / realcluster
├── docs/                           # results chapter, technical report, architecture
├── Dockerfile · docker-compose.yml # serving container
└── requirements.txt
```
*(`logs/`, `models/dqn/`, `models/pg/` hold training churn and are gitignored.)*

---

## Results

Trained on real Alibaba traces with an energy-aware reward; evaluated on 5 held-out test traces.

| Algorithm | Mean reward (test) | vs HPA |
|-----------|--------------------|--------|
| **PPO** (deployed) | **−340.2 ± 10.3** | competitive with a tuned HPA; beats the conservative default |
| HPA (baseline)     | −344.6 ± 8.6       | — |
| DQN                | −347.5 ± 9.3       | over-provisions (reported honestly) |
| REINFORCE          | −348.9 ± 12.0      | — |
| random             | ~−460              | floor |

Against the **conservative HPA@50%** teams deploy by default, PPO uses **~19% fewer pods and ~65%
less waste at equal reliability**, with no threshold tuning. It matches (does not beat) a perfectly
tuned HPA on pure energy. The champion was also validated on a **real Kubernetes cluster** (OrbStack),
where it held the service with ~30% fewer replicas — see [Chapter 4 Results](docs/CHAPTER_4_RESULTS.md).

See the [Technical Report](docs/REINFORCEMENT_LEARNING_REPORT.md) for the full analysis.
