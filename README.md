# Eco-Scale — Predictive Kubernetes Autoscaling with Deep RL

Reinforcement-learning autoscaler for Kubernetes, trained on **real Alibaba 2018 cluster traces**
and benchmarked against the production **Horizontal Pod Autoscaler (HPA)**. The agent learns to
right-size a namespace — observing CPU load, pod count, request queue, and time-of-day — to balance
latency against energy use.

Three algorithms are compared (**DQN**, **PPO**, **REINFORCE**); **PPO** is the deployed agent.

- 🌐 **Live console:** https://muhirwa56-eco-scale-console.hf.space/ (Simulation mode — no setup needed)
- 🎬 **Demo video:** https://www.veed.io/view/47da68ba-b65a-4497-9049-90ceb36d9bfe?source=editor&panel=share

---

## What the product is

Eco-Scale is a drop-in alternative to Kubernetes HPA. Instead of a static CPU threshold, it uses a
trained RL policy to decide — every interval — whether to **scale up**, **maintain**, or **scale
down** a workload. It ships in three usable forms:

| Form | What it does | Entry point |
|---|---|---|
| **Web console** (primary) | React control plane: watch the agent decide on a recorded trace, drive a **real cluster** live, or run a Benchmark vs native HPA | `uvicorn serving.api:app` (serves the built `web/` console + API on one port) |
| **Decision API** | `POST /predict` returns a scaling action for a given cluster state | `uvicorn serving.api:app` |
| **Cluster controller** | Headless RL agent driving a real Deployment via `kubectl scale` | `python deploy/controller/rl_controller.py` |

Standard Kubernetes autoscaling (HPA) is reactive — it scales only after thresholds are breached,
causing latency spikes and over-provisioning. Eco-Scale learns a scaling policy from historical
traffic to manage the latency-vs-energy trade-off.

---

## Run it

Pick the row that matches what you want. Most people only need the first one.

| I want to… | What to do | What I need installed |
|---|---|---|
| **Just see it work** | Open the [live console](https://muhirwa56-eco-scale-console.hf.space/) | nothing (Simulation mode only) |
| **Run it against my own cluster** | [Run locally](#run-locally-against-your-own-cluster) | Python 3.12 · Node 20+ · `kubectl` + metrics-server |
| **Reproduce the research** | [Reproduce](#reproduce-the-research) | full `requirements.txt` (adds training deps) |

> The public link runs in **Simulation** only — it replays a real Alibaba trace (RL vs HPA) with no
> cluster attached. The **Live cluster** and **Benchmark** modes need a real cluster, so they only work when
> you run it locally, next.

### Run locally against your own cluster

This is the real product: the console drives a **live** Kubernetes cluster. It runs on your **host**
(not in a container) so it inherits your `kubectl` access — a container wouldn't see your cluster.

**Prerequisites**
- Python 3.12 and Node 20+
- A Kubernetes cluster reachable via `kubectl` (OrbStack / minikube / kind / EKS …)
- **metrics-server** installed in that cluster (the agent reads live pod CPU from `kubectl top`)

```bash
git clone https://github.com/muhirwaJD/eco-scale.git
cd eco-scale

# 1. serving dependencies (lean — no training/eval extras)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-serving.txt

# 2. build the web console (the API serves it from web/dist).
#    VITE_API_URL="" makes the console call the API on the SAME origin — required,
#    or it will look for the API under /api and every request 404s.
cd web && npm ci && VITE_API_URL="" npm run build && cd ..

# 3. run API + console ON YOUR HOST, so it can reach your cluster
uvicorn serving.api:app --port 8000
#    → open http://localhost:8000
```

**Give it a workload to manage.** Live mode targets a Deployment named `eco-sample-app` in your
current `kubectl` context. Deploy the bundled CPU-bound sample app if you don't already have one:

```bash
docker build -t eco-sample-app:latest deploy/app
kubectl apply -f deploy/k8s/deployment.yaml     # creates the eco-sample-app Deployment
```

Then, in the console:
- **Live cluster** → *Recommend-only* (watch the agent), *Autopilot* (let it `kubectl scale` for real),
  or the *Kill switch* to hand control back instantly.
- **Benchmark** → runs the RL agent and native HPA back-to-back under the same load, and
  reports pods + latency for each.

**Just the decision API (no UI):** skip step 2; `uvicorn serving.api:app` exposes interactive docs at
`http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cpu_util":0.9,"pods":3,"queue_depth":900,"day_progress":0.5}'
# -> {"action":2,"action_name":"scale_up", ...}
```

### Real-cluster benchmark from the command line

The same experiment the console's Benchmark mode runs, scriptable and reproducible:

```bash
docker build -t eco-sample-app:latest deploy/app
kubectl apply -f deploy/k8s/deployment.yaml

kubectl apply -f deploy/k8s/hpa.yaml                    # baseline: let native HPA drive it
python deploy/run_experiment.py --mode hpa --duration 300

kubectl delete -f deploy/k8s/hpa.yaml                   # then let the RL agent drive it
python deploy/run_experiment.py --mode rl  --duration 300

python deploy/compare_realcluster.py                    # -> outputs/realcluster/
```

<details>
<summary><b>Other interfaces & deployment (optional)</b></summary>

- **Docker (simulation only):** `docker compose up` runs the console in a container on
  `http://localhost:8000`. Note that **Live-cluster and Benchmark modes won't work** this way — the container
  can't reach your kubeconfig. Use the host `uvicorn` path above for real-cluster use.
- **Streamlit research view:** `streamlit run serving/dashboard.py` — an older, simpler dashboard kept
  for offline analysis (works with the serving deps above).
- **Deploy the console publicly (Hugging Face Space):** see
  [`deploy/huggingface/DEPLOY.md`](deploy/huggingface/DEPLOY.md).

</details>

---

## Testing

Tests use pytest/httpx from the full requirements: `pip install -r requirements.txt`.

```bash
# Unit + integration tests (env, HPA, inference engine, API) — 43 tests
python -m pytest

# Performance on this machine (load time, latency, throughput)
python tests/benchmark_performance.py

# Same benchmark inside the container (different software spec)
docker compose run --rm app python tests/benchmark_performance.py

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

Training and evaluation need the **full** dependency set: `pip install -r requirements.txt`
(adds CPU torch-training, scipy, tensorboard on top of the serving deps).

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
├── data/                           # 13 real Alibaba traces + stratified split (make_split.py, split.json)
├── training/                       # DQN/PPO/REINFORCE sweeps, reward design, champion selection, diagnostics
├── baselines/hpa_controller.py     # realistic Kubernetes HPA baseline
├── evaluation/                     # evaluate_vs_hpa · energy_vs_hpa · cost_benefit
├── serving/                        # inference + API + live cluster + simulation
│   ├── inference_engine.py         # loads the champion, normalizes obs, returns a decision
│   ├── api.py                      # FastAPI: serves the React console (web/dist) + all endpoints
│   ├── simulation.py               # recorded-trace replay (RL vs HPA), used by Simulation mode
│   ├── live_cluster.py             # reads/scales a REAL cluster via kubectl (Live mode)
│   ├── experiment.py               # sequential RL-vs-HPA Benchmark runner
│   ├── load_generator.py           # UI-controllable traffic wave
│   ├── results_service.py          # reads pre-computed results + champion metadata (/results, /model)
│   └── dashboard.py                # older Streamlit research view (optional)
├── web/                            # React + Vite control-plane console (the primary UI)
│   └── src/
│       ├── sections/               # Dashboards · Decisions · Results · Model
│       ├── components/eco/         # shared UI primitives + sidebar
│       ├── api.ts, types.ts        # typed client for the FastAPI endpoints
│       └── App.tsx                 # console shell (mode switch, cluster selector)
├── deploy/                         # real-cluster validation + public deploy
│   ├── app/                        # CPU-bound sample app + Dockerfile
│   ├── k8s/                        # Deployment, Service, HPA manifests
│   ├── controller/rl_controller.py # headless RL agent driving a live cluster
│   ├── run_experiment.py           # load generator + experiment runner
│   ├── compare_realcluster.py      # aggregate one RL-vs-HPA run
│   ├── repeat_realcluster.py       # repeat runs → mean ± std
│   └── huggingface/                # public-console deploy notes (DEPLOY.md)
├── tests/                          # unit + integration tests + performance benchmark
├── utils/                          # champion auto-detection, agents, plotting helpers
├── models/                         # eco_scale_best.zip + champion_metadata.json (deployed agent)
├── outputs/                        # results: training / simulation / realcluster / data
├── docs/                           # results chapter, technical report, architecture
├── Dockerfile · docker-compose.yml # serving container (console + API on one port)
├── requirements.txt                # full: training + eval + serving
└── requirements-serving.txt        # lean: just what the console/API need
```
*(`venv/`, `logs/`, `models/dqn/`, `models/pg/`, `web/node_modules/`, `web/dist/` are gitignored.)*

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
