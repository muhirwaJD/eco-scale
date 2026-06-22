# Eco-Scale Console (React control plane)

A real web UI for the Eco-Scale autoscaler — an SRE-facing control plane (think
Grafana/Sedai, but for an RL autoscaler). It shows the agent's live decisions,
*why* it made them, and how it compares to native Kubernetes HPA on the same
traffic.

The frontend is a thin view: all autoscaling logic stays in the Python backend
(the validated env + champion agent + HPA baseline), exposed via FastAPI.

## Features
- **Live operations** — replicas (RL vs HPA) over a real daily trace, load shaded behind.
- **Decision explainability** — the agent's action, a plain-English rationale, its
  action-preference bars, and the exact observation it saw.
- **RL-vs-HPA counterfactual** — pick the HPA target to compare against (50% default → 90%).
- **Impact** — cumulative pods / energy (kWh) / cost (Frw) / SLA breaches saved.
- **Trust & safety** — Recommend-only vs Autopilot mode, and a kill switch.

## Run it

```bash
# 1. start the backend (from the repo root)
uvicorn serving.api:app --port 8000

# 2. start the frontend (from web/)
cd web
npm install
npm run dev          # http://localhost:5173  (proxies /api -> :8000)
```

## Build for production
```bash
npm run build        # outputs to web/dist/
```
Set `VITE_API_URL` to point at a hosted API if you don't use the dev proxy.
