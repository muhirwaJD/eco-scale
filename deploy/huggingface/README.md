---
title: Eco-Scale Console
emoji: 📈
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Eco-Scale Console

Live web console for **Eco-Scale**, a reinforcement-learning Kubernetes autoscaler.
Public deployment runs in **Simulation mode** (the agent vs HPA on real Alibaba
traces). The Live-cluster and Real A/B (Stage-2) tabs are disabled here because
they require a real Kubernetes cluster — see those in the demo video / local run.

> This file is the Hugging Face **Space** README. Copy it to the root of your
> Space repo (it carries the `sdk: docker` + `app_port: 7860` config HF needs).
> The Space builds from the repo's root `Dockerfile`.
