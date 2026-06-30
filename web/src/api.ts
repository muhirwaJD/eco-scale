import type {
  ClusterInfo, Config, ExperimentStatus, LoadStatus, ModelInfo, Results, SimState,
} from "./types";

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.ts).
const BASE = import.meta.env.VITE_API_URL ?? "/api";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const getConfig = () => jsonFetch<Config>("/config");

// ── simulation (recorded trace) ──────────────────────────────
export const simReset = (hpaTarget: number) =>
  jsonFetch<SimState>("/sim/reset", {
    method: "POST",
    body: JSON.stringify({ hpa_target: hpaTarget }),
  });

export const simStep = () => jsonFetch<SimState>("/sim/step", { method: "POST" });

// ── live cluster (real Kubernetes) ───────────────────────────
export const liveAvailable = () =>
  jsonFetch<{ available: boolean }>("/live/available");

export const liveReset = () =>
  jsonFetch<SimState>("/live/reset", { method: "POST" });

export const liveStep = (apply: boolean) =>
  jsonFetch<SimState>("/live/step", {
    method: "POST",
    body: JSON.stringify({ apply }),
  });

export const liveInfo = () => jsonFetch<ClusterInfo>("/live/info");

// ── kubectl contexts (real cluster selector) ─────────────────
export const getContexts = () =>
  jsonFetch<{ current: string; contexts: string[] }>("/contexts");

export const useKubeContext = (context: string) =>
  jsonFetch<{ ok: boolean; current?: string; error?: string }>("/contexts/use", {
    method: "POST",
    body: JSON.stringify({ context }),
  });

// ── UI-controllable load generator ───────────────────────────
export const loadStart = (duration: number) =>
  jsonFetch<LoadStatus>("/live/load/start", {
    method: "POST",
    body: JSON.stringify({ duration }),
  });

export const loadStop = () => jsonFetch<LoadStatus>("/live/load/stop", { method: "POST" });
export const loadStatus = () => jsonFetch<LoadStatus>("/live/load/status");

// ── real Stage-2 experiment (agent vs real HPA) ──────────────
export const experimentStart = (duration: number) =>
  jsonFetch<ExperimentStatus>("/experiment/start", {
    method: "POST",
    body: JSON.stringify({ duration }),
  });

export const experimentStop = () =>
  jsonFetch<ExperimentStatus>("/experiment/stop", { method: "POST" });

export const experimentStatus = () =>
  jsonFetch<ExperimentStatus>("/experiment/status");

// ── results + model metadata ─────────────────────────────────
export const getResults = () => jsonFetch<Results>("/results");
export const getModel = () => jsonFetch<ModelInfo>("/model");
