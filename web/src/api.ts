import type { Config, SimState } from "./types";

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

export const simReset = (hpaTarget: number) =>
  jsonFetch<SimState>("/sim/reset", {
    method: "POST",
    body: JSON.stringify({ hpa_target: hpaTarget }),
  });

export const simStep = () =>
  jsonFetch<SimState>("/sim/step", { method: "POST" });
