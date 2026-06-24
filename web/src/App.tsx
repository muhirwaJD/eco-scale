import { useCallback, useEffect, useRef, useState } from "react";
import { Activity, Cpu, Leaf, Server, ShieldCheck, Wallet } from "lucide-react";
import Header from "./components/Header";
import KpiCard from "./components/KpiCard";
import Controls from "./components/Controls";
import LiveChart, { type ChartPoint } from "./components/LiveChart";
import DecisionPanel from "./components/DecisionPanel";
import ClusterInfoPanel from "./components/ClusterInfo";
import LoadControl from "./components/LoadControl";
import ExperimentPanel from "./components/ExperimentPanel";
import {
  getConfig, liveAvailable, liveInfo, liveReset, liveStep,
  loadStart, loadStatus, loadStop, simReset, simStep,
} from "./api";
import type { ClusterInfo, Config, LoadStatus, Mode, SimState } from "./types";

type Source = "sim" | "live" | "exp";

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [state, setState] = useState<SimState | null>(null);
  const [history, setHistory] = useState<ChartPoint[]>([]);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(250);
  const [hpaTarget, setHpaTarget] = useState(0.5);
  const [mode, setMode] = useState<Mode>("autopilot");
  const [killed, setKilled] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<Source>("sim");
  const [liveOk, setLiveOk] = useState(false);
  const [cluster, setCluster] = useState<ClusterInfo | null>(null);
  const [load, setLoad] = useState<LoadStatus | null>(null);

  const timer = useRef<number | null>(null);
  const sourceRef = useRef(source);
  sourceRef.current = source;
  const applyRef = useRef(false);
  applyRef.current = source === "live" && mode === "autopilot" && !killed;

  const toPoint = (s: SimState): ChartPoint => ({
    hour: +((s.tick / s.max_ticks) * 24).toFixed(2),
    load: Math.round(s.cpu * 100),
    rlPods: s.rl.pods,
    hpaPods: s.hpa?.pods,
  });

  const reset = useCallback(
    async (target = hpaTarget, src: Source = sourceRef.current) => {
      setRunning(false);
      try {
        const s = src === "live" ? await liveReset() : await simReset(target);
        setState(s);
        setHistory([toPoint(s)]);
        setError(null);
      } catch {
        setError("Cannot reach the API. Start it with: uvicorn serving.api:app");
      }
    },
    [hpaTarget]
  );

  // initial load
  useEffect(() => {
    getConfig().then(setConfig).catch(() => {});
    liveAvailable().then((r) => setLiveOk(r.available)).catch(() => {});
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // play loop — sim or live (live is paced slower because it calls kubectl)
  useEffect(() => {
    if (!running) return;
    const intervalMs = sourceRef.current === "live" ? Math.max(speed, 3000) : speed;
    timer.current = window.setInterval(async () => {
      try {
        const s =
          sourceRef.current === "live" ? await liveStep(applyRef.current) : await simStep();
        setState(s);
        setHistory((h) => [...h, toPoint(s)]);
        if (s.done) setRunning(false);
      } catch {
        setRunning(false);
        setError("Lost connection to the API.");
      }
    }, intervalMs);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [running, speed]);

  // in live mode, keep cluster info + load status fresh
  useEffect(() => {
    if (source !== "live") return;
    const refresh = () => {
      liveInfo().then(setCluster).catch(() => {});
      loadStatus().then(setLoad).catch(() => {});
    };
    refresh();
    const id = window.setInterval(refresh, 2500);
    return () => window.clearInterval(id);
  }, [source]);

  const changeHpaTarget = (t: number) => {
    setHpaTarget(t);
    reset(t);
  };

  const changeSource = (src: Source) => {
    if (src === source) return;
    setRunning(false);
    setSource(src);
    if (src !== "exp") reset(hpaTarget, src);   // exp manages its own lifecycle
  };

  const startLoad = () => loadStart(300).then(setLoad).catch(() => {});
  const stopLoad = () => loadStop().then(setLoad).catch(() => {});

  const isLive = source === "live";
  const isExp = source === "exp";
  const sv = state?.savings;
  const st = state?.stats;

  return (
    <div className="min-h-screen">
      <Header
        config={config}
        mode={mode}
        killed={killed}
        live={isLive}
        onMode={setMode}
        onKill={() => setKilled((k) => !k)}
      />

      <main className="mx-auto max-w-7xl space-y-4 p-6">
        {/* data source toggle + status badge */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex rounded-lg border border-slate-700 p-0.5 text-xs">
            <button
              onClick={() => changeSource("sim")}
              className={`rounded-md px-3 py-1.5 transition ${
                source === "sim" ? "bg-slate-700 text-white" : "text-slate-400"
              }`}
            >
              ▶ Simulation (recorded trace)
            </button>
            <button
              onClick={() => changeSource("live")}
              disabled={!liveOk}
              title={liveOk ? "Read the real Kubernetes cluster" : "No reachable cluster"}
              className={`rounded-md px-3 py-1.5 transition ${
                source === "live" ? "bg-eco-green text-white" : "text-slate-400"
              } ${!liveOk ? "cursor-not-allowed opacity-40" : ""}`}
            >
              ◉ Live cluster
            </button>
            <button
              onClick={() => changeSource("exp")}
              disabled={!liveOk}
              title={liveOk ? "Run the real RL-vs-HPA experiment" : "No reachable cluster"}
              className={`rounded-md px-3 py-1.5 transition ${
                source === "exp" ? "bg-eco-green text-white" : "text-slate-400"
              } ${!liveOk ? "cursor-not-allowed opacity-40" : ""}`}
            >
              ⚗ Real A/B (Stage-2)
            </button>
          </div>

          <span
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              isLive || isExp ? "bg-eco-green/20 text-eco-light" : "bg-slate-700/50 text-slate-300"
            }`}
          >
            {isExp
              ? "● REAL A/B — agent vs native Kubernetes HPA"
              : isLive
              ? `● LIVE — ${cluster?.deployment ?? "cluster"}${
                  state?.applied ? " · autopilot scaling" : " · read-only"
                }`
              : "● SIMULATION — replaying a real Alibaba trace"}
          </span>
        </div>

        {error && (
          <div className="rounded-lg border border-eco-red/40 bg-eco-red/10 px-4 py-3 text-sm text-eco-red">
            {error}
          </div>
        )}

        {isExp && <ExperimentPanel />}

        {!isExp && (
        <>
        {/* controls */}
        <Controls
          source={source}
          running={running}
          speed={speed}
          hpaTarget={hpaTarget}
          tick={state?.tick ?? 0}
          maxTicks={state?.max_ticks ?? 288}
          onToggle={() => setRunning((r) => !r)}
          onReset={() => reset()}
          onSpeed={setSpeed}
          onHpaTarget={changeHpaTarget}
        />

        {/* TOP: the comparison (chart) + decision */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="space-y-2 lg:col-span-2">
            <LiveChart data={history} maxPods={config?.max_pods ?? 20} showHpa={!isLive} />
            <p className="px-1 text-xs text-slate-500">
              {isLive
                ? "Live Kubernetes cluster: the agent reads real pod CPU and scales the deployment. It is the only autoscaler — to compare against real HPA, use the Stage-2 experiment."
                : "Simulation on a held-out real Alibaba trace. The RL agent and HPA each drive their own copy of the workload."}
            </p>
          </div>
          <DecisionPanel s={state} mode={isLive ? mode : "autopilot"} live={isLive} />
        </div>

        {/* MIDDLE (live only): cluster details + traffic generator */}
        {isLive && (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <ClusterInfoPanel info={cluster} />
            <LoadControl status={load} onStart={startLoad} onStop={stopLoad} />
          </div>
        )}

        {/* BOTTOM: impact — live shows the agent's own real metrics; sim shows vs-HPA savings */}
        {isLive ? (
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <KpiCard
              label="Current replicas"
              value={`${st ? st.replicas : (state?.rl.pods ?? 0)}`}
              sub={`peak ${st?.peak_pods ?? state?.rl.pods ?? 0} this session`}
              icon={<Server size={15} />}
              tone="green"
            />
            <KpiCard
              label="Avg pod CPU"
              value={`${Math.round(state?.avg_cpu_millicores ?? 0)}m`}
              sub={`${Math.round((state?.cpu ?? 0) * 100)}% of one core`}
              icon={<Cpu size={15} />}
              tone="slate"
            />
            <KpiCard
              label="Scaling actions"
              value={`${st ? st.scaling_actions : 0}`}
              sub={`${st?.up ?? 0} up · ${st?.down ?? 0} down`}
              icon={<Activity size={15} />}
              tone="slate"
            />
            <KpiCard
              label="Mode"
              value={killed ? "Paused" : mode === "autopilot" ? "Autopilot" : "Advisory"}
              sub={state?.applied ? "scaling the cluster" : "read-only"}
              icon={<ShieldCheck size={15} />}
              tone={mode === "autopilot" && !killed ? "green" : "amber"}
            />
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <KpiCard
              label="Pods saved (cumulative)"
              value={`${sv ? sv.pod_ticks_saved.toFixed(0) : 0}`}
              sub="pod-ticks vs HPA"
              icon={<Server size={15} />}
              tone={sv && sv.pod_ticks_saved >= 0 ? "green" : "amber"}
            />
            <KpiCard
              label="Energy saved"
              value={`${sv ? sv.kwh.toFixed(2) : "0.00"} kWh`}
              sub="@ 50 W / pod"
              icon={<Leaf size={15} />}
              tone={sv && sv.kwh >= 0 ? "green" : "amber"}
            />
            <KpiCard
              label="Cost saved"
              value={`${sv ? sv.frw.toFixed(0) : 0} Frw`}
              sub="@ 175 Frw / kWh"
              icon={<Wallet size={15} />}
              tone={sv && sv.frw >= 0 ? "green" : "amber"}
            />
            <KpiCard
              label="SLA breaches avoided"
              value={`${sv ? sv.breaches_avoided : 0}`}
              sub="vs HPA over the day"
              icon={<ShieldCheck size={15} />}
              tone="slate"
            />
          </div>
        )}
        </>
        )}
      </main>
    </div>
  );
}
