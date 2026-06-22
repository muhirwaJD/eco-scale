import { useCallback, useEffect, useRef, useState } from "react";
import { Leaf, Server, ShieldCheck, Wallet } from "lucide-react";
import Header from "./components/Header";
import KpiCard from "./components/KpiCard";
import Controls from "./components/Controls";
import LiveChart, { type ChartPoint } from "./components/LiveChart";
import DecisionPanel from "./components/DecisionPanel";
import { getConfig, simReset, simStep } from "./api";
import type { Config, Mode, SimState } from "./types";

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

  const timer = useRef<number | null>(null);

  const toPoint = (s: SimState): ChartPoint => ({
    hour: +((s.tick / s.max_ticks) * 24).toFixed(2),
    load: Math.round(s.cpu * 100),
    rlPods: s.rl.pods,
    hpaPods: s.hpa.pods,
  });

  const reset = useCallback(
    async (target = hpaTarget) => {
      setRunning(false);
      try {
        const s = await simReset(target);
        setState(s);
        setHistory([toPoint(s)]);
        setError(null);
      } catch (e) {
        setError("Cannot reach the API. Start it with: uvicorn serving.api:app");
      }
    },
    [hpaTarget]
  );

  // initial load
  useEffect(() => {
    getConfig().then(setConfig).catch(() => {});
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // play loop
  useEffect(() => {
    if (!running) return;
    timer.current = window.setInterval(async () => {
      try {
        const s = await simStep();
        setState(s);
        setHistory((h) => [...h, toPoint(s)]);
        if (s.done) setRunning(false);
      } catch {
        setRunning(false);
        setError("Lost connection to the API.");
      }
    }, speed);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [running, speed]);

  const changeHpaTarget = (t: number) => {
    setHpaTarget(t);
    reset(t);
  };

  const sv = state?.savings;
  const podDelta = state ? state.hpa.pods - state.rl.pods : 0;

  return (
    <div className="min-h-screen">
      <Header
        config={config}
        mode={mode}
        killed={killed}
        onMode={setMode}
        onKill={() => setKilled((k) => !k)}
      />

      <main className="mx-auto max-w-7xl space-y-4 p-6">
        {error && (
          <div className="rounded-lg border border-eco-red/40 bg-eco-red/10 px-4 py-3 text-sm text-eco-red">
            {error}
          </div>
        )}

        {/* KPI row — cumulative value vs the chosen HPA baseline */}
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

        {/* main grid: chart + decision */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <Controls
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
            <LiveChart data={history} maxPods={config?.max_pods ?? 20} />
            <p className="px-1 text-xs text-slate-500">
              Live simulation on a held-out real Alibaba trace. The RL agent and HPA each
              drive their own copy of the workload; {podDelta >= 0 ? "fewer" : "more"} pods
              is {podDelta >= 0 ? "better" : "the latency trade-off"}.
            </p>
          </div>

          <DecisionPanel s={state} mode={mode} />
        </div>
      </main>
    </div>
  );
}
