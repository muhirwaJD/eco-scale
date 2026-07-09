import { useCallback, useEffect, useRef, useState } from "react";
import {
  ResponsiveContainer, ComposedChart, LineChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip,
} from "recharts";
import {
  Play, Pause, RotateCcw, Clock, Beaker, Radio, CircleDot, Server, Cpu, Activity,
  Zap, TrendingDown, Leaf, Coins, ShieldCheck, Gauge,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, Pill, Stat, LegendDot, MiniStat, ProbBar, ActionBadge, type Action } from "@/components/eco/primitives";
import { VariableBar } from "@/components/eco/time-range";

type Var = { key: string; value: string; options: string[] };
import type { ClusterInfo, Config, ExperimentStatus, LoadStatus, SimState } from "@/types";
import {
  simReset, simStep, liveReset, liveStep, liveInfo, loadStart, loadStop, loadStatus,
  experimentStart, experimentStatus, experimentStop,
} from "@/api";

type Mode = "sim" | "live" | "ab";

const ACT: Record<string, Action> = { scale_up: "up", scale_down: "down", maintain: "hold" };

// ---------- shared chart ----------
interface CP { t: number; load: number; rl: number; hpa?: number }

function ScalingChart({ data, showHPA, title, subtitle, fill }: { data: CP[]; showHPA: boolean; title: string; subtitle: string; fill?: boolean }) {
  return (
    <Card className={cn("p-4", fill ? "flex h-full min-h-[340px] flex-col" : "h-[340px]")}>
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">{title}</h3>
          <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <LegendDot color="var(--color-load)" label="Load %" muted />
          <LegendDot color="var(--color-agent)" label="RL agent" />
          {showHPA && <LegendDot color="var(--color-hpa)" label="HPA" dashed />}
        </div>
      </div>
      <div className={cn(fill ? "min-h-0 flex-1" : "h-[260px]")}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -10 }}>
            <defs>
              <linearGradient id="loadFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-load)" stopOpacity={0.25} />
                <stop offset="100%" stopColor="var(--color-load)" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => `${Number(v).toFixed(0)}h`}
              stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--color-border)" }} />
            <YAxis yAxisId="pods" domain={[0, 20]} stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={36} />
            <YAxis yAxisId="load" orientation="right" domain={[0, 100]} tickFormatter={(v) => `${v}%`}
              stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
            <Tooltip contentStyle={{ background: "var(--color-popover)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }}
              labelFormatter={(v) => `t = ${Number(v).toFixed(2)} h`} />
            <Area yAxisId="load" type="monotone" dataKey="load" name="Load %" stroke="var(--color-load)" strokeWidth={1} fill="url(#loadFill)" isAnimationActive={false} />
            {showHPA && <Line yAxisId="pods" type="stepAfter" dataKey="hpa" name="HPA" stroke="var(--color-hpa)" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls />}
            <Line yAxisId="pods" type="stepAfter" dataKey="rl" name="RL agent" stroke="var(--color-agent)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

// ---------- decision panel ----------
function DecisionPanel({ s, showHPACompare }: { s: SimState | null; showHPACompare: boolean }) {
  const action: Action = s ? (ACT[s.rl.action_name] ?? "hold") : "hold";
  const p = s?.rl.probs ?? null;
  const obs = s?.rl.observation ?? [0, 0, 0, 0];
  return (
    <Card className="flex flex-col">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Agent decision</h3>
        <span className="text-[11px] text-muted-foreground">PPO · interval 15s</span>
      </div>
      <div className="mt-3">
        <ActionBadge action={action} />
        <p className="mt-2 text-sm leading-snug text-muted-foreground">{s?.rl.rationale ?? "Waiting for the next decision…"}</p>
      </div>
      {p && (
        <div className="mt-3 space-y-1.5">
          <ProbBar label="down" value={p[0]} tone="hpa" />
          <ProbBar label="hold" value={p[1]} tone="muted" />
          <ProbBar label="up" value={p[2]} tone="agent" />
        </div>
      )}
      <div className="mt-3 grid grid-cols-2 gap-2">
        <MiniStat label="CPU load" val={`${Math.round(obs[0] * 100)}%`} />
        <MiniStat label="Pods" val={s?.rl.pods ?? 0} />
        <MiniStat label="Queue" val={`${Math.round(obs[2] * 100)}%`} />
        <MiniStat label="Time of day" val={`${(obs[3] * 24).toFixed(1)}h`} />
      </div>
      {showHPACompare && s?.hpa && (
        <div className="mt-3 grid grid-cols-2 gap-2 rounded-lg border border-border bg-background/40 p-2">
          <MiniStat label="RL pods" val={s.rl.pods} tone="agent" />
          <MiniStat label={`HPA@${Math.round(s.hpa.target * 100)}%`} val={s.hpa.pods} tone="hpa" />
          <MiniStat label="RL latency" val={`${Math.round((s.rl.latency ?? 0) * 100)}%`} tone="agent" />
          <MiniStat label="HPA latency" val={`${Math.round(s.hpa.latency * 100)}%`} tone="hpa" />
        </div>
      )}
    </Card>
  );
}

// ---------- mode switcher ----------
const MODES: { id: Mode; label: string; pill: React.ReactNode }[] = [
  { id: "sim", label: "Simulation", pill: <Pill tone="warn"><Beaker className="h-3 w-3" /> SIMULATION — replaying a real Alibaba trace</Pill> },
  { id: "live", label: "Live cluster", pill: <Pill tone="live"><CircleDot className="h-3 w-3 animate-pulse" /> LIVE — eco-sample-app</Pill> },
  { id: "ab", label: "Benchmark", pill: <Pill tone="agent"><Radio className="h-3 w-3" /> BENCHMARK — agent vs native Kubernetes HPA</Pill> },
];

function ModeSwitcher({ mode, setMode, liveOk, vars, onVarChange }: {
  mode: Mode; setMode: (m: Mode) => void; liveOk: boolean;
  vars: Var[]; onVarChange: (key: string, value: string) => void;
}) {
  const active = MODES.find((m) => m.id === mode)!;
  // cluster/namespace/deployment only affect the real cluster — hide in Simulation.
  const showFilters = mode !== "sim";
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="inline-flex rounded-xl border border-border bg-card p-1">
        {MODES.map((m) => {
          const disabled = m.id !== "sim" && !liveOk;
          return (
            <button key={m.id} disabled={disabled} onClick={() => setMode(m.id)}
              title={disabled ? "No reachable cluster" : ""}
              className={cn("rounded-lg px-4 py-1.5 text-sm font-medium transition",
                mode === m.id ? "bg-secondary text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
                disabled && "cursor-not-allowed opacity-40 hover:text-muted-foreground")}>
              {m.label}
            </button>
          );
        })}
      </div>
      {showFilters && <span className="hidden h-6 w-px bg-border sm:block" />}
      {showFilters && <VariableBar vars={vars} onChange={onVarChange} />}
      <span className="ml-auto">{active.pill}</span>
    </div>
  );
}

// ---------- SIMULATION ----------
// Plain-language meaning of each comparison target, for beginners.
const HPA_TARGET_HELP: Record<number, { tag: string; tone: "agent" | "hpa" | "warn"; blurb: string }> = {
  0.5: { tag: "Conservative", tone: "agent",
    blurb: "HPA keeps pods only ~50% busy, so it adds replicas early and holds lots of spare headroom. Safest for latency, most wasteful — the agent's savings look largest against this." },
  0.6: { tag: "Balanced", tone: "hpa",
    blurb: "A reasonably tuned HPA. Adds capacity at moderate load — the fairest head-to-head against the agent." },
  0.7: { tag: "Balanced", tone: "hpa",
    blurb: "A reasonably tuned HPA. Runs pods a bit hotter before scaling out — still a fair comparison, slightly cheaper than 60%." },
  0.9: { tag: "Aggressive", tone: "warn",
    blurb: "HPA packs pods near full (90%) before adding more. Cheapest, but it reacts late — under bursty load it risks SLA breaches." },
};

function HpaTargetHelp({ target, agentTarget }: { target: number; agentTarget?: number }) {
  const h = HPA_TARGET_HELP[target] ?? HPA_TARGET_HELP[0.6];
  const dot = h.tone === "agent" ? "bg-agent" : h.tone === "warn" ? "bg-warn" : "bg-hpa";
  return (
    <Card className="space-y-2">
      <div className="flex items-center gap-2">
        <span className={cn("h-2 w-2 rounded-full", dot)} />
        <h3 className="text-sm font-semibold">HPA @ {Math.round(target * 100)}% · {h.tag}</h3>
      </div>
      <p className="text-xs leading-relaxed text-muted-foreground">{h.blurb}</p>
      <p className="border-t border-border pt-2 text-[11px] leading-relaxed text-muted-foreground">
        This is the <span className="text-hpa">HPA baseline</span> the agent is racing.
        {agentTarget != null && (
          <> The <span className="text-agent">PPO agent</span> itself aims for ~{Math.round(agentTarget * 100)}% utilization,
          but learns <em>when</em> to scale from the traffic pattern — not a fixed threshold.</>
        )}
      </p>
    </Card>
  );
}

function SimulationView({ maxPods, agentTarget }: { maxPods: number; agentTarget?: number }) {
  const [target, setTarget] = useState(0.5);
  const [speed, setSpeed] = useState(2);
  const [playing, setPlaying] = useState(false);
  const [state, setState] = useState<SimState | null>(null);
  const [hist, setHist] = useState<CP[]>([]);
  const timer = useRef<number | null>(null);

  const toCP = (s: SimState): CP => ({ t: +((s.tick / s.max_ticks) * 24).toFixed(2), load: Math.round(s.cpu * 100), rl: s.rl.pods, hpa: s.hpa?.pods });

  const reset = useCallback(async (tg = target) => {
    setPlaying(false);
    const s = await simReset(tg);
    setState(s); setHist([toCP(s)]);
  }, [target]);

  useEffect(() => { reset(); /* eslint-disable-next-line */ }, []);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(async () => {
      const s = await simStep();
      setState(s); setHist((h) => [...h, toCP(s)]);
      if (s.done) setPlaying(false);
    }, [500, 250, 100][[1, 2, 4].indexOf(speed)] ?? 250);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, speed]);

  const sv = state?.savings;
  const cursor = state ? (state.tick / state.max_ticks) * 24 : 0;

  return (
    <div className="space-y-4">
      <Card className="flex flex-wrap items-center gap-3">
        <button onClick={() => setPlaying((p) => !p)}
          className={cn("inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition",
            playing ? "bg-agent text-agent-foreground hover:opacity-90" : "bg-agent/20 text-agent hover:bg-agent/30")}>
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}{playing ? "Pause" : "Play"}
        </button>
        <button onClick={() => reset()} className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground">
          <RotateCcw className="h-4 w-4" /> Reset
        </button>
        <div className="flex items-center rounded-lg border border-border bg-card p-1">
          {[1, 2, 4].map((s) => (
            <button key={s} onClick={() => setSpeed(s)}
              className={cn("rounded-md px-2.5 py-1 text-xs font-medium", speed === s ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground")}>{s}×</button>
          ))}
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground">Compare vs HPA target</span>
          <select value={target} onChange={(e) => { const t = parseFloat(e.target.value); setTarget(t); reset(t); }}
            className="rounded-md border border-border bg-card px-2 py-1 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring">
            <option value={0.5}>50%</option><option value={0.6}>60%</option><option value={0.7}>70%</option><option value={0.9}>90%</option>
          </select>
        </div>
        <div className="ml-auto inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs tabular-nums text-muted-foreground">
          <Clock className="h-3.5 w-3.5" /> day {cursor.toFixed(1)}h / 24h
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.7fr_1fr]">
        <ScalingChart data={hist} showHPA fill title="Live replicas — RL agent vs HPA" subtitle="Pods (left axis) · Load % (right axis) · 24h trace" />
        <div className="space-y-4">
          <DecisionPanel s={state} showHPACompare />
          <HpaTargetHelp target={target} agentTarget={agentTarget} />
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4 lg:grid-cols-4">
        <Stat tone="positive" label="Pods saved" value={sv ? sv.pod_ticks_saved.toFixed(0) : 0} sub="pod-intervals vs HPA" icon={<TrendingDown className="h-4 w-4" />} />
        <Stat tone="positive" label="Energy saved" value={`${sv ? sv.kwh.toFixed(2) : "0.00"} kWh`} sub="@ 50 W / pod" icon={<Leaf className="h-4 w-4" />} />
        <Stat tone="positive" label="Cost saved" value={`${sv ? sv.frw.toFixed(0) : 0} Frw`} sub="@ 175 Frw / kWh" icon={<Coins className="h-4 w-4" />} />
        <Stat tone="positive" label="SLA breaches avoided" value={sv ? sv.breaches_avoided : 0} sub="p95 < 1.0 vs HPA" icon={<ShieldCheck className="h-4 w-4" />} />
      </div>
    </div>
  );
}

// ---------- LIVE ----------
function LiveView({ apply }: { apply: boolean }) {
  const [state, setState] = useState<SimState | null>(null);
  const [hist, setHist] = useState<CP[]>([]);
  const [cluster, setCluster] = useState<ClusterInfo | null>(null);
  const [load, setLoad] = useState<LoadStatus | null>(null);
  const [playing, setPlaying] = useState(false);
  const applyRef = useRef(apply); applyRef.current = apply;

  const toCP = (s: SimState): CP => ({ t: +((s.tick / s.max_ticks) * 24).toFixed(2), load: Math.round(s.cpu * 100), rl: s.rl.pods });

  useEffect(() => { liveReset().then((s) => { setState(s); setHist([toCP(s)]); }).catch(() => {}); }, []);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(async () => {
      const s = await liveStep(applyRef.current);
      setState(s); setHist((h) => [...h.slice(-95), toCP(s)]);
    }, 3000);
    return () => window.clearInterval(id);
  }, [playing]);

  useEffect(() => {
    const r = () => { liveInfo().then(setCluster).catch(() => {}); loadStatus().then(setLoad).catch(() => {}); };
    r(); const id = window.setInterval(r, 2500); return () => window.clearInterval(id);
  }, []);

  const st = state?.stats;
  return (
    <div className="space-y-4">
      <Card className="flex flex-wrap items-center gap-3">
        <button onClick={() => setPlaying((p) => !p)}
          className={cn("inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition",
            playing ? "bg-agent text-agent-foreground hover:opacity-90" : "bg-agent/20 text-agent hover:bg-agent/30")}>
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}{playing ? "Pause" : "Play"}
        </button>
        <span className="text-xs text-muted-foreground">
          Agent {apply ? "is scaling the cluster (autopilot)" : "is in recommend-only mode"} · reads kubectl every 15s
        </span>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.7fr_1fr]">
        <ScalingChart data={hist} showHPA={false} title="RL agent on the cluster" subtitle="Live replicas vs load · streaming" />
        <DecisionPanel s={state} showHPACompare={false} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.4fr_1fr]">
        <ClusterPanel info={cluster} />
        <TrafficPanel load={load} onStart={() => loadStart(300).then(setLoad)} onStop={() => loadStop().then(setLoad)} />
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Current replicas" value={st?.replicas ?? state?.rl.pods ?? 0} sub={`peak ${st?.peak_pods ?? 0} this session`} icon={<Server className="h-4 w-4" />} />
        <Stat label="Avg pod CPU" value={`${Math.round(state?.avg_cpu_millicores ?? 0)}m`} sub="across live pods" icon={<Cpu className="h-4 w-4" />} />
        <Stat label="Scaling actions" value={st?.scaling_actions ?? 0} sub={`${st?.up ?? 0} up · ${st?.down ?? 0} down`} icon={<Activity className="h-4 w-4" />} />
        <Stat label="Mode" value={<span className="text-agent">{apply ? "Autopilot" : "Advisory"}</span>} sub={apply ? "agent is sole scaler" : "read-only"} icon={<Zap className="h-4 w-4" />} />
      </div>
    </div>
  );
}

function ClusterPanel({ info }: { info: ClusterInfo | null }) {
  const rows: [string, string][] = info ? [
    ["Context", info.context], ["Namespace", info.namespace], ["Deployment", info.deployment],
    ["Image", info.image], ["Native HPA", info.native_hpa ? "present" : "none (sole scaler)"],
    ["Replicas", `${info.replicas} · min ${info.min_pods} / max ${info.max_pods}`],
  ] : [];
  return (
    <Card>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Cluster</h3>
        <Pill tone="live"><span className="h-1.5 w-1.5 rounded-full bg-agent" /> connected</Pill>
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 border-b border-border pb-3 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3">
            <span className="text-muted-foreground">{k}</span>
            <span className="truncate font-medium text-foreground" title={v}>{v}</span>
          </div>
        ))}
      </div>
      <div className="mt-3">
        <div className="mb-1.5 text-[11px] uppercase tracking-wider text-muted-foreground">Pods</div>
        <div className="max-h-[160px] space-y-1 overflow-auto pr-1">
          {info?.pods.map((p) => (
            <div key={p.name} className="flex items-center justify-between rounded-md border border-border bg-background/40 px-2.5 py-1.5 text-xs">
              <div className="flex items-center gap-2 truncate">
                <span className={cn("h-1.5 w-1.5 rounded-full", p.phase === "Running" ? "bg-agent" : "bg-warn")} />
                <span className="truncate font-mono text-muted-foreground">{p.name}</span>
              </div>
              <span className="tabular-nums text-foreground">{p.cpu}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

function TrafficPanel({ load, onStart, onStop }: { load: LoadStatus | null; onStart: () => void; onStop: () => void }) {
  const on = load?.running ?? false;
  const intensity = Math.round((load?.intensity ?? 0) * 100);
  return (
    <Card>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Traffic generator</h3>
        <span className="text-[11px] text-muted-foreground">HTTP wave · /work</span>
      </div>
      <button onClick={on ? onStop : onStart}
        className={cn("inline-flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold transition",
          on ? "bg-destructive/20 text-destructive hover:bg-destructive/30" : "bg-agent text-agent-foreground hover:opacity-90")}>
        {on ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}{on ? "Stop traffic" : "Start traffic"}
      </button>
      <div className="mt-4">
        <div className="mb-1.5 flex justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
          <span>Intensity</span><span className="tabular-nums text-foreground">{intensity}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-muted">
          <div className="h-full rounded-full bg-gradient-to-r from-agent via-warn to-destructive transition-[width] duration-700" style={{ width: `${intensity}%` }} />
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2">
        <MiniStat label="measured p95" val={load?.p95_ms ? `${load.p95_ms.toFixed(0)} ms` : "—"} tone="agent" />
        <MiniStat label="SLO" val="≤ 1000 ms" />
      </div>
    </Card>
  );
}

// ---------- BENCHMARK (A/B) ----------
function BenchmarkView() {
  const [st, setSt] = useState<ExperimentStatus | null>(null);
  const [phaseSecs, setPhaseSecs] = useState(120);
  const poll = useRef<number | null>(null);
  useEffect(() => {
    const t = () => experimentStatus().then(setSt).catch(() => {});
    t(); poll.current = window.setInterval(t, 2000);
    return () => { if (poll.current) window.clearInterval(poll.current); };
  }, []);
  const running = st?.state === "running";
  const done = st?.state === "done";
  const liveData = (st && running && st.phase ? st[st.phase] : []).map((p) => ({ t: p.t, v: p.replicas }));
  const PHASE = st?.phase === "hpa" ? "Phase 2/2 — real HPA driving" : "Phase 1/2 — RL agent driving";

  return (
    <div className="space-y-4">
      <Card className="border-agent/25 bg-agent/[0.04]">
        <div className="flex items-start gap-3">
          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-agent/15 ring-1 ring-agent/30"><Beaker className="h-4 w-4 text-agent" /></div>
          <div className="flex-1">
            <h3 className="text-sm font-semibold">Benchmark — RL agent vs native Kubernetes HPA</h3>
            <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
              Runs the real head-to-head: the agent drives the deployment under a traffic wave, then the real native HPA does, then compares pod usage and p95 latency. Live cluster, real kubectl scale calls — not a simulation.
            </p>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button onClick={() => experimentStart(phaseSecs).then(setSt)} disabled={running}
            className={cn("inline-flex items-center gap-2 rounded-lg bg-agent px-4 py-2 text-sm font-semibold text-agent-foreground transition hover:opacity-90", running && "cursor-not-allowed opacity-60")}>
            <Play className="h-4 w-4" /> {running ? "Running…" : "Run benchmark"}
          </button>
          {running && <button onClick={() => experimentStop().then(setSt)} className="rounded-lg bg-destructive/20 px-4 py-2 text-sm font-semibold text-destructive hover:bg-destructive/30">Stop</button>}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Seconds per phase</span>
            <select value={phaseSecs} onChange={(e) => setPhaseSecs(parseInt(e.target.value))} disabled={running}
              className="rounded-md border border-border bg-card px-2 py-1 text-sm disabled:opacity-50">
              {[60, 120, 180, 240].map((s) => <option key={s} value={s}>{s}s</option>)}
            </select>
          </div>
        </div>
      </Card>

      {running && st && (
        <Card>
          <div className="flex items-center justify-between">
            <Pill tone={st.phase === "hpa" ? "hpa" : "agent"}><CircleDot className="h-3 w-3 animate-pulse" /> {PHASE}</Pill>
            <span className="text-xs tabular-nums text-muted-foreground">{st.elapsed.toFixed(0)}s / {st.duration}s</span>
          </div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
            <div className={cn("h-full rounded-full transition-[width] duration-200", st.phase === "hpa" ? "bg-destructive" : "bg-agent")}
              style={{ width: `${Math.min(100, (st.elapsed / st.duration) * 100)}%` }} />
          </div>
          <div className="mt-4 h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={liveData} margin={{ top: 8, right: 8, bottom: 4, left: -10 }}>
                <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="t" tickFormatter={(v) => `${Math.round(v)}s`} stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} />
                <YAxis domain={[0, 12]} stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={30} />
                <Tooltip contentStyle={{ background: "var(--color-popover)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }} />
                <Line type="stepAfter" dataKey="v" stroke={st.phase === "hpa" ? "var(--color-hpa)" : "var(--color-agent)"} strokeWidth={2.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {done && st?.summary && <ABResults st={st} />}

      {!running && !done && (
        <Card className="grid place-items-center py-16 text-center">
          <Gauge className="h-10 w-10 text-muted-foreground" />
          <p className="mt-3 max-w-sm text-sm text-muted-foreground">No experiment running. Hit “Run benchmark” to drive the cluster with the agent first, then the native HPA, then see the head-to-head verdict.</p>
        </Card>
      )}
    </div>
  );
}

function ABResults({ st }: { st: ExperimentStatus }) {
  const n = Math.max(st.rl.length, st.hpa.length);
  const data = Array.from({ length: n }, (_, i) => ({ t: st.rl[i]?.t ?? st.hpa[i]?.t ?? i, rl: st.rl[i]?.replicas, hpa: st.hpa[i]?.replicas }));
  const s = st.summary!;
  const v = s.verdict;
  return (
    <div className="space-y-4">
      <Card className="h-[360px] p-4">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold">Head-to-head — replicas over time</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">Same traffic wave, applied to each scaler in sequence</p>
          </div>
          <div className="flex items-center gap-3 text-[11px]">
            <LegendDot color="var(--color-agent)" label="RL agent" />
            <LegendDot color="var(--color-hpa)" label="real HPA" dashed />
          </div>
        </div>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -10 }}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="t" tickFormatter={(v) => `${Math.round(v)}s`} stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} />
              <YAxis domain={[0, 12]} stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={30} />
              <Tooltip contentStyle={{ background: "var(--color-popover)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }} />
              <Line type="stepAfter" dataKey="rl" stroke="var(--color-agent)" strokeWidth={2.5} dot={false} connectNulls isAnimationActive={false} />
              <Line type="stepAfter" dataKey="hpa" stroke="var(--color-hpa)" strokeWidth={2} strokeDasharray="5 4" dot={false} connectNulls isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.2fr_1fr]">
        <Card>
          <h3 className="mb-3 text-sm font-semibold">Summary</h3>
          <div className="overflow-hidden rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-[11px] uppercase tracking-wider text-muted-foreground">
                <tr><th className="px-3 py-2 text-left font-medium">Scaler</th><th className="px-3 py-2 text-right font-medium">Avg pods</th><th className="px-3 py-2 text-right font-medium">Max pods</th><th className="px-3 py-2 text-right font-medium">p95 latency</th></tr>
              </thead>
              <tbody className="divide-y divide-border">
                <tr className="bg-agent/[0.04]">
                  <td className="px-3 py-2.5 font-medium text-agent">RL agent</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{s.rl.avg_pods}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{s.rl.max_pods}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{s.rl.p95_ms} ms</td>
                </tr>
                <tr>
                  <td className="px-3 py-2.5 font-medium text-destructive">real HPA</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{s.hpa.avg_pods}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{s.hpa.max_pods}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{s.hpa.p95_ms} ms</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>
        <Card className="border-agent/30 bg-agent/[0.05]">
          <div className="flex items-start gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-agent/20 ring-1 ring-agent/40"><ShieldCheck className="h-4 w-4 text-agent" /></div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-agent">Verdict</div>
              <p className="mt-1 text-base font-medium leading-snug text-foreground">
                {v && v.rl_leaner
                  ? <>The agent used <span className="text-agent">~{Math.abs(v.pod_saving_pct)}% fewer pods</span> than the real HPA over the same traffic.</>
                  : <>The agent used <span className="text-warn">~{Math.abs(v?.pod_saving_pct ?? 0)}% more pods</span> than the real HPA.</>}
                {" "}Single run on a single-node cluster — a feasibility comparison, not a statistical test.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default function DashboardsSection({ mode, setMode, liveOk, apply, config, vars, onVarChange }: {
  mode: Mode; setMode: (m: Mode) => void; liveOk: boolean; apply: boolean; config: Config | null;
  vars: Var[]; onVarChange: (key: string, value: string) => void;
}) {
  return (
    <>
      <ModeSwitcher mode={mode} setMode={setMode} liveOk={liveOk} vars={vars} onVarChange={onVarChange} />
      <div className="mt-6">
        {mode === "sim" && <SimulationView maxPods={config?.max_pods ?? 20} agentTarget={config?.util_target} />}
        {mode === "live" && <LiveView apply={apply} />}
        {mode === "ab" && <BenchmarkView />}
      </div>
    </>
  );
}

export type { Mode };
