import { useEffect, useRef, useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { FlaskConical, Square } from "lucide-react";
import { experimentStart, experimentStatus, experimentStop } from "../api";
import type { ExperimentStatus } from "../types";

const PHASE_LABEL: Record<string, string> = {
  rl: "Phase 1/2 — RL agent driving the cluster",
  hpa: "Phase 2/2 — real Kubernetes HPA driving the cluster",
};

export default function ExperimentPanel() {
  const [st, setSt] = useState<ExperimentStatus | null>(null);
  const [duration, setDuration] = useState(120);
  const poll = useRef<number | null>(null);

  useEffect(() => {
    const tick = () => experimentStatus().then(setSt).catch(() => {});
    tick();
    poll.current = window.setInterval(tick, 2000);
    return () => { if (poll.current) window.clearInterval(poll.current); };
  }, []);

  const running = st?.state === "running";
  const done = st?.state === "done";

  const start = () => experimentStart(duration).then(setSt).catch(() => {});
  const stop = () => experimentStop().then(setSt).catch(() => {});

  const liveData = (st && running && st.phase ? st[st.phase] : []).map((p) => ({
    t: p.t, replicas: p.replicas, load: Math.round(p.intensity * 100),
  }));

  const n = Math.max(st?.rl.length ?? 0, st?.hpa.length ?? 0);
  const cmpData = Array.from({ length: n }, (_, i) => ({
    t: st?.rl[i]?.t ?? st?.hpa[i]?.t ?? i,
    rlPods: st?.rl[i]?.replicas,
    hpaPods: st?.hpa[i]?.replicas,
    load: Math.round((st?.rl[i]?.intensity ?? st?.hpa[i]?.intensity ?? 0) * 100),
  }));

  return (
    <div className="space-y-4">
      {/* intro + run control */}
      <div className="card p-5">
        <div className="mb-1 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FlaskConical size={16} className="text-eco-light" />
            <h2 className="text-sm font-semibold text-slate-100">
              Benchmark — RL agent vs native Kubernetes HPA
            </h2>
          </div>
          <span className="label">real cluster · sequential</span>
        </div>
        <p className="mb-4 text-xs leading-relaxed text-slate-400">
          Runs the real head-to-head on your cluster: the agent drives the deployment under a
          traffic wave, then the <span className="font-medium text-slate-200">real native
          HPA</span> drives it under the same wave. Everything is real — real pods, real CPU,
          real HPA, real latency. ~{Math.round((duration * 2) / 60) + 1} min total.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          {!running ? (
            <button
              onClick={start}
              className="flex items-center gap-2 rounded-xl bg-eco-green px-5 py-2 text-sm font-semibold text-[#06210f] transition-all hover:bg-eco-light active:scale-95"
            >
              <FlaskConical size={15} /> Run benchmark
            </button>
          ) : (
            <button
              onClick={stop}
              className="flex items-center gap-2 rounded-xl bg-eco-red/15 px-5 py-2 text-sm font-semibold text-eco-red ring-1 ring-eco-red/30 transition-all hover:bg-eco-red/25"
            >
              <Square size={14} /> Stop
            </button>
          )}
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span>seconds per phase</span>
            <select
              value={duration}
              disabled={running}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-slate-200 outline-none transition-colors hover:border-white/20 focus:border-eco-green/50 disabled:opacity-40"
            >
              <option value={60}>60</option>
              <option value={120}>120</option>
              <option value={180}>180</option>
              <option value={240}>240</option>
            </select>
          </div>
          {st?.state === "error" && (
            <span className="text-xs text-eco-red">Error: {st.message}</span>
          )}
        </div>
      </div>

      {/* running: phase badge + progress + live chart */}
      {running && st && (
        <div className="card p-5 ring-1 ring-eco-green/20">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-2 text-sm font-medium text-eco-light">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-eco-light" />
              {st.phase ? PHASE_LABEL[st.phase] : "Starting…"}
            </span>
            <span className="text-xs tabular-nums text-slate-400">
              {st.elapsed.toFixed(0)}s / {st.duration}s
            </span>
          </div>
          <div className="mb-4 h-2 overflow-hidden rounded-full bg-white/[0.06]">
            <div
              className="h-full rounded-full bg-eco-green transition-all duration-500"
              style={{ width: `${Math.min(100, (st.elapsed / st.duration) * 100)}%` }}
            />
          </div>
          <PhaseChart data={liveData} color={st.phase === "hpa" ? "#f87171" : "#4ade80"} />
        </div>
      )}

      {/* done: comparison + summary */}
      {done && st?.summary && (
        <>
          <div className="card p-5">
            <h3 className="mb-1 text-sm font-semibold text-slate-100">
              Result — replicas over the same traffic wave
            </h3>
            <p className="mb-3 text-xs text-slate-500">RL agent vs the real native HPA</p>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={cmpData} margin={{ top: 8, right: 8, bottom: 4, left: -10 }}>
                <defs>
                  <linearGradient id="expLoad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#64748b" stopOpacity={0.28} />
                    <stop offset="100%" stopColor="#64748b" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="t" stroke="#475569" tick={{ fill: "#64748b" }} fontSize={11}
                  tickLine={false} axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                  tickFormatter={(t) => `${Math.round(t)}s`} />
                <YAxis yAxisId="pods" stroke="#475569" tick={{ fill: "#64748b" }} fontSize={11}
                  tickLine={false} axisLine={false} />
                <YAxis yAxisId="load" orientation="right" stroke="#475569" tick={{ fill: "#64748b" }}
                  fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} unit="%" />
                <Tooltip contentStyle={{ background: "rgba(13,18,26,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 12 }} labelStyle={{ color: "#94a3b8" }} />
                <Area yAxisId="load" type="monotone" dataKey="load" name="Load %" stroke="#64748b" strokeWidth={1} fill="url(#expLoad)" isAnimationActive={false} />
                <Line yAxisId="pods" type="stepAfter" dataKey="hpaPods" name="real HPA"
                  stroke="#f87171" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls />
                <Line yAxisId="pods" type="stepAfter" dataKey="rlPods" name="RL agent"
                  stroke="#4ade80" strokeWidth={2.75} dot={false} isAnimationActive={false} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="mt-2 flex items-center gap-4 text-xs">
              <Legend2 color="#4ade80" label="RL agent" />
              <Legend2 color="#f87171" label="real HPA" dashed />
            </div>
          </div>

          <SummaryTable st={st} />
        </>
      )}
    </div>
  );
}

function PhaseChart({ data, color }: { data: any[]; color: string }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -10 }}>
        <defs>
          <linearGradient id="phaseLoad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#64748b" stopOpacity={0.28} />
            <stop offset="100%" stopColor="#64748b" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis dataKey="t" stroke="#475569" tick={{ fill: "#64748b" }} fontSize={11}
          tickLine={false} axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
          tickFormatter={(t) => `${Math.round(t)}s`} />
        <YAxis yAxisId="pods" stroke="#475569" tick={{ fill: "#64748b" }} fontSize={11} tickLine={false} axisLine={false} />
        <YAxis yAxisId="load" orientation="right" stroke="#475569" tick={{ fill: "#64748b" }} fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} unit="%" />
        <Tooltip contentStyle={{ background: "rgba(13,18,26,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, fontSize: 12 }} labelStyle={{ color: "#94a3b8" }} />
        <Area yAxisId="load" type="monotone" dataKey="load" name="Load %" stroke="#64748b" strokeWidth={1} fill="url(#phaseLoad)" isAnimationActive={false} />
        <Line yAxisId="pods" type="stepAfter" dataKey="replicas" name="replicas" stroke={color} strokeWidth={2.75} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function SummaryTable({ st }: { st: ExperimentStatus }) {
  const s = st.summary!;
  const v = s.verdict;
  return (
    <div className="card p-5">
      <h3 className="mb-3 text-sm font-semibold text-slate-100">Summary</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="label text-left">
            <th className="pb-2 font-semibold">Controller</th>
            <th className="pb-2 text-right font-semibold">Avg pods</th>
            <th className="pb-2 text-right font-semibold">Max pods</th>
            <th className="pb-2 text-right font-semibold">p95 latency</th>
          </tr>
        </thead>
        <tbody className="tabular-nums">
          <tr className="border-t border-white/[0.06]">
            <td className="py-2.5 font-medium text-eco-light">RL agent</td>
            <td className="py-2.5 text-right text-slate-200">{s.rl.avg_pods}</td>
            <td className="py-2.5 text-right text-slate-200">{s.rl.max_pods}</td>
            <td className="py-2.5 text-right text-slate-200">{s.rl.p95_ms} ms</td>
          </tr>
          <tr className="border-t border-white/[0.06]">
            <td className="py-2.5 font-medium text-eco-red">real HPA</td>
            <td className="py-2.5 text-right text-slate-200">{s.hpa.avg_pods}</td>
            <td className="py-2.5 text-right text-slate-200">{s.hpa.max_pods}</td>
            <td className="py-2.5 text-right text-slate-200">{s.hpa.p95_ms} ms</td>
          </tr>
        </tbody>
      </table>
      {v && (
        <p className="mt-3 rounded-xl bg-white/[0.04] px-3 py-2.5 text-xs leading-relaxed text-slate-400 ring-1 ring-white/5">
          {v.rl_leaner ? (
            <>The agent used <span className="font-semibold text-eco-light">{Math.abs(v.pod_saving_pct)}% fewer pods</span> than the real HPA over the same traffic.</>
          ) : (
            <>The agent used <span className="font-semibold text-eco-amber">{Math.abs(v.pod_saving_pct)}% more pods</span> than the real HPA over the same traffic.</>
          )}{" "}
          Single run on a single-node cluster — a feasibility comparison, not a statistical test.
        </p>
      )}
    </div>
  );
}

function Legend2({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-slate-400">
      <span
        className="inline-block h-0.5 w-4 rounded-full"
        style={ dashed
          ? { backgroundImage: `repeating-linear-gradient(90deg, ${color} 0 4px, transparent 4px 7px)` }
          : { background: color }
        }
      />
      {label}
    </span>
  );
}
