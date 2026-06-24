import { useEffect, useRef, useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Legend, Line,
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

  // live chart of the phase currently running
  const liveData = (st && running && st.phase ? st[st.phase] : []).map((p) => ({
    t: p.t, replicas: p.replicas, load: Math.round(p.intensity * 100),
  }));

  // comparison data (align rl & hpa by index)
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
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
        <div className="mb-2 flex items-center gap-2">
          <FlaskConical size={16} className="text-eco-light" />
          <h2 className="text-sm font-semibold text-slate-200">
            Real A/B experiment — RL agent vs native Kubernetes HPA
          </h2>
        </div>
        <p className="mb-3 text-xs text-slate-400">
          Runs the genuine Stage-2 head-to-head on your cluster: the agent drives the
          deployment under a traffic wave, then the <span className="text-slate-300">real
          native HPA</span> drives it under the same wave. Everything is real — real pods,
          real CPU, real HPA, real latency. ~{Math.round((duration * 2) / 60) + 1} min total.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          {!running ? (
            <button
              onClick={start}
              className="flex items-center gap-1.5 rounded-lg bg-eco-green px-4 py-2 text-sm font-medium text-white hover:bg-eco-light"
            >
              <FlaskConical size={15} /> Run experiment
            </button>
          ) : (
            <button
              onClick={stop}
              className="flex items-center gap-1.5 rounded-lg bg-eco-red/20 px-4 py-2 text-sm font-medium text-eco-red hover:bg-eco-red/30"
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
              className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-200 disabled:opacity-40"
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
        <div className="rounded-xl border border-eco-green/30 bg-slate-900/50 p-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-medium text-eco-light">
              {st.phase ? PHASE_LABEL[st.phase] : "Starting…"}
            </span>
            <span className="text-xs tabular-nums text-slate-400">
              {st.elapsed.toFixed(0)}s / {st.duration}s
            </span>
          </div>
          <div className="mb-3 h-1.5 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-eco-light transition-all"
              style={{ width: `${Math.min(100, (st.elapsed / st.duration) * 100)}%` }}
            />
          </div>
          <PhaseChart data={liveData} color={st.phase === "hpa" ? "#ef4444" : "#4CAF50"} />
        </div>
      )}

      {/* done: comparison + summary */}
      {done && st?.summary && (
        <>
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
            <h3 className="mb-2 text-sm font-semibold text-slate-200">
              Result — replicas over the same traffic wave
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={cmpData} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="t" stroke="#64748b" fontSize={11}
                  tickFormatter={(t) => `${Math.round(t)}s`} />
                <YAxis yAxisId="pods" stroke="#64748b" fontSize={11}
                  label={{ value: "pods", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }} />
                <YAxis yAxisId="load" orientation="right" stroke="#64748b" fontSize={11} domain={[0, 100]} unit="%" />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area yAxisId="load" type="monotone" dataKey="load" name="Load %"
                  stroke="#334155" fill="#1e293b" fillOpacity={0.7} isAnimationActive={false} />
                <Line yAxisId="pods" type="stepAfter" dataKey="hpaPods" name="real HPA"
                  stroke="#ef4444" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls />
                <Line yAxisId="pods" type="stepAfter" dataKey="rlPods" name="RL agent"
                  stroke="#4CAF50" strokeWidth={2.5} dot={false} isAnimationActive={false} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
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
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis dataKey="t" stroke="#64748b" fontSize={11} tickFormatter={(t) => `${Math.round(t)}s`} />
        <YAxis yAxisId="pods" stroke="#64748b" fontSize={11} />
        <YAxis yAxisId="load" orientation="right" stroke="#64748b" fontSize={11} domain={[0, 100]} unit="%" />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }} />
        <Area yAxisId="load" type="monotone" dataKey="load" name="Load %" stroke="#334155" fill="#1e293b" fillOpacity={0.7} isAnimationActive={false} />
        <Line yAxisId="pods" type="stepAfter" dataKey="replicas" name="replicas" stroke={color} strokeWidth={2.5} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function SummaryTable({ st }: { st: ExperimentStatus }) {
  const s = st.summary!;
  const v = s.verdict;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <h3 className="mb-3 text-sm font-semibold text-slate-200">Summary</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[10px] uppercase tracking-wide text-slate-500">
            <th className="pb-2">Controller</th>
            <th className="pb-2 text-right">Avg pods</th>
            <th className="pb-2 text-right">Max pods</th>
            <th className="pb-2 text-right">p95 latency</th>
          </tr>
        </thead>
        <tbody className="tabular-nums">
          <tr className="border-t border-slate-800">
            <td className="py-2 text-eco-light">RL agent</td>
            <td className="py-2 text-right">{s.rl.avg_pods}</td>
            <td className="py-2 text-right">{s.rl.max_pods}</td>
            <td className="py-2 text-right">{s.rl.p95_ms} ms</td>
          </tr>
          <tr className="border-t border-slate-800">
            <td className="py-2 text-eco-red">real HPA</td>
            <td className="py-2 text-right">{s.hpa.avg_pods}</td>
            <td className="py-2 text-right">{s.hpa.max_pods}</td>
            <td className="py-2 text-right">{s.hpa.p95_ms} ms</td>
          </tr>
        </tbody>
      </table>
      {v && (
        <p className="mt-3 text-xs text-slate-400">
          {v.rl_leaner ? (
            <>The agent used <span className="text-eco-light">{Math.abs(v.pod_saving_pct)}% fewer pods</span> than the real HPA over the same traffic.</>
          ) : (
            <>The agent used <span className="text-eco-amber">{Math.abs(v.pod_saving_pct)}% more pods</span> than the real HPA over the same traffic.</>
          )}{" "}
          Single run on a single-node cluster — a feasibility comparison, not a statistical test.
        </p>
      )}
    </div>
  );
}
