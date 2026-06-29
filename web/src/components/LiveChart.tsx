import { useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ChartPoint {
  hour: number;
  load: number; // %
  rlPods: number;
  hpaPods?: number;
  action?: string; // agent action at this tick (for annotations)
  [k: string]: number | string | undefined;
}

const WINDOWS = [
  { label: "6h", h: 6 },
  { label: "12h", h: 12 },
  { label: "24h", h: 24 },
];

export default function LiveChart({
  data,
  maxPods,
  showHpa = true,
}: {
  data: ChartPoint[];
  maxPods: number;
  showHpa?: boolean;
}) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [win, setWin] = useState(24);

  const toggle = (k: string) =>
    setHidden((s) => {
      const n = new Set(s);
      n.has(k) ? n.delete(k) : n.add(k);
      return n;
    });

  // time-window filter (Grafana-style)
  const maxHour = data.length ? data[data.length - 1].hour : 24;
  const view = win >= 24 ? data : data.filter((d) => d.hour >= maxHour - win);

  // annotation markers — small triangles where the agent scaled up/down
  const renderDot = (props: any) => {
    const { cx, cy, payload, index } = props;
    if (cx == null || cy == null) return <g key={index} />;
    if (payload.action === "scale_up")
      return <polygon key={index} points={`${cx},${cy - 7} ${cx - 4},${cy - 1} ${cx + 4},${cy - 1}`} fill="#4ade80" />;
    if (payload.action === "scale_down")
      return <polygon key={index} points={`${cx},${cy + 7} ${cx - 4},${cy + 1} ${cx + 4},${cy + 1}`} fill="#38bdf8" />;
    return <g key={index} />;
  };

  // stat strip values (over the visible window)
  const pods = view.map((d) => d.rlPods);
  const loads = view.map((d) => d.load);
  const stat = {
    podMin: pods.length ? Math.min(...pods) : 0,
    podMax: pods.length ? Math.max(...pods) : 0,
    podAvg: pods.length ? pods.reduce((a, b) => a + b, 0) / pods.length : 0,
    podNow: pods.length ? pods[pods.length - 1] : 0,
    loadAvg: loads.length ? loads.reduce((a, b) => a + b, 0) / loads.length : 0,
    loadPeak: loads.length ? Math.max(...loads) : 0,
  };

  return (
    <div className="card p-5">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-100">
            {showHpa ? "Live replicas — RL agent vs HPA" : "RL agent on the cluster"}
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Pods (left) · Load % (right) · ▲ scale-up · ▼ scale-down
          </p>
        </div>
        {/* interactive legend + time window */}
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-3 text-xs">
            <LegendToggle color="#64748b" label="Load %" k="load" hidden={hidden} onToggle={toggle} />
            <LegendToggle color="#4ade80" label="RL agent" k="rl" hidden={hidden} onToggle={toggle} />
            {showHpa && (
              <LegendToggle color="#f87171" label="HPA" k="hpa" dashed hidden={hidden} onToggle={toggle} />
            )}
          </div>
          <div className="flex items-center gap-0.5 rounded-lg border border-white/10 bg-white/5 p-0.5 text-[11px]">
            {WINDOWS.map((w) => (
              <button
                key={w.h}
                onClick={() => setWin(w.h)}
                className={`rounded-md px-2 py-0.5 font-medium transition-all ${
                  win === w.h ? "bg-white/10 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                {w.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={view} margin={{ top: 8, right: 8, bottom: 4, left: -10 }}>
          <defs>
            <linearGradient id="loadFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#64748b" stopOpacity={0.28} />
              <stop offset="100%" stopColor="#64748b" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey="hour"
            tickFormatter={(h) => `${h}h`}
            stroke="#475569"
            tick={{ fill: "#64748b" }}
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
            type="number"
            domain={["dataMin", "dataMax"]}
          />
          <YAxis
            yAxisId="pods"
            stroke="#475569"
            tick={{ fill: "#64748b" }}
            fontSize={11}
            tickLine={false}
            axisLine={false}
            domain={[0, maxPods]}
          />
          <YAxis
            yAxisId="load"
            orientation="right"
            stroke="#475569"
            tick={{ fill: "#64748b" }}
            fontSize={11}
            tickLine={false}
            axisLine={false}
            domain={[0, 100]}
            unit="%"
          />
          <Tooltip
            contentStyle={{
              background: "rgba(13,18,26,0.95)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 12,
              fontSize: 12,
              boxShadow: "0 12px 30px -12px rgba(0,0,0,0.8)",
            }}
            labelStyle={{ color: "#94a3b8" }}
            labelFormatter={(h) => `Hour ${h}`}
          />
          {/* capacity threshold (Grafana-style) */}
          <ReferenceLine
            yAxisId="pods"
            y={maxPods}
            stroke="rgba(248,113,113,0.35)"
            strokeDasharray="4 4"
            label={{ value: `capacity ${maxPods}`, position: "insideTopRight", fill: "#94a3b8", fontSize: 10 }}
          />
          <Area
            yAxisId="load"
            type="monotone"
            dataKey="load"
            name="Load %"
            stroke="#64748b"
            strokeWidth={1}
            fill="url(#loadFill)"
            isAnimationActive={false}
            hide={hidden.has("load")}
          />
          {showHpa && (
            <Line
              yAxisId="pods"
              type="stepAfter"
              dataKey="hpaPods"
              name="HPA"
              stroke="#f87171"
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
              hide={hidden.has("hpa")}
            />
          )}
          <Line
            yAxisId="pods"
            type="stepAfter"
            dataKey="rlPods"
            name="RL agent"
            stroke="#4ade80"
            strokeWidth={2.75}
            dot={renderDot}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
            hide={hidden.has("rl")}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* stat strip — Grafana legend-table values */}
      <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-6">
        <StatCell label="pods now" value={`${stat.podNow}`} />
        <StatCell label="pods avg" value={stat.podAvg.toFixed(1)} />
        <StatCell label="pods min" value={`${stat.podMin}`} />
        <StatCell label="pods max" value={`${stat.podMax}`} />
        <StatCell label="load avg" value={`${Math.round(stat.loadAvg)}%`} />
        <StatCell label="load peak" value={`${stat.loadPeak}%`} />
      </div>
    </div>
  );
}

function LegendToggle({
  color, label, k, dashed, hidden, onToggle,
}: { color: string; label: string; k: string; dashed?: boolean; hidden: Set<string>; onToggle: (k: string) => void }) {
  const off = hidden.has(k);
  return (
    <button
      onClick={() => onToggle(k)}
      className={`flex items-center gap-1.5 transition-opacity ${off ? "opacity-35" : "opacity-100"}`}
      title="Toggle series"
    >
      <span
        className="inline-block h-0.5 w-4 rounded-full"
        style={
          dashed
            ? { backgroundImage: `repeating-linear-gradient(90deg, ${color} 0 4px, transparent 4px 7px)` }
            : { background: color }
        }
      />
      <span className={`text-slate-400 ${off ? "line-through" : ""}`}>{label}</span>
    </button>
  );
}

function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-white/[0.03] px-2.5 py-1.5 text-center ring-1 ring-white/5">
      <div className="label !text-[9px]">{label}</div>
      <div className="mt-0.5 text-sm font-semibold tabular-nums text-slate-200">{value}</div>
    </div>
  );
}
