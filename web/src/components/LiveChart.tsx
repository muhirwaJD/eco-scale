import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
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
}

export default function LiveChart({
  data,
  maxPods,
  showHpa = true,
}: {
  data: ChartPoint[];
  maxPods: number;
  showHpa?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200">
          {showHpa ? "Live replicas — RL agent vs HPA" : "Live replicas — RL agent on the cluster"}
        </h2>
        <span className="text-xs text-slate-500">load shaded · replicas as lines</span>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="hour"
            tickFormatter={(h) => `${h}h`}
            stroke="#64748b"
            fontSize={11}
            type="number"
            domain={[0, 24]}
            ticks={[0, 4, 8, 12, 16, 20, 24]}
          />
          <YAxis
            yAxisId="pods"
            stroke="#64748b"
            fontSize={11}
            domain={[0, maxPods]}
            label={{ value: "pods", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }}
          />
          <YAxis
            yAxisId="load"
            orientation="right"
            stroke="#64748b"
            fontSize={11}
            domain={[0, 100]}
            unit="%"
          />
          <Tooltip
            contentStyle={{
              background: "#0f172a",
              border: "1px solid #334155",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelFormatter={(h) => `Hour ${h}`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Area
            yAxisId="load"
            type="monotone"
            dataKey="load"
            name="Load %"
            stroke="#334155"
            fill="#1e293b"
            fillOpacity={0.7}
            isAnimationActive={false}
          />
          {showHpa && (
            <Line
              yAxisId="pods"
              type="stepAfter"
              dataKey="hpaPods"
              name="HPA"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
            />
          )}
          <Line
            yAxisId="pods"
            type="stepAfter"
            dataKey="rlPods"
            name="RL agent"
            stroke="#4CAF50"
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
