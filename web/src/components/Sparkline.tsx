// Tiny inline-SVG trend line for the KPI stat cards (no chart lib needed).
export default function Sparkline({
  data,
  color = "#4ade80",
  height = 30,
}: {
  data: number[];
  color?: string;
  height?: number;
}) {
  const series = (data ?? []).filter((n) => Number.isFinite(n));
  if (series.length < 2) return <div style={{ height }} />;

  const w = 100;
  const h = height;
  const min = Math.min(...series);
  const max = Math.max(...series);
  const range = max - min || 1;
  const y = (v: number) => h - ((v - min) / range) * (h - 6) - 3;
  const x = (i: number) => (i / (series.length - 1)) * w;

  const line = series.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `0,${h} ${line} ${w},${h}`;
  const id = `sg-${color.replace("#", "")}`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="w-full" style={{ height }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <polygon points={area} fill={`url(#${id})`} />
      <polyline
        points={line}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
