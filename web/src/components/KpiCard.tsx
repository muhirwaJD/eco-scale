import type { ReactNode } from "react";
import Sparkline from "./Sparkline";

interface Props {
  label: string;
  value: string;
  sub?: string;
  icon: ReactNode;
  tone?: "green" | "amber" | "slate";
  spark?: number[];
}

const tones: Record<string, string> = {
  green: "text-eco-light bg-eco-green/10 ring-1 ring-eco-green/25",
  amber: "text-eco-amber bg-eco-amber/10 ring-1 ring-eco-amber/25",
  slate: "text-slate-300 bg-white/5 ring-1 ring-white/10",
};

const sparkColor: Record<string, string> = {
  green: "#4ade80",
  amber: "#fbbf24",
  slate: "#94a3b8",
};

export default function KpiCard({ label, value, sub, icon, tone = "slate", spark }: Props) {
  return (
    <div className="card flex flex-col p-5 transition-all duration-200 hover:-translate-y-0.5 hover:border-white/15">
      <div className="flex items-start justify-between">
        <span className="label">{label}</span>
        <span className={`grid h-9 w-9 place-items-center rounded-xl ${tones[tone]}`}>
          {icon}
        </span>
      </div>
      <div className="mt-3 text-3xl font-semibold tracking-tight tabular-nums text-white">
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
      {spark && spark.length > 1 && (
        <div className="mt-3">
          <Sparkline data={spark} color={sparkColor[tone]} height={28} />
        </div>
      )}
    </div>
  );
}
