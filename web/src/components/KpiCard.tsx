import type { ReactNode } from "react";

interface Props {
  label: string;
  value: string;
  sub?: string;
  icon: ReactNode;
  tone?: "green" | "amber" | "slate";
}

const tones: Record<string, string> = {
  green: "text-eco-light bg-eco-green/15",
  amber: "text-eco-amber bg-eco-amber/15",
  slate: "text-slate-300 bg-slate-700/40",
};

export default function KpiCard({ label, value, sub, icon, tone = "slate" }: Props) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-slate-400">{label}</span>
        <span className={`grid h-7 w-7 place-items-center rounded-md ${tones[tone]}`}>
          {icon}
        </span>
      </div>
      <div className="mt-2 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}
