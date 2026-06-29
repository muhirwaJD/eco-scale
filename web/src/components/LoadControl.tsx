import { Activity, Play, Square, Waves } from "lucide-react";
import type { LoadStatus } from "../types";

interface Props {
  status: LoadStatus | null;
  onStart: () => void;
  onStop: () => void;
}

export default function LoadControl({ status, onStart, onStop }: Props) {
  const running = status?.running ?? false;
  const intensity = status?.intensity ?? 0;
  const p95 = status?.p95_ms ?? 0;
  const pct = Math.round(intensity * 100);

  return (
    <div className="card p-5">
      <div className="mb-1 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Waves size={15} className="text-eco-light" />
          <h2 className="text-sm font-semibold text-slate-100">Traffic generator</h2>
        </div>
        <span className="label">HTTP wave · /work</span>
      </div>

      <p className="mb-4 text-xs leading-relaxed text-slate-400">
        Sends a real request wave to the app so the agent has demand to react to.
        Start this <span className="font-medium text-slate-200">before</span> pressing Play.
      </p>

      <button
        onClick={running ? onStop : onStart}
        className={`mb-4 flex w-full items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-sm font-semibold transition-all active:scale-[0.99] ${
          running
            ? "bg-eco-red/15 text-eco-red ring-1 ring-eco-red/30 hover:bg-eco-red/25"
            : "bg-eco-green text-[#06210f] hover:bg-eco-light"
        }`}
      >
        {running ? <Square size={14} /> : <Play size={14} />}
        {running ? "Stop traffic" : "Start traffic wave"}
      </button>

      {/* live intensity — gradient fill so the wave reads at a glance */}
      <div className="mb-1.5 flex items-center justify-between">
        <span className="label">Intensity</span>
        <span className="text-xs font-medium tabular-nums text-slate-300">{pct}%</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-eco-green via-eco-amber to-eco-red transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* measured p95 + SLO, side by side */}
      <div className="mt-4 grid grid-cols-2 gap-2">
        <div className="rounded-xl bg-white/[0.04] px-3 py-2 ring-1 ring-white/5">
          <div className="label flex items-center gap-1">
            <Activity size={11} /> Measured p95
          </div>
          <div className={`mt-0.5 text-base font-semibold tabular-nums ${
            p95 && p95 > 1000 ? "text-eco-amber" : "text-eco-light"
          }`}>
            {p95 ? `${p95.toFixed(0)} ms` : "—"}
          </div>
        </div>
        <div className="rounded-xl bg-white/[0.04] px-3 py-2 ring-1 ring-white/5">
          <div className="label">SLO</div>
          <div className="mt-0.5 text-base font-semibold tabular-nums text-slate-300">≤ 1000 ms</div>
        </div>
      </div>

      {running && status && (
        <div className="mt-2 text-right text-[10px] tabular-nums text-slate-500">
          {status.elapsed.toFixed(0)}s / {status.duration}s
        </div>
      )}
    </div>
  );
}
