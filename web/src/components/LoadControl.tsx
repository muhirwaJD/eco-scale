import { Activity, Waves } from "lucide-react";
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

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <div className="mb-3 flex items-center gap-2">
        <Waves size={15} className="text-eco-light" />
        <h2 className="text-sm font-semibold text-slate-200">Traffic generator</h2>
      </div>

      <p className="mb-3 text-xs text-slate-400">
        Sends a real request wave to the app so the agent has demand to react to.
        Start this <span className="text-slate-300">before</span> pressing Play.
      </p>

      <button
        onClick={running ? onStop : onStart}
        className={`mb-3 w-full rounded-lg px-3 py-2 text-sm font-medium transition ${
          running
            ? "bg-eco-red/20 text-eco-red hover:bg-eco-red/30"
            : "bg-eco-green text-white hover:bg-eco-light"
        }`}
      >
        {running ? "■ Stop traffic" : "▶ Start traffic wave"}
      </button>

      {/* live intensity bar */}
      <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wide text-slate-500">
        <span>Intensity</span>
        <span className="tabular-nums">{Math.round(intensity * 100)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-full bg-eco-light transition-all"
          style={{ width: `${Math.round(intensity * 100)}%` }}
        />
      </div>

      <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
        <span className="flex items-center gap-1">
          <Activity size={12} /> measured p95
        </span>
        <span className="tabular-nums text-slate-300">{p95 ? `${p95.toFixed(0)} ms` : "—"}</span>
      </div>
      {running && status && (
        <div className="mt-1 text-right text-[10px] text-slate-500 tabular-nums">
          {status.elapsed.toFixed(0)}s / {status.duration}s
        </div>
      )}
    </div>
  );
}
