import { Pause, Play, RotateCcw } from "lucide-react";

interface Props {
  source: "sim" | "live";
  running: boolean;
  speed: number;
  hpaTarget: number;
  tick: number;
  maxTicks: number;
  onToggle: () => void;
  onReset: () => void;
  onSpeed: (ms: number) => void;
  onHpaTarget: (t: number) => void;
}

const SPEEDS = [
  { label: "1×", ms: 500 },
  { label: "2×", ms: 250 },
  { label: "4×", ms: 100 },
];

export default function Controls({
  source, running, speed, hpaTarget, tick, maxTicks,
  onToggle, onReset, onSpeed, onHpaTarget,
}: Props) {
  const isLive = source === "live";
  const hours = ((tick / maxTicks) * 24).toFixed(1);
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3">
      <button
        onClick={onToggle}
        className="flex items-center gap-1.5 rounded-lg bg-eco-green px-4 py-2 text-sm font-medium text-white hover:bg-eco-light"
      >
        {running ? <Pause size={16} /> : <Play size={16} />}
        {running ? "Pause" : "Play"}
      </button>
      <button
        onClick={onReset}
        className="flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800"
      >
        <RotateCcw size={15} /> Reset
      </button>

      {/* speed is a simulation-only concept (live is paced by kubectl) */}
      {!isLive && (
        <div className="flex items-center gap-1 text-xs">
          {SPEEDS.map((s) => (
            <button
              key={s.ms}
              onClick={() => onSpeed(s.ms)}
              className={`rounded-md px-2.5 py-1.5 ${
                speed === s.ms ? "bg-slate-700 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      )}

      {!isLive && (
        <div className="ml-auto flex items-center gap-2 text-xs text-slate-400">
          <span>Compare vs HPA target</span>
          <select
            value={hpaTarget}
            onChange={(e) => onHpaTarget(Number(e.target.value))}
            className="rounded-md border border-slate-700 bg-slate-800 px-2 py-1.5 text-slate-200"
          >
            <option value={0.5}>50% (conservative default)</option>
            <option value={0.6}>60%</option>
            <option value={0.7}>70% (tuned)</option>
            <option value={0.9}>90% (aggressive)</option>
          </select>
        </div>
      )}

      {isLive && (
        <div className="ml-auto rounded-md bg-slate-800 px-3 py-1.5 text-xs text-slate-400">
          Agent is the only autoscaler
        </div>
      )}

      {!isLive && (
        <div className="rounded-md bg-slate-800 px-3 py-1.5 text-xs tabular-nums text-slate-300">
          day {hours}h / 24h
        </div>
      )}
    </div>
  );
}
