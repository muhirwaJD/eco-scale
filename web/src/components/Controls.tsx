import { Pause, Play, RotateCcw, Clock } from "lucide-react";

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
    <div className="card flex flex-wrap items-center gap-3 px-4 py-3">
      <button
        onClick={onToggle}
        className="flex items-center gap-2 rounded-xl bg-eco-green px-5 py-2 text-sm font-semibold text-[#06210f] shadow-sm transition-all hover:bg-eco-light active:scale-95"
      >
        {running ? <Pause size={16} /> : <Play size={16} />}
        {running ? "Pause" : "Play"}
      </button>
      <button
        onClick={onReset}
        className="flex items-center gap-1.5 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-300 transition-colors hover:bg-white/5"
      >
        <RotateCcw size={15} /> Reset
      </button>

      {/* speed is a simulation-only concept (live is paced by kubectl) */}
      {!isLive && (
        <div className="flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 p-1 text-sm">
          {SPEEDS.map((s) => (
            <button
              key={s.ms}
              onClick={() => onSpeed(s.ms)}
              className={`rounded-lg px-2.5 py-1.5 font-medium transition-all ${
                speed === s.ms ? "bg-white/10 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      )}

      {!isLive && (
        <div className="ml-auto flex items-center gap-2 text-sm text-slate-400">
          <span>Compare vs HPA target</span>
          <select
            value={hpaTarget}
            onChange={(e) => onHpaTarget(Number(e.target.value))}
            className="rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-slate-200 outline-none transition-colors hover:border-white/20 focus:border-eco-green/50"
          >
            <option value={0.5}>50% (conservative default)</option>
            <option value={0.6}>60%</option>
            <option value={0.7}>70% (tuned)</option>
            <option value={0.9}>90% (aggressive)</option>
          </select>
        </div>
      )}

      {isLive && (
        <div className="ml-auto rounded-lg bg-white/5 px-3 py-1.5 text-sm text-slate-400 ring-1 ring-white/10">
          Agent is the only autoscaler
        </div>
      )}

      {!isLive && (
        <div className="flex items-center gap-1.5 rounded-lg bg-white/5 px-3 py-1.5 text-sm tabular-nums text-slate-300 ring-1 ring-white/10">
          <Clock size={13} className="text-slate-500" />
          day {hours}h / 24h
        </div>
      )}
    </div>
  );
}
