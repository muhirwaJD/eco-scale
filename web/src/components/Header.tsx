import { Activity, Power, ShieldCheck, Bot } from "lucide-react";
import type { Config, Mode } from "../types";

interface Props {
  config: Config | null;
  mode: Mode;
  killed: boolean;
  live: boolean;            // scaling controls only matter on a real cluster
  onMode: (m: Mode) => void;
  onKill: () => void;
}

export default function Header({ config, mode, killed, live, onMode, onKill }: Props) {
  return (
    <header className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="grid h-9 w-9 place-items-center rounded-lg bg-eco-green/20 text-eco-light">
          <Activity size={20} />
        </div>
        <div>
          <h1 className="text-lg font-semibold leading-tight">Eco-Scale Console</h1>
          <p className="text-xs text-slate-400">
            RL autoscaler · agent{" "}
            <span className="text-eco-light">{config?.agent ?? "…"}</span>
            {config && ` (run ${config.run})`}
          </p>
        </div>
      </div>

      {live && (
      <div className="flex items-center gap-3">
        {/* mode switch: recommend-only vs autopilot */}
        <div className="flex rounded-lg border border-slate-700 p-0.5 text-xs">
          <button
            onClick={() => onMode("recommend")}
            className={`flex items-center gap-1 rounded-md px-3 py-1.5 transition ${
              mode === "recommend" ? "bg-slate-700 text-white" : "text-slate-400"
            }`}
          >
            <ShieldCheck size={14} /> Recommend-only
          </button>
          <button
            onClick={() => onMode("autopilot")}
            className={`flex items-center gap-1 rounded-md px-3 py-1.5 transition ${
              mode === "autopilot" ? "bg-eco-green text-white" : "text-slate-400"
            }`}
          >
            <Bot size={14} /> Autopilot
          </button>
        </div>

        <button
          onClick={onKill}
          className={`flex items-center gap-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${
            killed
              ? "border-eco-red bg-eco-red/20 text-eco-red"
              : "border-slate-700 text-slate-300 hover:border-eco-red hover:text-eco-red"
          }`}
          title="Fall back to native HPA instantly"
        >
          <Power size={14} /> {killed ? "Agent paused (HPA)" : "Kill switch"}
        </button>
      </div>
      )}
    </header>
  );
}
