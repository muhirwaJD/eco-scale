import { Activity, Power, ShieldCheck, Bot } from "lucide-react";
import type { Config, Mode } from "../types";

interface Props {
  config: Config | null;
  mode: Mode;
  killed: boolean;
  live: boolean; // scaling controls only matter on a real cluster
  onMode: (m: Mode) => void;
  onKill: () => void;
}

export default function Header({ config, mode, killed, live, onMode, onKill }: Props) {
  return (
    <header className="sticky top-0 z-20 flex items-center justify-between border-b border-white/[0.06] bg-[#0a0e14]/80 px-6 py-4 backdrop-blur-xl">
      <div className="flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-eco-green/15 text-eco-light shadow-glow ring-1 ring-eco-green/30">
          <Activity size={20} strokeWidth={2.4} />
        </div>
        <div>
          <h1 className="text-[17px] font-semibold leading-tight tracking-tight text-white">
            Eco-Scale Console
          </h1>
          <p className="text-xs text-slate-400">
            RL autoscaler · agent{" "}
            <span className="font-medium text-eco-light">{config?.agent ?? "…"}</span>
            {config && ` · run ${config.run}`}
          </p>
        </div>
      </div>

      {live && (
        <div className="flex items-center gap-3">
          {/* mode switch: recommend-only vs autopilot */}
          <div className="flex rounded-xl border border-white/10 bg-white/5 p-1 text-xs">
            <button
              onClick={() => onMode("recommend")}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-medium transition-all ${
                mode === "recommend"
                  ? "bg-white/10 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <ShieldCheck size={14} /> Recommend-only
            </button>
            <button
              onClick={() => onMode("autopilot")}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-medium transition-all ${
                mode === "autopilot"
                  ? "bg-eco-green text-[#06210f] shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Bot size={14} /> Autopilot
            </button>
          </div>

          <button
            onClick={onKill}
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition-all ${
              killed
                ? "border-eco-red/60 bg-eco-red/15 text-eco-red"
                : "border-white/10 text-slate-300 hover:border-eco-red/60 hover:bg-eco-red/10 hover:text-eco-red"
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
