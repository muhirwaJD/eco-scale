import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { Mode, SimState } from "../types";

const ACTION_UI: Record<string, { label: string; icon: JSX.Element; cls: string }> = {
  scale_up: { label: "Scale up", icon: <ArrowUp size={18} />, cls: "text-eco-light bg-eco-green/15" },
  scale_down: { label: "Scale down", icon: <ArrowDown size={18} />, cls: "text-sky-400 bg-sky-500/15" },
  maintain: { label: "Maintain", icon: <Minus size={18} />, cls: "text-slate-300 bg-slate-700/40" },
};

const PROB_LABELS = ["down", "hold", "up"];

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-800/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-sm font-medium tabular-nums text-slate-200">{value}</div>
    </div>
  );
}

export default function DecisionPanel({ s, mode }: { s: SimState | null; mode: Mode }) {
  if (!s) return null;
  const ui = ACTION_UI[s.rl.action_name] ?? ACTION_UI.maintain;
  const obs = s.rl.observation;

  return (
    <div className="flex flex-col gap-4 rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <h2 className="text-sm font-semibold text-slate-200">Agent decision</h2>

      {/* the action */}
      <div className={`flex items-center gap-3 rounded-lg px-4 py-3 ${ui.cls}`}>
        {ui.icon}
        <div>
          <div className="text-base font-semibold">{ui.label}</div>
          <div className="text-xs opacity-80">
            {mode === "autopilot" ? "applied automatically" : "awaiting approval"}
          </div>
        </div>
      </div>

      {/* rationale (explainability) */}
      <p className="text-sm leading-relaxed text-slate-300">{s.rl.rationale}</p>

      {/* action preference bars: why not the others? */}
      {s.rl.probs && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
            Action preference
          </div>
          <div className="space-y-1.5">
            {s.rl.probs.map((p, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-8 text-xs text-slate-400">{PROB_LABELS[i]}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-eco-light"
                    style={{ width: `${Math.round(p * 100)}%` }}
                  />
                </div>
                <span className="w-9 text-right text-xs tabular-nums text-slate-400">
                  {Math.round(p * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* what the agent saw */}
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
          Observation
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="CPU load" value={`${Math.round(obs[0] * 100)}%`} />
          <Stat label="Pods" value={`${s.rl.pods}`} />
          <Stat label="Queue" value={`${Math.round(obs[2] * 100)}%`} />
          <Stat label="Time of day" value={`${(obs[3] * 24).toFixed(1)}h`} />
        </div>
      </div>

      {/* head-to-head current state */}
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
          RL vs HPA (now)
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="RL pods" value={`${s.rl.pods}`} />
          <Stat label={`HPA@${Math.round(s.hpa.target * 100)}% pods`} value={`${s.hpa.pods}`} />
          <Stat label="RL latency" value={`${Math.round(s.rl.latency * 100)}%`} />
          <Stat label="HPA latency" value={`${Math.round(s.hpa.latency * 100)}%`} />
        </div>
      </div>
    </div>
  );
}
