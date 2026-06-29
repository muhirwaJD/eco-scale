import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import type { Mode, SimState } from "../types";

const ACTION_UI: Record<string, { label: string; icon: JSX.Element; cls: string }> = {
  scale_up: {
    label: "Scale up",
    icon: <ArrowUp size={18} strokeWidth={2.5} />,
    cls: "text-eco-light bg-eco-green/12 ring-1 ring-eco-green/25",
  },
  scale_down: {
    label: "Scale down",
    icon: <ArrowDown size={18} strokeWidth={2.5} />,
    cls: "text-sky-300 bg-sky-500/12 ring-1 ring-sky-400/25",
  },
  maintain: {
    label: "Maintain",
    icon: <Minus size={18} strokeWidth={2.5} />,
    cls: "text-slate-200 bg-white/5 ring-1 ring-white/10",
  },
};

// down / hold / up — each gets its own colour so the bias reads instantly
const PROB = [
  { label: "down", bar: "bg-eco-red" },
  { label: "hold", bar: "bg-slate-400" },
  { label: "up", bar: "bg-eco-light" },
];

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-white/[0.04] px-3 py-2 ring-1 ring-white/5">
      <div className="label">{label}</div>
      <div className="mt-0.5 text-sm font-semibold tabular-nums text-slate-100">{value}</div>
    </div>
  );
}

export default function DecisionPanel({
  s, mode, live = false,
}: { s: SimState | null; mode: Mode; live?: boolean }) {
  if (!s) return null;
  const ui = ACTION_UI[s.rl.action_name] ?? ACTION_UI.maintain;
  const obs = s.rl.observation;

  return (
    <div className="card flex flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-100">Agent decision</h2>
        <span className="label">PPO · interval 15s</span>
      </div>

      {/* the action */}
      <div className={`flex items-center gap-3 rounded-xl px-4 py-3 ${ui.cls}`}>
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
          <div className="label mb-2">Action preference</div>
          <div className="space-y-2">
            {s.rl.probs.map((p, i) => (
              <div key={i} className="flex items-center gap-2.5">
                <span className="w-9 text-xs font-medium text-slate-400">{PROB[i].label}</span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                  <div
                    className={`h-full rounded-full ${PROB[i].bar} transition-all duration-300`}
                    style={{ width: `${Math.round(p * 100)}%` }}
                  />
                </div>
                <span className="w-9 text-right text-xs font-medium tabular-nums text-slate-300">
                  {Math.round(p * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* what the agent saw */}
      <div>
        <div className="label mb-2">Observation</div>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="CPU load" value={`${Math.round(obs[0] * 100)}%`} />
          <Stat label="Pods" value={`${s.rl.pods}`} />
          <Stat label="Queue" value={`${Math.round(obs[2] * 100)}%`} />
          <Stat label="Time of day" value={`${(obs[3] * 24).toFixed(1)}h`} />
        </div>
      </div>

      {/* current state — live shows the real cluster; sim shows RL vs HPA */}
      {live ? (
        <div>
          <div className="label mb-2">Cluster now</div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Replicas" value={`${s.rl.pods}`} />
            <Stat label="Peak pods" value={`${s.stats?.peak_pods ?? s.rl.pods}`} />
            <Stat label="Avg pod CPU" value={`${Math.round(s.avg_cpu_millicores ?? 0)}m`} />
            <Stat label="Scaling actions" value={`${s.stats?.scaling_actions ?? 0}`} />
          </div>
        </div>
      ) : (
        s.hpa && (
          <div>
            <div className="label mb-2">RL vs HPA (now)</div>
            <div className="grid grid-cols-2 gap-2">
              <Stat label="RL pods" value={`${s.rl.pods}`} />
              <Stat label={`HPA@${Math.round(s.hpa.target * 100)}% pods`} value={`${s.hpa.pods}`} />
              <Stat label="RL latency" value={`${Math.round((s.rl.latency ?? 0) * 100)}%`} />
              <Stat label="HPA latency" value={`${Math.round(s.hpa.latency * 100)}%`} />
            </div>
          </div>
        )
      )}
    </div>
  );
}
