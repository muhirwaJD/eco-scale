import { ArrowDown, ArrowUp, Minus, ScrollText } from "lucide-react";

export interface EventItem {
  hour: number;
  action: string;
  rationale: string;
}

const ICON: Record<string, { node: JSX.Element; cls: string }> = {
  scale_up: { node: <ArrowUp size={13} />, cls: "text-eco-light bg-eco-green/10 ring-eco-green/25" },
  scale_down: { node: <ArrowDown size={13} />, cls: "text-sky-300 bg-sky-500/10 ring-sky-400/25" },
  maintain: { node: <Minus size={13} />, cls: "text-slate-400 bg-white/5 ring-white/10" },
};

export default function EventLog({ events }: { events: EventItem[] }) {
  const rows = [...events].reverse(); // newest first

  return (
    <div className="card flex h-full flex-col p-5">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ScrollText size={15} className="text-eco-light" />
          <h2 className="text-sm font-semibold text-slate-100">Decision log</h2>
        </div>
        <span className="label">{events.length} events</span>
      </div>

      {rows.length === 0 ? (
        <div className="grid flex-1 place-items-center py-6 text-xs text-slate-500">
          Press Play — decisions stream here.
        </div>
      ) : (
        <div className="max-h-[260px] space-y-1.5 overflow-auto pr-1">
          {rows.map((e, i) => {
            const ui = ICON[e.action] ?? ICON.maintain;
            return (
              <div
                key={i}
                className="flex items-start gap-2.5 rounded-lg bg-white/[0.03] px-2.5 py-2 ring-1 ring-white/5"
              >
                <span className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-md ring-1 ${ui.cls}`}>
                  {ui.node}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium capitalize text-slate-200">
                      {e.action.replace("_", " ")}
                    </span>
                    <span className="shrink-0 text-[10px] tabular-nums text-slate-500">
                      {e.hour.toFixed(1)}h
                    </span>
                  </div>
                  <p className="truncate text-[11px] text-slate-500" title={e.rationale}>
                    {e.rationale}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
