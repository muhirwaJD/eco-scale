import { useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Minus, Pause, Play, ScrollText } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, Pill } from "@/components/eco/primitives";
import { simReset, simStep } from "@/api";
import type { SimState } from "@/types";

interface Ev { hour: number; action: string; rationale: string; pods: number; cpu: number }

const ICON: Record<string, { node: JSX.Element; cls: string }> = {
  scale_up: { node: <ArrowUp className="h-3.5 w-3.5" />, cls: "bg-agent/15 text-agent border-agent/30" },
  scale_down: { node: <ArrowDown className="h-3.5 w-3.5" />, cls: "bg-destructive/15 text-destructive border-destructive/30" },
  maintain: { node: <Minus className="h-3.5 w-3.5" />, cls: "bg-muted text-muted-foreground border-border" },
};

export default function DecisionsSection() {
  const [events, setEvents] = useState<Ev[]>([]);
  const [playing, setPlaying] = useState(false);
  const [counts, setCounts] = useState({ up: 0, hold: 0, down: 0 });
  const timer = useRef<number | null>(null);

  const push = (s: SimState) => {
    const e: Ev = {
      hour: +((s.tick / s.max_ticks) * 24).toFixed(2),
      action: s.rl.action_name, rationale: s.rl.rationale,
      pods: s.rl.pods, cpu: Math.round(s.cpu * 100),
    };
    setEvents((arr) => [...arr, e].slice(-120));
    setCounts((c) => ({
      up: c.up + (e.action === "scale_up" ? 1 : 0),
      hold: c.hold + (e.action === "maintain" ? 1 : 0),
      down: c.down + (e.action === "scale_down" ? 1 : 0),
    }));
  };

  useEffect(() => { simReset(0.5).catch(() => {}); }, []);
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(async () => {
      const s = await simStep();
      push(s);
      if (s.done) setPlaying(false);
    }, 250);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing]);

  const total = counts.up + counts.hold + counts.down || 1;
  const rows = [...events].reverse();

  return (
    <div className="space-y-4">
      <Card className="flex flex-wrap items-center gap-3">
        <button onClick={() => setPlaying((p) => !p)}
          className={cn("inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition",
            playing ? "bg-agent text-agent-foreground hover:opacity-90" : "bg-agent/20 text-agent hover:bg-agent/30")}>
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}{playing ? "Pause" : "Stream decisions"}
        </button>
        <span className="text-xs text-muted-foreground">The PPO agent's decisions on the held-out Alibaba trace, with the reason for each.</span>
        <div className="ml-auto flex items-center gap-2 text-[11px]">
          <Pill tone="agent">{counts.up} up</Pill>
          <Pill tone="muted">{counts.hold} hold</Pill>
          <Pill tone="hpa">{counts.down} down</Pill>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_1.6fr]">
        <Card>
          <h3 className="mb-3 text-sm font-semibold">Action mix</h3>
          {(["up", "hold", "down"] as const).map((k) => (
            <div key={k} className="mb-2.5">
              <div className="mb-1 flex justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
                <span>{k}</span><span className="tabular-nums text-foreground">{Math.round((counts[k] / total) * 100)}%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full transition-[width] duration-300"
                  style={{ width: `${(counts[k] / total) * 100}%`, background: k === "up" ? "var(--color-agent)" : k === "down" ? "var(--color-hpa)" : "var(--color-muted-foreground)" }} />
              </div>
            </div>
          ))}
          <p className="mt-3 text-xs text-muted-foreground">{total - 1 || 0} decisions logged this session.</p>
        </Card>

        <Card className="flex h-[420px] flex-col">
          <div className="mb-3 flex items-center gap-2">
            <ScrollText className="h-4 w-4 text-agent" />
            <h3 className="text-sm font-semibold">Decision log</h3>
          </div>
          {rows.length === 0 ? (
            <div className="grid flex-1 place-items-center text-sm text-muted-foreground">Press “Stream decisions” to begin.</div>
          ) : (
            <div className="space-y-1.5 overflow-auto pr-1">
              {rows.map((e, i) => {
                const ui = ICON[e.action] ?? ICON.maintain;
                return (
                  <div key={i} className="flex items-start gap-2.5 rounded-md border border-border bg-background/40 px-2.5 py-2">
                    <span className={cn("mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-md border", ui.cls)}>{ui.node}</span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium capitalize text-foreground">{e.action.replace("_", " ")}</span>
                        <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground">{e.hour.toFixed(1)}h · {e.pods} pods · {e.cpu}%</span>
                      </div>
                      <p className="truncate text-xs text-muted-foreground" title={e.rationale}>{e.rationale}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
