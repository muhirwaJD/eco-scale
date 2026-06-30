import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

export type Action = "down" | "hold" | "up";

export function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "surface-1 p-4 shadow-[0_1px_0_0_rgba(255,255,255,0.03)_inset]",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Pill({
  tone = "muted",
  children,
}: {
  tone?: "muted" | "agent" | "hpa" | "warn" | "live";
  children: React.ReactNode;
}) {
  const map: Record<string, string> = {
    muted: "bg-muted text-muted-foreground border-border",
    agent: "bg-agent/15 text-agent border-agent/30",
    hpa: "bg-destructive/15 text-destructive border-destructive/30",
    warn: "bg-warn/15 text-warn border-warn/30",
    live: "bg-agent/10 text-agent border-agent/30",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium uppercase tracking-wider",
        map[tone],
      )}
    >
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
  icon,
  tone = "default",
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  icon?: React.ReactNode;
  tone?: "default" | "positive";
}) {
  return (
    <Card className={cn("relative overflow-hidden", tone === "positive" && "bg-agent/5 border-agent/25")}>
      <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
        <span>{label}</span>
        <span className={tone === "positive" ? "text-agent" : "text-muted-foreground"}>{icon}</span>
      </div>
      <div
        className={cn(
          "mt-2 text-3xl font-semibold tabular-nums",
          tone === "positive" ? "text-agent" : "text-foreground",
        )}
      >
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-muted-foreground">{sub}</div>}
    </Card>
  );
}

export function LegendDot({
  color,
  label,
  dashed,
  muted,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  muted?: boolean;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1.5", muted && "text-muted-foreground")}>
      <span
        className="inline-block h-[3px] w-5 rounded-sm"
        style={{
          background: dashed ? undefined : color,
          borderTop: dashed ? `2px dashed ${color}` : undefined,
          height: dashed ? 0 : 3,
        }}
      />
      {label}
    </span>
  );
}

export function MiniStat({
  label,
  val,
  tone,
}: {
  label: string;
  val: React.ReactNode;
  tone?: "agent" | "hpa";
}) {
  const color =
    tone === "agent" ? "text-agent" : tone === "hpa" ? "text-destructive" : "text-foreground";
  return (
    <div className="rounded-md border border-border bg-background/40 px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={cn("text-sm font-semibold tabular-nums", color)}>{val}</div>
    </div>
  );
}

export function ProbBar({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "agent" | "hpa" | "muted";
}) {
  const color =
    tone === "agent" ? "var(--color-agent)" : tone === "hpa" ? "var(--color-hpa)" : "var(--color-muted-foreground)";
  return (
    <div className="flex items-center gap-2">
      <span className="w-10 text-[11px] uppercase tracking-wider text-muted-foreground">{label}</span>
      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${value * 100}%`, background: color }}
        />
      </div>
      <span className="w-10 text-right text-[11px] tabular-nums text-muted-foreground">
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

export function ActionBadge({ action }: { action: Action }) {
  const map: Record<Action, { label: string; icon: React.ReactNode; cls: string }> = {
    up: { label: "Scale up", icon: <ArrowUp className="h-4 w-4" />, cls: "bg-agent/15 text-agent border-agent/40" },
    down: { label: "Scale down", icon: <ArrowDown className="h-4 w-4" />, cls: "bg-destructive/15 text-destructive border-destructive/40" },
    hold: { label: "Maintain", icon: <Minus className="h-4 w-4" />, cls: "bg-muted text-foreground border-border" },
  };
  const a = map[action];
  return (
    <span className={cn("inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-semibold", a.cls)}>
      {a.icon} {a.label}
    </span>
  );
}
