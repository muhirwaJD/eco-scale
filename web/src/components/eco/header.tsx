import { Power } from "lucide-react";
import { cn } from "@/lib/utils";

interface AgentControlHeaderProps {
  autopilot: boolean;
  setAutopilot: (val: boolean) => void;
  killed: boolean;
  setKilled: React.Dispatch<React.SetStateAction<boolean>>;
}

export function AgentControlHeader({
  autopilot,
  setAutopilot,
  killed,
  setKilled,
}: AgentControlHeaderProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-[1400px] items-center gap-3 px-6 py-3">
        <span className="mr-auto inline-flex items-center gap-2 text-xs font-medium text-muted-foreground">
          <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-agent" /> Agent control
        </span>
        <div className="flex items-center gap-1 rounded-lg border border-border bg-card p-1">
          <button
            onClick={() => setAutopilot(false)}
            className={cn(
              "rounded-md px-2.5 py-1 text-xs font-medium transition",
              !autopilot ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground"
            )}
          >
            Recommend-only
          </button>
          <button
            onClick={() => setAutopilot(true)}
            className={cn(
              "rounded-md px-2.5 py-1 text-xs font-medium transition",
              autopilot ? "bg-agent text-agent-foreground" : "text-muted-foreground hover:text-foreground"
            )}
          >
            Autopilot
          </button>
        </div>
        <button
          onClick={() => setKilled((k) => !k)}
          className={cn(
            "inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-semibold transition",
            killed
              ? "border-destructive/60 bg-destructive/25 text-destructive"
              : "border-destructive/40 bg-destructive/15 text-destructive hover:bg-destructive/25"
          )}
        >
          <Power className="h-3.5 w-3.5" /> {killed ? "Paused (HPA)" : "Kill switch"}
        </button>
      </div>
    </header>
  );
}
