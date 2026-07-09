import { LayoutDashboard, ScrollText, BarChart3, Cpu, Leaf } from "lucide-react";
import { cn } from "@/lib/utils";

export type Section = "dashboards" | "decisions" | "results" | "model";

const ITEMS: { id: Section; label: string; icon: React.ReactNode }[] = [
  { id: "dashboards", label: "Dashboards", icon: <LayoutDashboard className="h-4 w-4" /> },
  { id: "decisions", label: "Decision log", icon: <ScrollText className="h-4 w-4" /> },
  { id: "results", label: "Results", icon: <BarChart3 className="h-4 w-4" /> },
  { id: "model", label: "Model", icon: <Cpu className="h-4 w-4" /> },
];

export function Sidebar({
  section,
  setSection,
  agent,
  run,
}: {
  section: Section;
  setSection: (s: Section) => void;
  agent?: string;
  run?: number;
}) {
  return (
    <aside className="sticky top-0 hidden h-screen w-[220px] shrink-0 flex-col border-r border-border bg-card/40 px-3 py-4 lg:flex">
      <div className="mb-6 flex items-center gap-2.5 px-2">
        <div className="relative grid h-8 w-8 place-items-center rounded-lg bg-agent/15 ring-1 ring-agent/30">
          <Leaf className="h-4 w-4 text-agent" />
          <span className="pulse-dot absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-agent" />
        </div>
        <div>
          <div className="text-[13px] font-semibold leading-none">Eco-Scale</div>
          <div className="mt-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            Console
          </div>
        </div>
      </div>

      <nav className="flex flex-col gap-0.5">
        {ITEMS.map((it) => (
          <button
            key={it.id}
            onClick={() => setSection(it.id)}
            className={cn(
              "group flex items-center justify-between rounded-md px-2.5 py-2 text-sm transition",
              section === it.id
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
          >
            <span className="flex items-center gap-2.5">
              <span
                className={cn(
                  section === it.id ? "text-agent" : "text-muted-foreground group-hover:text-foreground",
                )}
              >
                {it.icon}
              </span>
              {it.label}
            </span>
          </button>
        ))}
      </nav>

      <div className="mt-auto rounded-lg border border-border bg-background/40 p-3 text-[11px] leading-snug text-muted-foreground">
        Live cluster + simulation
        <div className="mt-1 text-foreground">
          {agent ?? "PPO"} · run {run ?? 6}
        </div>
      </div>
    </aside>
  );
}
