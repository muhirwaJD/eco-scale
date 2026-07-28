import type { Config } from "@/types";

interface FooterProps {
  config: Config | null;
}

export function Footer({ config }: FooterProps) {
  return (
    <footer className="mt-10 flex items-center justify-between border-t border-border pt-4 text-xs text-muted-foreground">
      <span>
        Eco-Scale · {config?.agent ?? "PPO"} agent
        {config && ` · run ${config.run} · min ${config.min_pods} / max ${config.max_pods} pods`}
      </span>
      <span>v0.7 — console</span>
    </footer>
  );
}
