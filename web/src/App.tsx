import { useEffect, useState } from "react";
import { Activity, Power } from "lucide-react";
import { cn } from "@/lib/utils";
import { Sidebar, type Section } from "@/components/eco/sidebar";
import DashboardsSection, { type Mode } from "@/sections/Dashboards";
import DecisionsSection from "@/sections/Decisions";
import ResultsSection from "@/sections/Results";
import ModelSection from "@/sections/Model";
import { getConfig, liveAvailable, liveInfo, getContexts, useKubeContext } from "@/api";
import type { Config } from "@/types";

const TITLE: Record<Section, { h: string; sub: string }> = {
  dashboards: { h: "Dashboards", sub: "Agent vs HPA · live cluster · benchmark experiments" },
  decisions: { h: "Decision log", sub: "Every scaling decision the PPO agent makes, with its reasoning" },
  results: { h: "Results", sub: "Algorithm sweep and the real-cluster head-to-head against native HPA" },
  model: { h: "Model", sub: "The deployed champion: reward design, training, and environment" },
};

export default function App() {
  const [section, setSection] = useState<Section>("dashboards");
  const [config, setConfig] = useState<Config | null>(null);
  const [liveOk, setLiveOk] = useState(false);
  const [mode, setMode] = useState<Mode>("sim");
  const [autopilot, setAutopilot] = useState(false);
  const [killed, setKilled] = useState(false);
  // real cluster binding: $cluster lists actual kubectl contexts; namespace/deployment from /live/info
  const [vars, setVars] = useState([
    { key: "cluster", value: "—", options: ["—"] },
    { key: "namespace", value: "default", options: ["default"] },
    { key: "deployment", value: "eco-sample-app", options: ["eco-sample-app"] },
  ]);

  // refresh cluster-derived bindings (after mount or a context switch)
  const refreshCluster = (contexts?: string[], current?: string) => {
    liveAvailable().then((r) => {
      setLiveOk(r.available);
      liveInfo()
        .then((info) =>
          setVars((v) => [
            { key: "cluster", value: current ?? info.context, options: contexts ?? v[0].options },
            { key: "namespace", value: info.namespace, options: [info.namespace] },
            { key: "deployment", value: info.deployment, options: [info.deployment] },
          ]),
        )
        .catch(() => {
          // selected context has no reachable deployment — keep the cluster choice, blank the rest
          setVars((v) => [
            { key: "cluster", value: current ?? v[0].value, options: contexts ?? v[0].options },
            { key: "namespace", value: "—", options: ["—"] },
            { key: "deployment", value: "—", options: ["—"] },
          ]);
        });
    }).catch(() => { });
  };

  useEffect(() => {
    getConfig().then(setConfig).catch(() => { });
    getContexts()
      .then((c) => {
        if (c.contexts.length) refreshCluster(c.contexts, c.current);
        else refreshCluster();
      })
      .catch(() => refreshCluster());
  }, []);

  const onVarChange = (key: string, value: string) => {
    setVars((v) => v.map((x) => (x.key === key ? { ...x, value } : x)));
    if (key === "cluster") {
      useKubeContext(value)
        .then(() => getContexts())
        .then((c) => refreshCluster(c.contexts, c.current))
        .catch(() => { });
    }
  };

  const liveControls = section === "dashboards" && mode === "live";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex">
        <Sidebar section={section} setSection={setSection} agent={config?.agent} run={config?.run} />
        <div className="flex min-w-0 flex-1 flex-col">
          {/* agent control bar — only in live mode; the kill switch stays pinned while operating */}
          {liveControls && (
            <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur-xl">
              <div className="mx-auto flex w-full max-w-[1400px] items-center gap-3 px-6 py-3">
                <span className="mr-auto inline-flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-agent" /> Agent control
                </span>
                <div className="flex items-center gap-1 rounded-lg border border-border bg-card p-1">
                  <button onClick={() => setAutopilot(false)}
                    className={cn("rounded-md px-2.5 py-1 text-xs font-medium transition",
                      !autopilot ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground")}>
                    Recommend-only
                  </button>
                  <button onClick={() => setAutopilot(true)}
                    className={cn("rounded-md px-2.5 py-1 text-xs font-medium transition",
                      autopilot ? "bg-agent text-agent-foreground" : "text-muted-foreground hover:text-foreground")}>
                    Autopilot
                  </button>
                </div>
                <button onClick={() => setKilled((k) => !k)}
                  className={cn("inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-semibold transition",
                    killed ? "border-destructive/60 bg-destructive/25 text-destructive"
                      : "border-destructive/40 bg-destructive/15 text-destructive hover:bg-destructive/25")}>
                  <Power className="h-3.5 w-3.5" /> {killed ? "Paused (HPA)" : "Kill switch"}
                </button>
              </div>
            </header>
          )}

          <main className="mx-auto w-full max-w-[1400px] flex-1 px-6 py-6">

            {section === "dashboards" && (
              <DashboardsSection
                mode={mode}
                setMode={setMode}
                liveOk={liveOk}
                apply={autopilot && !killed}
                config={config}
                vars={vars}
                onVarChange={onVarChange}
              />
            )}
            {section === "decisions" && <DecisionsSection />}
            {section === "results" && <ResultsSection />}
            {section === "model" && <ModelSection />}

            <footer className="mt-10 flex items-center justify-between border-t border-border pt-4 text-xs text-muted-foreground">
              <span>
                Eco-Scale · {config?.agent ?? "PPO"} agent
                {config && ` · run ${config.run} · min ${config.min_pods} / max ${config.max_pods} pods`}
              </span>
              <span>v0.7 — console</span>
            </footer>
          </main>
        </div>
      </div>
    </div>
  );
}
