import { useState } from "react";
import { Sidebar, type Section } from "@/components/eco/sidebar";
import { AgentControlHeader } from "@/components/eco/header";
import { Footer } from "@/components/eco/footer";
import DashboardsSection, { type Mode } from "@/sections/Dashboards";
import DecisionsSection from "@/sections/Decisions";
import ResultsSection from "@/sections/Results";
import ModelSection from "@/sections/Model";
import { useClusterConfig } from "@/hooks/useClusterConfig";

export default function App() {
  const [section, setSection] = useState<Section>("dashboards");
  const [mode, setMode] = useState<Mode>("sim");
  const [autopilot, setAutopilot] = useState(false);
  const [killed, setKilled] = useState(false);

  const { config, liveOk, vars, onVarChange } = useClusterConfig();

  const liveControls = section === "dashboards" && mode === "live";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex">
        <Sidebar section={section} setSection={setSection} agent={config?.agent} run={config?.run} />
        <div className="flex min-w-0 flex-1 flex-col">
          {/* Agent control bar header — active only during Live Cluster mode */}
          {liveControls && (
            <AgentControlHeader
              autopilot={autopilot}
              setAutopilot={setAutopilot}
              killed={killed}
              setKilled={setKilled}
            />
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

            <Footer config={config} />
          </main>
        </div>
      </div>
    </div>
  );
}
