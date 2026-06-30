import { useEffect, useState } from "react";
import { BarChart3, ShieldCheck, Trophy } from "lucide-react";
import { Card, Stat } from "@/components/eco/primitives";
import { getResults } from "@/api";
import type { Results } from "@/types";

export default function ResultsSection() {
  const [r, setR] = useState<Results | null>(null);
  useEffect(() => { getResults().then(setR).catch(() => {}); }, []);

  const best = r?.algorithms.reduce((a, b) => (b.best_reward > (a?.best_reward ?? -Infinity) ? b : a), r.algorithms[0]);
  const rc = r?.realcluster;

  return (
    <div className="space-y-4">
      {/* real-cluster headline */}
      {rc && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <Stat tone="positive" label="Fewer pods vs HPA" value={`${rc.pod_saving_pct}%`} sub="real cluster · 3-run mean" icon={<ShieldCheck className="h-4 w-4" />} />
          <Stat label="RL avg pods" value={`${rc.rl_pods.toFixed(2)}`} sub={`± ${rc.rl_pods_sd.toFixed(2)}`} icon={<BarChart3 className="h-4 w-4" />} />
          <Stat label="HPA avg pods" value={`${rc.hpa_pods.toFixed(2)}`} sub={`± ${rc.hpa_pods_sd.toFixed(2)}`} icon={<BarChart3 className="h-4 w-4" />} />
          <Stat label="p95 latency" value={`${Math.round(rc.rl_p95)} ms`} sub={`HPA ${Math.round(rc.hpa_p95)} ms`} icon={<ShieldCheck className="h-4 w-4" />} />
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.2fr_1fr]">
        {/* algorithm sweep */}
        <Card>
          <h3 className="mb-3 text-sm font-semibold">Algorithm comparison — training sweeps</h3>
          <div className="overflow-hidden rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-[11px] uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Algorithm</th>
                  <th className="px-3 py-2 text-right font-medium">Best reward</th>
                  <th className="px-3 py-2 text-right font-medium">Mean reward</th>
                  <th className="px-3 py-2 text-right font-medium">Runs</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {r?.algorithms.map((a) => (
                  <tr key={a.algorithm} className={a.algorithm === best?.algorithm ? "bg-agent/[0.05]" : ""}>
                    <td className="px-3 py-2.5 font-medium">
                      <span className={a.algorithm === best?.algorithm ? "text-agent" : "text-foreground"}>{a.algorithm}</span>
                      {a.algorithm === best?.algorithm && <span className="ml-2 inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-agent"><Trophy className="h-3 w-3" /> champion</span>}
                    </td>
                    <td className="px-3 py-2.5 text-right tabular-nums">{a.best_reward}</td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-muted-foreground">{a.mean_reward}</td>
                    <td className="px-3 py-2.5 text-right tabular-nums text-muted-foreground">{a.runs}</td>
                  </tr>
                ))}
                {!r?.algorithms.length && <tr><td className="px-3 py-4 text-center text-muted-foreground" colSpan={4}>No sweep results found.</td></tr>}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Higher (less negative) reward is better. PPO was selected as the deployed champion — most stable across runs.
          </p>
        </Card>

        {/* verdict */}
        <Card className="border-agent/30 bg-agent/[0.05]">
          <div className="flex items-start gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-agent/20 ring-1 ring-agent/40"><Trophy className="h-4 w-4 text-agent" /></div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-agent">Headline</div>
              <p className="mt-1 text-base font-medium leading-snug text-foreground">
                In simulation PPO is competitive with a tuned HPA and beats the conservative default. On a real
                cluster it used {rc ? <span className="text-agent">~{rc.pod_saving_pct}% fewer pods</span> : "fewer pods"} than
                the native HPA at comparable latency, reproducibly across 3 runs.
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                Honest limits: the agent is reactive (not yet predictive) and the real-cluster node saturates under load — a
                feasibility result, not a large statistical study.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
