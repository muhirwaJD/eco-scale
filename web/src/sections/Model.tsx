import { useEffect, useState } from "react";
import { Cpu, Gauge, Layers, Sparkles } from "lucide-react";
import { Card, MiniStat } from "@/components/eco/primitives";
import { getModel } from "@/api";
import type { ModelInfo } from "@/types";

export default function ModelSection() {
  const [m, setM] = useState<ModelInfo | null>(null);
  useEffect(() => { getModel().then(setM).catch(() => {}); }, []);
  if (!m) return <Card className="grid place-items-center py-16 text-sm text-muted-foreground">Loading model…</Card>;

  const meta = m.metadata ?? {};
  const hp = meta.hyperparameters ?? {};

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Card className="bg-agent/5 border-agent/25">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
            <span>Deployed agent</span><Sparkles className="h-4 w-4 text-agent" />
          </div>
          <div className="mt-2 text-3xl font-semibold text-agent">{meta.algorithm ?? "PPO"}</div>
          <div className="mt-1 text-xs text-muted-foreground">run {meta.run ?? "—"} · champion</div>
        </Card>
        <MetricCard label="Test reward" value={meta.mean_reward != null ? `${meta.mean_reward}` : "—"} sub={meta.std_reward != null ? `± ${meta.std_reward}` : ""} icon={<Gauge className="h-4 w-4" />} />
        <MetricCard label="Pod range" value={`${m.env.min_pods}–${m.env.max_pods}`} sub={`start ${m.env.start_pods}`} icon={<Layers className="h-4 w-4" />} />
        <MetricCard label="Healthy target" value={`${Math.round(m.reward.util_target * 100)}%`} sub="utilization the agent aims for" icon={<Cpu className="h-4 w-4" />} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* reward design */}
        <Card>
          <h3 className="mb-1 text-sm font-semibold">Reward design</h3>
          <p className="mb-3 text-xs text-muted-foreground">
            The reward the agent maximises — balancing service quality against energy. Energy is charged on the
            absolute pod count, which is what makes tracking demand optimal instead of over-provisioning.
          </p>
          <div className="grid grid-cols-2 gap-2">
            <MiniStat label="Latency penalty" val={`× ${m.reward.latency}`} />
            <MiniStat label="Energy / pod" val={`× ${m.reward.energy}`} />
            <MiniStat label="SLA breach" val={`× ${m.reward.sla_breach}`} />
            <MiniStat label="Scaling cost" val={`× ${m.reward.scaling}`} />
          </div>
        </Card>

        {/* training config */}
        <Card>
          <h3 className="mb-1 text-sm font-semibold">Training</h3>
          <p className="mb-3 text-xs text-muted-foreground">{meta.trained_on ?? "Real Alibaba 2018 cluster traces"}</p>
          <div className="grid grid-cols-2 gap-2">
            <MiniStat label="Learning rate" val={hp.learning_rate ?? "—"} />
            <MiniStat label="Gamma" val={hp.gamma ?? "—"} />
            <MiniStat label="Batch size" val={hp.batch_size ?? "—"} />
            <MiniStat label="n_steps" val={hp.n_steps ?? "—"} />
            <MiniStat label="Clip range" val={hp.clip_range ?? "—"} />
            <MiniStat label="Entropy coef" val={hp.ent_coef ?? "—"} />
          </div>
          {meta.note && <p className="mt-3 border-t border-border pt-3 text-[11px] leading-snug text-muted-foreground">{meta.note}</p>}
        </Card>
      </div>
    </div>
  );
}

function MetricCard({ label, value, sub, icon }: { label: string; value: string; sub?: string; icon: React.ReactNode }) {
  return (
    <Card>
      <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
        <span>{label}</span><span>{icon}</span>
      </div>
      <div className="mt-2 text-3xl font-semibold tabular-nums text-foreground">{value}</div>
      {sub && <div className="mt-1 text-xs text-muted-foreground">{sub}</div>}
    </Card>
  );
}
