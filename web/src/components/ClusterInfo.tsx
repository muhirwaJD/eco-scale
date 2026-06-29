import { Boxes, Server } from "lucide-react";
import type { ClusterInfo as Info } from "../types";

export default function ClusterInfo({ info }: { info: Info | null }) {
  if (!info) return null;
  return (
    <div className="card p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Server size={15} className="text-eco-light" />
          <h2 className="text-sm font-semibold text-slate-100">Cluster</h2>
        </div>
        <span className="flex items-center gap-1.5 rounded-full bg-eco-green/10 px-2.5 py-1 text-[11px] font-medium text-eco-light ring-1 ring-eco-green/25">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-eco-light" />
          CONNECTED
        </span>
      </div>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-2.5 text-xs">
        <Row k="Context" v={info.context} />
        <Row k="Namespace" v={info.namespace} />
        <Row k="Deployment" v={info.deployment} />
        <Row k="Image" v={info.image} />
        <Row k="Native HPA" v={info.native_hpa ? "present" : "none (sole scaler)"} />
        <Row k="Replicas" v={`${info.replicas} · min ${info.min_pods} / max ${info.max_pods}`} />
      </dl>

      <div className="mt-4 border-t border-white/[0.06] pt-3">
        <div className="label mb-2 flex items-center gap-1.5">
          <Boxes size={12} /> Pods ({info.pods.length})
        </div>
        <div className="max-h-40 space-y-1 overflow-auto pr-1">
          {info.pods.map((p) => (
            <div
              key={p.name}
              className="flex items-center justify-between rounded-lg bg-white/[0.04] px-2.5 py-1.5 ring-1 ring-white/5"
            >
              <span className="truncate font-mono text-[11px] text-slate-300" title={p.name}>
                {p.name}
              </span>
              <span className="ml-2 flex shrink-0 items-center gap-2">
                <span className="tabular-nums text-[11px] text-slate-400">{p.cpu}</span>
                <span
                  className={`h-2 w-2 rounded-full ${
                    p.phase === "Running" ? "bg-eco-light shadow-[0_0_6px] shadow-eco-green/60" : "bg-eco-amber"
                  }`}
                  title={p.phase}
                />
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <dt className="label">{k}</dt>
      <dd className="truncate font-mono text-[12px] text-slate-200" title={v}>
        {v}
      </dd>
    </div>
  );
}
