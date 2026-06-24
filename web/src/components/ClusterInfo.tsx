import { Boxes, Server } from "lucide-react";
import type { ClusterInfo as Info } from "../types";

export default function ClusterInfo({ info }: { info: Info | null }) {
  if (!info) return null;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
      <div className="mb-3 flex items-center gap-2">
        <Server size={15} className="text-eco-light" />
        <h2 className="text-sm font-semibold text-slate-200">Cluster</h2>
      </div>

      <dl className="space-y-1.5 text-xs">
        <Row k="Context" v={info.context} />
        <Row k="Namespace" v={info.namespace} />
        <Row k="Deployment" v={info.deployment} />
        <Row k="Image" v={info.image} />
        <Row k="Native HPA" v={info.native_hpa ? "present" : "none (agent is sole scaler)"} />
        <Row k="Replicas" v={`${info.replicas} (min ${info.min_pods} · max ${info.max_pods})`} />
      </dl>

      <div className="mt-3 border-t border-slate-800 pt-3">
        <div className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-slate-500">
          <Boxes size={12} /> Pods ({info.pods.length})
        </div>
        <div className="max-h-40 space-y-1 overflow-auto">
          {info.pods.map((p) => (
            <div key={p.name} className="flex items-center justify-between rounded-md bg-slate-800/60 px-2 py-1">
              <span className="truncate font-mono text-[11px] text-slate-300" title={p.name}>
                {p.name}
              </span>
              <span className="ml-2 flex shrink-0 items-center gap-2">
                <span className="tabular-nums text-[11px] text-slate-400">{p.cpu}</span>
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    p.phase === "Running" ? "bg-eco-light" : "bg-eco-amber"
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
    <div className="flex items-center justify-between gap-3">
      <dt className="text-slate-500">{k}</dt>
      <dd className="truncate font-mono text-[11px] text-slate-300" title={v}>{v}</dd>
    </div>
  );
}
