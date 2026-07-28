import { useEffect, useState } from "react";
import { getConfig, liveAvailable, liveInfo, getContexts, useKubeContext } from "@/api";
import type { Config } from "@/types";

export interface ClusterVar {
  key: string;
  value: string;
  options: string[];
}

export function useClusterConfig() {
  const [config, setConfig] = useState<Config | null>(null);
  const [liveOk, setLiveOk] = useState(false);
  const [vars, setVars] = useState<ClusterVar[]>([
    { key: "cluster", value: "—", options: ["—"] },
    { key: "namespace", value: "default", options: ["default"] },
    { key: "deployment", value: "eco-sample-app", options: ["eco-sample-app"] },
  ]);

  // refresh cluster-derived bindings (after mount or a context switch)
  const refreshCluster = (contexts?: string[], current?: string) => {
    liveAvailable()
      .then((r) => {
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
      })
      .catch(() => {});
  };

  useEffect(() => {
    getConfig().then(setConfig).catch(() => {});
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
        .catch(() => {});
    }
  };

  return {
    config,
    liveOk,
    vars,
    onVarChange,
  };
}
