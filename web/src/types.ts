export interface SimState {
  tick: number;
  max_ticks: number;
  day_progress: number;
  cpu: number;
  mode?: "live";              // present only in live-cluster mode
  avg_cpu_millicores?: number;
  applied?: boolean;          // live: whether the action was actually kubectl-scaled
  rl: {
    pods: number;
    latency?: number;
    required_pods?: number;
    action: number;
    action_name: string;
    observation: number[];
    probs: number[] | null;
    rationale: string;
  };
  hpa?: {
    pods: number;
    latency: number;
    action: number;
    action_name: string;
    target: number;
  };
  savings?: {
    pod_ticks_saved: number;
    kwh: number;
    frw: number;
    breaches_avoided: number;
  };
  stats?: {
    replicas: number;
    peak_pods: number;
    scaling_actions: number;
    up: number;
    down: number;
    hold: number;
  };
  done: boolean;
}

export interface Config {
  min_pods: number;
  max_pods: number;
  pod_capacity: number;
  util_target: number;
  agent: string;
  run: number;
}

export type Mode = "recommend" | "autopilot";

export interface ClusterInfo {
  context: string;
  namespace: string;
  deployment: string;
  image: string;
  replicas: number;
  native_hpa: boolean;
  min_pods: number;
  max_pods: number;
  pods: { name: string; phase: string; cpu: string }[];
}

export interface LoadStatus {
  running: boolean;
  intensity: number;
  elapsed: number;
  duration: number;
  p95_ms: number;
}
