export interface SimState {
  tick: number;
  max_ticks: number;
  day_progress: number;
  cpu: number;
  rl: {
    pods: number;
    latency: number;
    required_pods: number;
    action: number;
    action_name: string;
    observation: number[];
    probs: number[] | null;
    rationale: string;
  };
  hpa: {
    pods: number;
    latency: number;
    action: number;
    action_name: string;
    target: number;
  };
  savings: {
    pod_ticks_saved: number;
    kwh: number;
    frw: number;
    breaches_avoided: number;
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
