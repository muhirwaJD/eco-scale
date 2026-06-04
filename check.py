import os
os.environ["TRACE_DIR"] = "./data"

from environment.custom_env import KubernetesEnv

for tt in ["cyclical", "burst"]:
    env = KubernetesEnv(trace_type=tt)
    obs, _ = env.reset()
    total = 0
    for _ in range(288):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc: break
    print(tt, "steps", info["step"], "reward", round(total,1), "cpu range seen", round(env.trace.min(),2), round(env.trace.max(),2))        