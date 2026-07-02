"""
plot_training_diagnostics.py — Optimizer-level training health metrics.

The learning curve (plot_learning_curves.py) shows *what* the agent achieved;
this shows *how* it trained, read from the TensorBoard logs SB3 writes during
training (`logs/<algo>/run_<n>/<ALGO>_1/events.out.tfevents.*`):

  PPO (champion, run 6):  value loss · policy-gradient loss · entropy loss
  DQN (best, run 1):      training loss · exploration rate (ε-decay)

REINFORCE is a custom (non-SB3) agent and does not emit these logs, so it is not
shown — this is the SB3 optimizer view.

Outputs:
  outputs/training/training_diagnostics.png
  outputs/training/training_diagnostics.csv   (persisted so the figure is
                                               reproducible without the logs/)

Run:  python utils/plot_training_diagnostics.py
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
OUT_DIR = os.path.join(ROOT, "outputs", "training")

PPO_COLOR, DQN_COLOR = "#19A979", "#E8743B"

# (algo, run) → the two runs we report on (champion PPO, best DQN).
PPO_RUN, DQN_RUN = 6, 1


def load_scalars(algo: str, run: int):
    """Return an EventAccumulator for one run's TB log, or None if absent."""
    matches = sorted(glob.glob(
        os.path.join(LOG_DIR, algo.lower(), f"run_{run}", f"{algo.upper()}_1", "events.out.tfevents.*")))
    if not matches:
        return None
    ea = event_accumulator.EventAccumulator(matches[0])
    ea.Reload()
    return ea


def series(ea, tag):
    """(steps, values) for a scalar tag, or ([], []) if the tag is missing."""
    if ea is None or tag not in ea.Tags()["scalars"]:
        return [], []
    ev = ea.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


def main() -> None:
    ppo = load_scalars("PPO", PPO_RUN)
    dqn = load_scalars("DQN", DQN_RUN)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training Diagnostics — optimizer health during training",
                 fontsize=15, fontweight="bold")
    rows = []

    def panel(a, steps, vals, title, ylabel, color, note=""):
        a.plot(steps, vals, color=color, linewidth=2)
        a.set_title(title, fontsize=12, fontweight="bold")
        a.set_xlabel("Training timesteps", fontsize=10)
        a.set_ylabel(ylabel, fontsize=10)
        a.grid(alpha=0.3, linestyle="--")
        a.set_facecolor("#F8F9FA")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        if note:
            a.text(0.98, 0.94, note, transform=a.transAxes, ha="right", va="top",
                   fontsize=8.5, color="#555", style="italic")

    # ── PPO (top row) ────────────────────────────────────────────
    specs_ppo = [
        ("train/value_loss", "PPO — Value Loss", "value loss", "critic learns the return"),
        ("train/policy_gradient_loss", "PPO — Policy-Gradient Loss", "PG loss", "→ 0 as policy stabilises"),
        ("train/entropy_loss", "PPO — Entropy Loss", "entropy loss", "less negative = policy sharpening"),
    ]
    for a, (tag, title, ylab, note) in zip(ax[0], specs_ppo):
        s, v = series(ppo, tag)
        panel(a, s, v, title, ylab, PPO_COLOR, note)
        rows += [{"algorithm": "PPO", "run": PPO_RUN, "metric": tag, "step": st, "value": round(val, 5)}
                 for st, val in zip(s, v)]

    # ── DQN (bottom row) ─────────────────────────────────────────
    s, v = series(dqn, "train/loss")
    panel(ax[1][0], s, v, "DQN — Training Loss", "TD loss", DQN_COLOR, "Bellman error shrinks")
    rows += [{"algorithm": "DQN", "run": DQN_RUN, "metric": "train/loss", "step": st, "value": round(val, 5)}
             for st, val in zip(s, v)]

    s, v = series(dqn, "rollout/exploration_rate")
    panel(ax[1][1], s, v, "DQN — Exploration Rate (ε)", "epsilon", DQN_COLOR, "ε-greedy anneals 1.0 → 0.05")
    rows += [{"algorithm": "DQN", "run": DQN_RUN, "metric": "rollout/exploration_rate", "step": st, "value": round(val, 5)}
             for st, val in zip(s, v)]

    # 6th cell: short legend/notes instead of an empty axis.
    ax[1][2].axis("off")
    ax[1][2].text(0.0, 0.9,
                  "Champion = PPO run 6\nBest DQN = run 1\n\n"
                  "PPO (on-policy): value loss collapses as the\n"
                  "critic learns; entropy drifts up (less negative)\n"
                  "as the policy commits.\n\n"
                  "DQN (off-policy): TD loss shrinks while ε\n"
                  "anneals from full exploration to 5%.\n\n"
                  "REINFORCE: custom non-SB3 agent — no\n"
                  "optimizer logs, so not shown.",
                  transform=ax[1][2].transAxes, va="top", fontsize=9.5, color="#333")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, "training_diagnostics.png")
    csv = os.path.join(OUT_DIR, "training_diagnostics.csv")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {png}")
    print(f"wrote {csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
