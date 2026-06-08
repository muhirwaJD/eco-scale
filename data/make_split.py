"""
make_split.py — Deterministic train/test split of the 13 Alibaba traces.

Stratified by difficulty (number of "stress" steps where load exceeds what the
start-pod count can serve) so both train and test span the easy->hard range.
This keeps train and test from the SAME distribution, which is required for the
DQN-vs-HPA paired t-test to be a fair comparison (no covariate shift).

Writes data/split.json:  {"train": [...paths...], "test": [...paths...]}

Run: python data/make_split.py
"""

import os
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_GLOB = os.path.join(HERE, "traces", "trace_*.npy")
SPLIT_PATH = os.path.join(HERE, "split.json")

N_TEST = 5            # 13 traces -> 8 train / 5 test
STRESS_LEVEL = 0.40   # load above this stresses the start-pod count (5 * 0.08)


def difficulty(path):
    a = np.load(path)
    return int((a > STRESS_LEVEL).sum())   # # of stressed steps


def main():
    paths = sorted(glob.glob(TRACE_GLOB))
    if not paths:
        raise SystemExit(f"No traces found at {TRACE_GLOB}")

    # sort easy -> hard, then interleave so test samples the whole range
    ranked = sorted(paths, key=difficulty)
    test, train = [], []
    # take every (len/N_TEST)-th trace for test -> evenly spread across difficulty
    step = len(ranked) / N_TEST
    test_idx = {int(round(i * step)) for i in range(N_TEST)}
    for i, p in enumerate(ranked):
        (test if i in test_idx else train).append(p)

    # store repo-relative paths for portability
    rel = lambda p: os.path.relpath(p, os.path.dirname(HERE))
    split = {"train": [rel(p) for p in sorted(train)],
             "test":  [rel(p) for p in sorted(test)]}
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)

    print(f"✅ Split written to {os.path.relpath(SPLIT_PATH, os.path.dirname(HERE))}")
    print(f"   train ({len(train)}): " +
          ", ".join(f"{os.path.basename(p)}[{difficulty(p)}]" for p in sorted(train)))
    print(f"   test  ({len(test)}): " +
          ", ".join(f"{os.path.basename(p)}[{difficulty(p)}]" for p in sorted(test)))
    print("   ([N] = stressed steps out of 288)")


if __name__ == "__main__":
    main()
