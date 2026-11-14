"""Replay the audit draws of run_audit_planner.py carbon on ranks alone (paper/AUDIT_PLANNER_DESIGN.md, execution record, D1).
Checks that every arm's |R| equals the run's (so ties and implementation are excluded), and redraws the whole-map arms with other
seeds to see whether the excess variance of |R| repeats.
    ~/miniforge3/envs/favela/bin/python experiments/check_audit_replay.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import nhypergeom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = [(1.0, 92), (1.0, 186), (1.0, 1869), (1.0, 3000), (0.05, 92), (0.05, 150), (0.10, 186), (0.10, 300)]


def replay(root, mid, n, M):
    out = []
    for f in range(20):
        for k in range(100):
            p = np.random.default_rng([root, mid, f, k, M]).permutation(M)  # positions in the map's order; 0 = highest key
            C = p[:n]; keep = np.ones(6000, bool); keep[C] = False
            out.append((C < np.flatnonzero(keep)[30]).sum())                 # above the 31st candidate
    return np.array(out)


def main():
    for mid, lab in ((0, "site"), (1, "cci")):
        d = pd.read_csv(os.path.join(ROOT, "results", "fia_db", "planner", f"carbon_{lab}.csv.gz"))
        for q, n in ARMS:
            g = d[(d.q == q) & (d.n == n)].sort_values(["fold", "split"])
            r = replay(6201, mid, n, int(round(q * 6000)))
            print(lab, q, n, "mismatches", int((r != g.nR.values).sum()), "var ratio %.3f" % (r.var(ddof=1) / nhypergeom(int(round(q * 6000)), n, 31).var()))
    for n in (1869, 3000):
        print("other seeds, whole map", n, [round(replay(7000 + s, 0, n, 6000).var(ddof=1) / nhypergeom(6000, n, 31).var(), 3) for s in range(12)])


if __name__ == "__main__":
    main()
