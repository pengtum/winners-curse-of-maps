"""Reference-set size of a top- or bottom-K rule under a random split, and the share of JOMI's infinite intervals.

Under a random split the calibration and test units are exchangeable in their ranks by the sort key, so the number of
calibration units ranked above the (K+1)-th test unit, |R| (Methods: n_R = l - K), is negative hypergeometric given the
calibration size n, the test size m and K. JOMI's 90% interval is infinite when |R| <= 8, so its probability is known
before any audited outcome is measured. This script compares that probability with the observed share of infinite
intervals (JOMI, normalized score; the per-split results of run_main.py) in the policy direction, and tabulates the
calibration size at which a finite interval has probability 0.5 and 0.9.
    ~/miniforge3/envs/favela/bin/python experiments/refset_size_law.py
reads results/main_<ds>.csv; writes results/RESULTS_REFSET_SIZE.md
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import nhypergeom

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import data as DS                                                       # noqa: E402

OUT = os.path.join(ROOT, "results")
NMAX = 8                                                                # +inf when |R| <= 8 at alpha = 0.1


def p_infinite(n, m, K):
    return float(nhypergeom(n + m, n, K + 1).cdf(NMAX))


def n_needed(m, K, prob):
    n = NMAX + 1
    while 1.0 - p_infinite(n, m, K) < prob:
        n += 1
    return n


def main():
    rows = []
    for ds in DS.DISPLAY:
        d = pd.read_csv(os.path.join(OUT, f"main_{ds}.csv"))
        for share in (1, 5, 20):
            rule = f"{DS.POLICY[ds]}{share}"
            x = d[(d.method == "JOMI stud") & (d.rule == rule)]
            n, m, K = x.n_cal.unique(), x.m.unique(), x.n_sel.unique()
            assert len(n) == len(m) == len(K) == 1, (ds, rule)
            n, m, K = int(n[0]), int(m[0]), int(K[0])
            pinf = p_infinite(n, m, K)
            rows.append(dict(dataset=ds, rule=rule, n=n, m=m, K=K, R_mean_pred=float(nhypergeom(n + m, n, K + 1).mean()),
                             R_mean_obs=float(x.R_mean.mean()), inf_pred=pinf, inf_obs=float(x.n_inf.sum() / x.n_sel.sum()),
                             inf_iff_small=bool(((x.n_inf > 0) == (x.R_min <= NMAX)).all())))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "refset_size_law.csv"), index=False, float_format="%.5f")
    lab = {nm: l for nm, l in zip(DS.DISPLAY, DS.DISPLAY)}
    L = ["# Reference-set size under a random split and JOMI's infinite intervals (2026-10-30)", "",
         "`experiments/refset_size_law.py`. For a top- or bottom-K rule, |R| is the number of calibration units ranked above "
         "the (K+1)-th test unit, negative hypergeometric given n, m and K; JOMI's 90% interval is infinite when |R| <= 8. "
         "Observed: JOMI with the normalized score, policy direction, 100 splits (`results/main_<ds>.csv`). "
         "'Infinite only when |R| <= 8' checks, split by split, that a split has infinite intervals exactly when its |R| <= 8.", "",
         "| Setting | Rule | n | m | K | mean abs R, predicted | observed | share infinite, predicted | observed | infinite only when abs R <= 8 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        if r.inf_pred < 0.005 and r.inf_obs == 0:
            continue
        L.append(f"| {lab[r.dataset]} | {r.rule} | {r.n} | {r.m} | {r.K} | {r.R_mean_pred:.1f} | {r.R_mean_obs:.1f} | "
                 f"{r.inf_pred:.2f} | {r.inf_obs:.2f} | {'yes' if r.inf_iff_small else 'no'} |")
    L += ["", f"Settings and rules not listed had predicted and observed shares below 0.005 and 0. "
          f"All {len(T)} (setting, rule) pairs: infinite intervals occur exactly in the splits with |R| <= 8: "
          f"{'yes' if T.inf_iff_small.all() else 'NO'}.", "",
          "## Calibration size for a finite 90% interval (probability 0.5 and 0.9), as n times K/m", "",
          "| m | K/m | n for probability 0.5 | x K/m | n for probability 0.9 | x K/m |", "|---|---|---|---|---|---|"]
    for m in (1000, 3000, 10000):
        for f in (0.01, 0.05, 0.20):
            K = int(round(f * m))
            a, b = n_needed(m, K, 0.5), n_needed(m, K, 0.9)
            L.append(f"| {m} | {f:.2f} | {a} | {a * K / m:.1f} | {b} | {b * K / m:.1f} |")
    open(os.path.join(OUT, "RESULTS_REFSET_SIZE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
