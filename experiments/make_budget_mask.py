"""Budget-rule and publication-mask coverage among the selected units, 11 random-forest settings ([S2], [S3]).

Same pooling as figures/fig_data.py (coverage = covered / selected over the 100 splits; 95% interval from the
split-to-split variation) and the same 12 methods as Fig. 4a. Rules (experiments/run_main.py, run_baselines.py,
run_scgeoccp.py): the greedy travel-cost budget, value per cost (knapsack; for ERA5 the opposite of its bottom policy
direction) and need per cost for DHS (knapsack_need;
figures/fig_data.budget_rule) and the two kernel publication masks (mask_h2: GeoCP's tuned bandwidth; mask_base: the
median distance to the 10th-nearest training unit). SC-GeoCCP was not run on masks.
    ~/miniforge3/envs/favela/bin/python experiments/make_budget_mask.py
writes results/RESULTS_BUDGET_MASK.md and results/budget_mask_coverage.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures")); sys.dont_write_bytecode = True
import fig_data as D                                                  # noqa: E402
import style as S                                                     # noqa: E402

METHODS = ["SCP |res|", "SCP stud", "CQR", "BY stud", "GeoCP", "LSCP", "RLCP", "Regression kriging", "QRF",
           "Bootstrap ensemble + residual", "JOMI stud", "SC-GeoCCP"]
RULES = {"budget": D.budget_rule, "mask_h2": lambda nm: "mask_h2", "mask_base": lambda nm: "mask_base"}


def table():
    rows = []
    for nm, lab, _ in S.DATASETS:
        P = D.pooled(D.runs(nm))
        for key, rule in RULES.items():
            for m in METHODS:
                if (rule(nm), m) not in P.index:
                    continue
                r = P.loc[(rule(nm), m)]
                rows.append(dict(rule=key, dataset=nm, label=lab, method=S.SHORT[m], coverage=r["cov"], ci_lo=r.lo,
                                 ci_hi=r.hi, inf_share=r.inf, n_selected=r.K, splits=r.splits))
    return pd.DataFrame(rows)


def md(T):
    L = ["# Budget and publication-mask rules on real data (2026-10-28)", "",
         "Coverage of 90% intervals among the selected units, pooled over 100 random splits per setting "
         "(`experiments/make_budget_mask.py`; same pooling as Fig. 4a). Columns are the Fig. 4a methods. "
         "† JOMI and SC-GeoCCP return infinite intervals for this share of the selected units.", ""]
    order = list(dict.fromkeys(T.method))
    for key, title in (("budget", "Greedy travel-cost budget: value per cost, need per cost for DHS (Supplementary Fig. 2)"),
                       ("mask_h2", "Publication mask, GeoCP's tuned bandwidth"),
                       ("mask_base", "Publication mask, base bandwidth")):
        X = T[T.rule == key]
        cov = X.pivot_table(index="label", columns="method", values="coverage", sort=False).reindex(columns=order)
        n = X.groupby("label", sort=False).n_selected.mean()
        inf = X[X.method == "JOMI"].set_index("label").inf_share       # JOMI and SC-GeoCCP share it; others noted below
        L += [f"## {title}", "", "| setting | selected per split | " + " | ".join(c for c in cov.columns if c in cov) + " |",
              "|---|---|" + "---|" * cov.shape[1]]
        for lab, row in cov.iterrows():
            flag = f" (†{inf[lab]:.2f})" if inf.get(lab, 0) > 0.05 else ""
            L.append(f"| {lab}{flag} | {n[lab]:.1f} | " + " | ".join("" if np.isnan(v) else f"{v:.3f}" for v in row) + " |")
        run = cov.notna().any()                                       # SC-GeoCCP was not run on masks
        L.append("| **median** | | " + " | ".join(f"{v:.3f}" if r else "not run" for v, r in zip(cov.median(), run)) + " |")
        L.append("| **settings < 0.85** | | " + " | ".join(str(int(v)) if r else "" for v, r in zip((cov < 0.85).sum(), run)) + " |")
        big = X[X.inf_share > 0.05]
        if len(big):
            L.append("Infinite intervals (share of the selected units): " + "; ".join(
                f"{r.method}, {r.label} {r.inf_share:.2f}" for r in big.itertuples()) + ".")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    T = table()
    T.to_csv(os.path.join(ROOT, "results", "budget_mask_coverage.csv"), index=False, float_format="%.4f")
    open(os.path.join(ROOT, "results", "RESULTS_BUDGET_MASK.md"), "w", encoding="utf-8").write(md(T) + "\n")
    print(md(T))
