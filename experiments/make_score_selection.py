"""Score x calibration decomposition of the selected-unit coverage gap, 11 random-forest settings ([S: score x calibration]).

Two scores x two calibrations on the same maps and splits (results/main_<ds>.csv; experiments/run_main.py):
  score |Y - mu|          split CP ("SCP |res|")      vs  JOMI on the reference set ("JOMI |res|")
  score |Y - mu| / sigma  normalized CP ("SCP stud")  vs  JOMI on the reference set ("JOMI stud")
so  score effect     = normalized CP - split CP            (a better error-spread model, no selection)
    selection effect = JOMI - split / normalized CP, same score  (conditioning on the selection)
Rules: the policy direction at 1%, 5% and 20% (figures/fig_data.policy_rule), and the quota rule's worst group
(states / countries selected >= 50 times over the 100 splits; figures/fig_data.quota_rule). Coverage pools all
splits (covered / selected); infinite intervals count as covering and their share is reported, with the
coverage among finite intervals alongside. Width: mean finite half-width, JOMI over the non-selective method.
    ~/miniforge3/envs/favela/bin/python experiments/make_score_selection.py
writes results/score_selection.csv and results/RESULTS_SCORE_SELECTION.md
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures")); sys.dont_write_bytecode = True
import fig_data as D                                                  # noqa: E402
import style as S                                                     # noqa: E402

PAIRS = {"res": ("SCP |res|", "JOMI |res|"), "stud": ("SCP stud", "JOMI stud")}
METH = [m for p in PAIRS.values() for m in p]
NAME = {"SCP |res|": "split CP", "JOMI |res|": "JOMI |res|", "SCP stud": "normalized CP", "JOMI stud": "JOMI stud"}


def pooled(nm):
    d = pd.read_csv(os.path.join(D.RES, f"main_{nm}.csv"))
    d = d[d.method.isin(METH)]
    rows = []
    for pct in (1, 5, 20):
        r = D.policy_rule(nm, pct)
        for m, g in d[d.rule == r].groupby("method"):
            sel, cov, inf = g.n_sel.sum(), g.n_cov.sum(), g.n_inf.sum()
            per = g.n_cov / g.n_sel
            rows.append(dict(rule=f"{pct}%", method=m, cov=cov / sel, inf=inf / sel,
                             cov_finite=(cov - inf) / (sel - inf) if sel > inf else np.nan,
                             width=g.hw_mean.mean(), split_below_085=(per < 0.85).mean(), K=g.n_sel.mean()))
    s = pd.read_csv(os.path.join(D.RES, f"state_{nm}.csv"))
    if "rule" not in s:
        s["rule"] = "quota"
    s = s[(s.rule == D.quota_rule(nm)) & s.method.isin(METH)].groupby(["method", "state"])[["n_sel", "n_cov", "n_inf"]].sum()
    for m, g in s[s.n_sel >= D.NMIN].groupby(level="method"):
        cv = g.n_cov / g.n_sel; k = cv.values.argmin()
        rows.append(dict(rule="quota worst group", method=m, cov=cv.values[k], inf=(g.n_inf / g.n_sel).values[k],
                         cov_finite=np.nan, width=np.nan, split_below_085=np.nan, K=float(len(g))))
    return pd.DataFrame(rows).assign(dataset=nm)


def table():
    T = pd.concat([pooled(nm) for nm, *_ in S.DATASETS], ignore_index=True)
    lab = {nm: lb for nm, lb, _ in S.DATASETS}; tr = {nm: t for nm, _, t in S.DATASETS}
    return T.assign(label=T.dataset.map(lab), truth=T.dataset.map(tr), name=T.method.map(NAME))


def wide(T, rule):
    X = T[T.rule == rule]
    C = X.pivot_table(index="dataset", columns="method", values="cov", sort=False)[METH]
    I = X.pivot_table(index="dataset", columns="method", values="inf", sort=False)[METH]
    return C, I


def md(T):
    order = [nm for nm, *_ in S.DATASETS]; lab = {nm: lb for nm, lb, _ in S.DATASETS}
    obs = [nm for nm, _, t in S.DATASETS if t == "observed"]
    L = ["# Score x calibration: what removes the selected-unit coverage gap (2026-10-29)", "",
         "Coverage of 90% intervals among the selected units, pooled over 100 random splits per setting "
         "(`experiments/make_score_selection.py`, from `results/main_*.csv`). Split CP and JOMI |res| share the "
         "score |Y - mu|; normalized CP and JOMI stud share |Y - mu| / sigma. Score effect = normalized CP - split CP; "
         "selection effect = JOMI - the non-selective method with the same score. † share of infinite intervals "
         "(counted as covering) where above 5%.", ""]
    for rule in ("1%", "5%", "20%", "quota worst group"):
        C, I = wide(T, rule)
        C = C.reindex(order); I = I.reindex(order)
        L += [f"## {rule}" + (" (policy direction)" if rule.endswith("%") else " (states / countries selected >= 50 times)"), "",
              "| setting | split CP | JOMI \\|res\\| | normalized CP | JOMI stud | score effect | selection effect, \\|res\\| | selection effect, stud |",
              "|---|---|---|---|---|---|---|---|"]
        for nm in order:
            c, i = C.loc[nm], I.loc[nm]
            cell = [f"{c[m]:.3f}" + (f" (†{i[m]:.2f})" if i[m] > 0.05 else "") for m in METH]
            L.append(f"| {lab[nm]} | " + " | ".join(cell) + f" | {c['SCP stud'] - c['SCP |res|']:+.3f} | "
                     f"{c['JOMI |res|'] - c['SCP |res|']:+.3f} | {c['JOMI stud'] - c['SCP stud']:+.3f} |")
        below = (C < 0.85).sum(); below_obs = (C.loc[obs] < 0.85).sum()
        L.append("| **settings < 0.85 (of 11)** | " + " | ".join(str(int(below[m])) for m in METH) + " | | | |")
        L.append("| **observed outcomes < 0.85 (of 6)** | " + " | ".join(str(int(below_obs[m])) for m in METH) + " | | | |")
        L.append("| **median** | " + " | ".join(f"{C[m].median():.3f}" for m in METH) + " | | | |")
        if rule.endswith("%"):
            X = T[T.rule == rule].set_index(["dataset", "method"])
            wr = {k: (X.xs(j, level="method").width / X.xs(s, level="method").width).median() for k, (s, j) in PAIRS.items()}
            fin = X.xs("JOMI stud", level="method").cov_finite.reindex(order)
            spl = X.xs("JOMI stud", level="method").split_below_085.reindex(order)
            L.append("")
            L.append(f"Median finite width, JOMI over the same-score method: |res| {wr['res']:.2f}x, stud {wr['stud']:.2f}x. "
                     "JOMI stud coverage among finite intervals: " + ", ".join(
                         f"{lab[nm]} {fin[nm]:.3f}" for nm in order if I.loc[nm, 'JOMI stud'] > 0.05) +
                     (" (others have no more than 5% infinite intervals)." if (I['JOMI stud'] > 0.05).any() else "none above 5% infinite.") +
                     f" Share of splits with JOMI stud below 0.85: median {spl.median():.2f}, max {spl.max():.2f} ({lab[spl.idxmax()]}).")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    T = table()
    T.to_csv(os.path.join(D.RES, "score_selection.csv"), index=False, float_format="%.4f")
    open(os.path.join(D.RES, "RESULTS_SCORE_SELECTION.md"), "w", encoding="utf-8").write(md(T) + "\n")
    print(md(T))
