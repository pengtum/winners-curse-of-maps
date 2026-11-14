"""FIA pre-2000 sensitivity ([S21]): every FIA coverage and carbon result, main table vs plots inventoried in 2000 or later.

The FIA table (experiments/data.py) keeps the latest inventory of 28,424 plot locations; the 5,356 last inventoried
before 2000 (5,355 in Florida) have basal area coded 0. "fia_post2000" drops them (23,068 plots, subsampled to
10,000 with the same seed) and was rerun with the same scripts and seeds (2026-10-29, excp snapshot of
vendor/excp_20261021):
    run_main.py / run_baselines.py 100 4 fia_post2000;  L1_FIXED=0.95 run_scgeoccp.py 100 4 fia_post2000
    CARBON_DATA=fia_post2000 CARBON_OUT=results/sens_fia_post2000 CARBON_PARTS=results/cache/carbon_parts_post2000
        run_carbon.py 20 100 4   (and CARBON_SELECT=random ... 10 100 4)
Coverage: as Fig. 4a (pooled over 100 splits; figures/fig_data.py), worst state under the quota rule (states
selected >= 50 times). Carbon: as Fig. 7 / results/RESULTS_CARBON.md, audit share 0.5, VCS family; paired random
control as Table C9.
    ~/miniforge3/envs/favela/bin/python experiments/make_fia_sensitivity.py
writes results/RESULTS_FIA_SENSITIVITY.md and results/fia_sensitivity_coverage.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "figures")); sys.dont_write_bytecode = True
import fig_data as D                                                  # noqa: E402
import style as S                                                     # noqa: E402
import make_carbon as MC                                              # noqa: E402

RES = os.path.join(ROOT, "results")
SETS = {"fia": "All plots (main)", "fia_post2000": "Inventoried 2000 or later"}
CARBON_DIR = {"fia": RES, "fia_post2000": os.path.join(RES, "sens_fia_post2000")}
METHODS = ["SCP |res|", "SCP stud", "CQR", "BY stud", "GeoCP", "LSCP", "RLCP", "Regression kriging", "QRF",
           "Bootstrap ensemble + residual", "JOMI stud", "SC-GeoCCP"]
RULES = {"top 1%": "top1", "top 5%": "top5", "top 20%": "top20", "budget": "knapsack"}


def coverage():
    rows = []
    for nm, lab in SETS.items():
        P = D.pooled(D.runs(nm))
        for rl, r in RULES.items():
            for m in METHODS:
                if (r, m) in P.index:
                    x = P.loc[(r, m)]
                    rows.append(dict(set=lab, rule=rl, method=S.SHORT[m], cov=x["cov"], inf=x.inf, width=x.width))
        jr = pd.read_csv(os.path.join(RES, f"main_{nm}.csv"))              # JOMI with the additive score
        for rl, r in RULES.items():
            g = jr[(jr.rule == r) & (jr.method == "JOMI |res|")]
            if len(g):
                rows.append(dict(set=lab, rule=rl, method="JOMI |res|", cov=g.n_cov.sum() / g.n_sel.sum(),
                                 inf=g.n_inf.sum() / g.n_sel.sum(), width=g.hw_mean.mean()))
        st = pd.read_csv(os.path.join(RES, f"state_{nm}.csv"))
        st = st[st.method.isin(["SCP |res|", "SCP stud", "JOMI stud"])].groupby(["method", "state"])[["n_sel", "n_cov"]].sum()
        for m, g in st[st.n_sel >= D.NMIN].groupby(level="method"):
            rows.append(dict(set=lab, rule="quota, worst state", method=S.SHORT.get(m, m),
                             cov=(g.n_cov / g.n_sel).min(), inf=np.nan, width=np.nan))
        G = D.groups(nm)
        for m in ("GeoCP", "SC-GeoCCP"):
            w = D.worst(G, "quota", m)
            rows.append(dict(set=lab, rule="quota, worst state", method=S.SHORT[m], cov=w[0], inf=w[1], width=np.nan))
    return pd.DataFrame(rows)


def carbon(nm):
    d = CARBON_DIR[nm]
    load = lambda tag: pd.concat([pd.read_csv(os.path.join(d, f"carbon_{f}{tag}.csv.gz")) for f in MC.MAPS.values()  # noqa: E731
                                  if os.path.exists(os.path.join(d, f"carbon_{f}{tag}.csv.gz"))], ignore_index=True)
    a, rnd = load(""), load("_random")
    a, rnd = a[a.audit == 0.5], rnd[rnd.audit == 0.5]
    rows = []
    for mp in MC.MAPS:
        for f in (0.01, 0.05):
            x = a[(a["map"] == mp) & (a.frac == f)]
            r = dict(set=SETS[nm], map=MC.TITLE[mp], frac=f, r2=x.drop_duplicates("fold").r2.mean(), maps=x.fold.nunique())
            for role in ("point", "map", "ma", "ours", "exact"):
                m = MC.col("VCS", role)
                r[f"risk_{role}"] = float(np.mean(x[m] > x.total))
            r["ours_credited"] = float(x[MC.col("VCS", "ours")].sum() / x.total.sum())
            y = rnd[(rnd["map"] == mp) & (rnd.frac == f)]
            xm = x[x.fold.isin(y.fold.unique())]
            for role in ("map", "ma"):
                m = MC.col("VCS", role)
                dd = ((xm[m] > xm.total).groupby(xm.fold).mean() - (y[m] > y.total).groupby(y.fold).mean()).dropna()
                r[f"paired_{role}"] = dd.mean(); r[f"paired_{role}_ci"] = 1.96 * dd.std(ddof=1) / np.sqrt(len(dd))
            rows.append(r)
    return pd.DataFrame(rows)


def md(C, K):
    L = ["# FIA pre-2000 sensitivity (2026-10-29)", "",
         "The FIA table holds 79,878 records (1,255 repeating a location and year) of 28,424 plot locations in ten states (FL 9,005, WA 5,129, MO 4,367, "
         "CA 3,292, ME 3,248, MT 1,490, VT 1,000, AZ 490, NM 388, CO 15 plots; latest inventory per location). "
         "Of the 5,356 plots last inventoried before 2000 (18.8%), the 5,355 in Florida (1980–1995) have basal area coded 0 although "
         "trees are recorded (median 59 trees per ha, mean diameter 6.3). This file compares every FIA coverage and "
         "carbon result on the main table with a rerun on the 23,068 plots inventoried in 2000 or later "
         "(`experiments/make_fia_sensitivity.py`; same scripts, seeds and subsample size).", "",
         "## Coverage among the selected units (90% intervals, 100 splits)", ""]
    order = list(dict.fromkeys(C.method))
    for rl in list(RULES) + ["quota, worst state"]:
        X = C[C.rule == rl].pivot_table(index="method", columns="set", values="cov", sort=False).reindex(order).dropna(how="all")
        L += [f"### {rl}", "", "| method | " + " | ".join(SETS.values()) + " | difference |", "|---|---|---|---|"]
        for m, r in X.iterrows():
            a, b = r.get(SETS["fia"], np.nan), r.get(SETS["fia_post2000"], np.nan)
            L.append(f"| {m} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
        if rl != "quota, worst state":
            n = {s: int((X[s] < 0.85).sum()) for s in SETS.values() if s in X}
            L.append("")
            L.append("Methods below 0.85: " + "; ".join(f"{s} {v}" for s, v in n.items()) + ".")
        L.append("")
    L += ["## Carbon: realized over-crediting risk under VCS (declared 1/3), audit share 0.5", "",
          "Columns: map total without deduction; protocol rule with the map's error layer; survey-calibrated estimate; "
          "selection-aware estimate (ours) and its credited share; exact bound; paired increase from map-based over "
          "random selection on the same maps (± map-clustered 95% half-width).", "",
          "| set | map | selected | R² | maps | no deduction | error layer | survey-calibrated | selection-aware | credited (sel.-aware) | exact | paired, error layer | paired, survey-calibrated |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in K.iterrows():
        L.append(f"| {r['set']} | {r['map']} | {r.frac:.0%} | {r.r2:.2f} | {r.maps} | {r.risk_point:.3f} | {r.risk_map:.3f} | "
                 f"{r.risk_ma:.3f} | {r.risk_ours:.3f} | {r.ours_credited:.3f} | {r.risk_exact:.3f} | "
                 f"{r.paired_map:+.3f} (±{r.paired_map_ci:.3f}) | {r.paired_ma:+.3f} (±{r.paired_ma_ci:.3f}) |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    C = coverage()
    K = pd.concat([carbon(nm) for nm in SETS], ignore_index=True)
    C.to_csv(os.path.join(RES, "fia_sensitivity_coverage.csv"), index=False, float_format="%.4f")
    K.to_csv(os.path.join(RES, "fia_sensitivity_carbon.csv"), index=False, float_format="%.4f")
    out = md(C, K)
    open(os.path.join(RES, "RESULTS_FIA_SENSITIVITY.md"), "w", encoding="utf-8").write(out)
    print(out)
