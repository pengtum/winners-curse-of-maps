"""Tables for the out-of-design evaluation (paper/OOD_DESIGN.md; experiments/run_ood.py).

Held-out groups (loco: DHS countries; loso: FIA states): coverage among the selected units per arm (R0 map with
the group's own data; R1 map from other groups, local calibration; R2 the same map, calibration from other,
non-training groups), pooled over groups x maps x re-splits (K-weighted), equal-weight mean over groups, the
number of groups below 0.85 with its binomial noise reference, the worst group, coverage among finite intervals
and the infinite share; paired R2 - R1 differences for JOMI with a group-clustered interval; per-group JOMI coverage; the local-survey
curve (R1-n); and, for DHS at 25.2%, how often the two mean-wealth bounds held. Block designs: coverage per
method and rule against the random-split comparison run through the same pipeline.
    ~/miniforge3/envs/favela/bin/python experiments/make_ood.py
writes results/RESULTS_OOD.md and results/ood_summary.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures")); sys.dont_write_bytecode = True
import fig_data as D                                                  # noqa: E402
import style as S                                                     # noqa: E402

RES = os.path.join(ROOT, "results")
FIA = os.environ.get("QW2_FIA", "fia_db")               # == experiments/data.py FIA
OFIA = "fia_post2000" if FIA == "fia" else FIA           # old table: held-out states on the 2000+ plots
HELD = [("loco_poverty", "DHS wealth, leave one country out"),
        (f"loso_{OFIA}_siteheight", "FIA biomass (site + height map), leave one state out"),
        (f"loso_{OFIA}_full", "FIA biomass (full-inventory map), leave one state out")]
BLOCK = [("poverty", "country", "DHS wealth, countries as blocks"), (OFIA, "k100", "FIA, 100 k-means blocks"),
         (OFIA, "k25", "FIA, 25 k-means blocks")]
ORDER = ["SCP |res|", "SCP stud", "BY stud", "CQR", "GeoCP", "LSCP", "RLCP", "Regression kriging", "QRF",
         "Bootstrap ensemble + residual", "JOMI |res|", "JOMI stud", "SC-GeoCCP"]
NAME = {**S.SHORT, "JOMI |res|": "JOMI (additive score)", "JOMI stud": "JOMI"}
LOCAL = {"GeoCP", "LSCP", "RLCP"}            # need calibration units near the test unit: not ranked in R2
K10 = 571                                     # 5% rule: groups with n_g >= 571 (|T_g| >= 200, K >= 10), fixed in advance
IS5 = ("bottom5", "top5")
SURV = {"all (half of the group)": 0, "200": 1, "100": 2, "50": 3}


def group_se(g):
    """Per group: pooled coverage and its re-split standard error given the maps. The units selected in one
    split share one calibration draw, so the draw (map, re-split), not the unit, is the independent replicate:
    se^2 = sum_m s_m^2 / (I M^2), s_m^2 the variance of per-split coverage within map m."""
    d = g.groupby(["group", "map", "inner"])[["n_cov", "n_sel"]].sum(); d["c"] = d.n_cov / d.n_sel
    v = d.groupby(["group", "map"]).c.agg(["var", "count"])
    se = np.sqrt((v["var"] / v["count"]).groupby(level="group").sum()) / v.groupby(level="group").size()
    by = g.groupby("group")[["n_cov", "n_sel"]].sum()
    return by.n_cov / by.n_sel, se.reindex(by.index)


def noise(se, reps=4000, seed=0):
    """Expected number of groups below 0.85 and expected worst group if every group's coverage given each map
    were exactly 0.90 and only re-split noise (normal, se from group_se) moved the estimates."""
    z = 0.9 + np.random.default_rng(seed).standard_normal((reps, len(se))) * se.values[None, :]
    return float((z < 0.85).sum(1).mean()), float(z.min(1).mean())


def load(tag):
    f = os.path.join(RES, f"ood_{tag}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def pool(x):
    s = x[["n_cov", "n_sel", "n_inf"]].sum()
    fin = s.n_sel - s.n_inf
    return pd.Series(dict(cov=s.n_cov / s.n_sel, cov_fin=(s.n_cov - s.n_inf) / fin if fin > 0 else np.nan,
                          inf=s.n_inf / s.n_sel, width=x.hw_mean_fin.mean()))


def held_table(df, rule, arms=("R0", "R1", "R2")):
    x = df[(df.rule == rule) & df.arm.isin(arms)]
    if rule in IS5:
        x = x[x.n_group >= K10]
    rows = []
    for (arm, m), g in x.groupby(["arm", "method"]):
        p = pool(g); cg, se = group_se(g); nb, nw = noise(se)
        rows.append(dict(arm=arm, method=m, groups=len(cg), cov=p["cov"], cov_fin=p.cov_fin, inf=p.inf,
                         width=p.width, cov_eq=cg.mean(), worst=cg.min(), worst_group=int(cg.idxmin()),
                         below=int((cg < 0.85).sum()), noise_worst=nw, noise_below=nb,
                         sig_below=int((cg + 1.645 * se < 0.90).sum()), se_med=float(se.median())))
    return pd.DataFrame(rows)


def paired(df, rule, a="R1", b="R2", method="JOMI stud"):
    x = df[(df.rule == rule) & (df.method == method) & df.arm.isin([a, b])]
    if rule in IS5:
        x = x[x.n_group >= K10]
    w = x.pivot_table(index=["group", "map", "inner"], columns="arm", values="n_cov", aggfunc="sum")
    n = x.pivot_table(index=["group", "map", "inner"], columns="arm", values="n_sel", aggfunc="sum")
    dlt = (w[b] / n[b] - w[a] / n[a]).groupby(level="group").mean()
    G = len(dlt); se = dlt.std(ddof=1) / np.sqrt(G)
    from scipy.stats import t
    return dlt.mean(), t.ppf(0.975, G - 1) * se, G, int((dlt > 0).sum())


def paired_fin(df, rule, a="R1", b="R2", method="JOMI stud"):
    """Finite intervals only: per group, pooled finite-only coverage in b minus a; groups as clusters."""
    x = df[(df.rule == rule) & (df.method == method) & df.arm.isin([a, b])]
    if rule in IS5:
        x = x[x.n_group >= K10]
    s = x.groupby(["group", "arm"])[["n_cov", "n_sel", "n_inf"]].sum()
    c = ((s.n_cov - s.n_inf) / (s.n_sel - s.n_inf)).unstack("arm"); dlt = (c[b] - c[a]).dropna()
    G = len(dlt)
    from scipy.stats import t
    return dlt.mean(), t.ppf(0.975, G - 1) * dlt.std(ddof=1) / np.sqrt(G), G


def per_group(df, rule, method="JOMI stud"):
    x = df[(df.rule == rule) & (df.method == method) & df.arm.isin(["R0", "R1", "R2"])]
    if rule in IS5:
        x = x[x.n_group >= K10]
    out = {}
    for arm, g in x.groupby("arm"):
        c, se = group_se(g); fin = g.groupby("group")[["n_inf", "n_sel"]].sum()
        m = g.groupby(["group", "map"])[["n_cov", "n_sel"]].sum(); cm = (m.n_cov / m.n_sel).groupby(level="group")
        out[arm] = pd.DataFrame({"cov": c, "se": se, "se_between": cm.std(ddof=1) / np.sqrt(cm.size()),
                                 "inf": fin.n_inf / fin.n_sel})
    T = pd.concat(out, axis=1); T["n_group"] = x.groupby("group").n_group.first()
    return T


def survey(df, method="JOMI stud"):
    """Local survey size: R1 with n of the C_g units vs all of them, on the first SURVEY_INNER re-splits, pooled over
    the groups that have every n (R1-n exists only where n < |C_g|)."""
    sv = df[df.arm.str.startswith("R1") & (df.method == method) & (df.rule != "all") & (df.inner < 20)]
    rows = []
    for rule, g in sv.groupby("rule"):
        g = g[g.n_group >= K10] if rule in IS5 else g
        arms = g.arm.unique(); have = g.groupby("group").arm.nunique(); keep = have.index[have == len(arms)]
        g = g[g.group.isin(keep)]
        for arm, x in g.groupby("arm"):
            n = arm.split("-")[1] if "-" in arm else "all (half of the group)"
            rows.append(dict(rule=rule, n=n, order=SURV[n], groups=len(keep), **pool(x)))
    return pd.DataFrame(rows).sort_values(["rule", "order"]) if rows else pd.DataFrame()


def fmt_held(T):
    T = T.copy(); T["m"] = pd.Categorical(T.method, ORDER, ordered=True); T = T.sort_values(["m", "arm"])
    L = ["| method | arm | coverage | finite only | infinite | equal-weight mean | groups < 0.85 (noise) | worst group (noise) | "
         "groups below 0.90 (one-sided 5%) | mean half-width |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        flag = " ‡" if r.arm == "R2" and r.method in LOCAL else ""
        L.append(f"| {NAME.get(r.method, r.method)}{flag} | {r.arm} | {r['cov']:.3f} | {r.cov_fin:.3f} | {r.inf:.2f} | "
                 f"{r.cov_eq:.3f} | {r.below} of {r.groups} ({r.noise_below:.1f}) | {r.worst:.3f} ({r.noise_worst:.3f}) | "
                 f"{r.sig_below} | {r.width:.3g} |")
    return L


def held_section(tag, title):
    df = load(tag)
    if df is None:
        return [f"## {title}", "", "Not run yet.", ""], None
    rules = [r for r in df.rule.unique() if r != "all"]
    L = [f"## {title}", "",
         f"{df.group.nunique()} groups x {df['map'].nunique()} maps x {df.inner.nunique()} calibration / test re-splits. "
         "R0: map trained with 30% of the held-out group plus other groups; R1: map from the other (training) groups only, "
         "calibrated on a random half of the 70% of the held-out group not used by the R0 map (35% of the group); R2: the "
         "R1 map calibrated on as many units from other groups that did not train the map. "
         "The 5% rule is reported only for groups with n_g >= 571 (K >= 10), fixed before the run. "
         "Coverage counts infinite intervals as covering; 'finite only' drops them. Noise: expected number of groups below 0.85, "
         "and expected worst group, if every group's coverage given each map were exactly 0.90 and only re-split noise moved "
         "the estimates (normal; standard error from the variance of per-split coverage within each map, because the units "
         "selected in one split share one calibration draw; the pre-registered independent-binomial reference ignores this "
         "and is slightly tighter, with an expected worst group within 0.002 here). These standard errors are conditional "
         "on the five map and group-split draws; the per-group tables also give the between-draw standard error (standard "
         "deviation of the five per-map coverages / sqrt 5), which is larger in R2. 'Below 0.90': groups whose coverage plus 1.645 standard errors is below 0.90 (unadjusted). "
         "‡ localized methods with calibration units in other groups: kernel mass near zero, so their R2 rows reflect "
         "fallbacks and are not ranked.", ""]
    out = []
    for rule in rules + ["all"]:
        T = held_table(df, rule)
        if T.empty:
            L += [f"### {rule}", "", "No group meets the size threshold.", ""]; continue
        T.insert(0, "rule", rule); T.insert(0, "design", tag); out.append(T)
        L += [f"### {rule}", ""] + fmt_held(T) + [""]
        if rule != "all":
            for meth in ("JOMI stud", "JOMI |res|", "SCP stud", "SCP |res|"):
                mdl, ci, G, pos = paired(df, rule, method=meth)
                L.append(f"Paired R2 − R1, {NAME.get(meth, meth)}: {mdl:+.3f} (95% interval ±{ci:.3f}, {G} groups as clusters; "
                         f"R2 higher in {pos} of {G}).  ")
                if meth.startswith("JOMI"):
                    mf, cf, Gf = paired_fin(df, rule, method=meth)
                    L.append(f"Paired R2 − R1, {NAME.get(meth, meth)}, finite intervals only: {mf:+.3f} (±{cf:.3f}, {Gf} groups).  ")
                mdl, ci, G, pos = paired(df, rule, "R0", "R1", method=meth)
                L.append(f"Paired R1 − R0, {NAME.get(meth, meth)}: {mdl:+.3f} (±{ci:.3f}).  ")
            L.append("")
            P = per_group(df, rule)
            L += [f"Per group, JOMI ({rule}): coverage (re-split s.e.; between-draw s.e.) [infinite share]", "",
                  "| group | n_g | R0 | R1 | R2 | R2 − R1 |", "|---|---|---|---|---|---|"]
            for gi, r in P.sort_values(("R2", "cov")).iterrows():
                cell = lambda a: f"{r[(a, 'cov')]:.3f} ({r[(a, 'se')]:.3f}; {r[(a, 'se_between')]:.3f}) [{r[(a, 'inf')]:.2f}]"   # noqa: E731
                L.append(f"| {gi} | {int(r.n_group.iloc[0]) if hasattr(r.n_group, 'iloc') else int(r.n_group)} | {cell('R0')} | {cell('R1')} | {cell('R2')} | "
                         f"{r[('R2', 'cov')] - r[('R1', 'cov')]:+.3f} |")
            L.append("")
            P.columns = ["_".join(c) if isinstance(c, tuple) and c[1] else (c[0] if isinstance(c, tuple) else c) for c in P.columns]
            P.insert(0, "rule", rule); P.insert(0, "design", tag)
            P.reset_index().to_csv(os.path.join(RES, f"ood_pergroup_{tag}_{rule}.csv"), index=False, float_format="%.4f")
    S_ = survey(df)
    if len(S_):
        L += ["### Local survey size (R1 map, JOMI, first 20 re-splits)", "",
              "R1-n is run only where n < |C_g|; each rule is pooled over the groups that have every n.", "",
              "| rule | calibration units | groups | coverage | finite only | infinite | mean half-width |",
              "|---|---|---|---|---|---|---|"]
        for _, r in S_.iterrows():
            L.append(f"| {r.rule} | {r.n} | {r.groups} | {r['cov']:.3f} | {r.cov_fin:.3f} | {r.inf:.2f} | {r.width:.3g} |")
        L.append("")
    b = os.path.join(RES, f"ood_{tag}_bounds.csv")
    if os.path.exists(b):
        B = pd.read_csv(b)
        L += ["### 90% upper bound on the selected clusters' mean wealth (25.2% rule)", "",
              "| arm | bound | holds (pooled) | countries below 0.80 | worst country | mean bound − truth | mean error at the selected | mean error in calibration |",
              "|---|---|---|---|---|---|---|---|"]
        for (arm, bn), g in B.groupby(["arm", "bound"]):
            c = g.groupby("group").holds.mean()
            L.append(f"| {arm} | {bn} | {g.holds.mean():.3f} | {int((c < 0.80).sum())} of {len(c)} | {c.min():.3f} | "
                     f"{g.excess.replace([np.inf], np.nan).mean():+.3f} | {g.bias_sel.mean():+.3f} | {g.bias_cal.mean():+.3f} |")
        L.append("")
    return L, (pd.concat(out, ignore_index=True) if out else None)


def block_section():
    L = ["## Block-randomized splits", "",
         "Blocks allotted ~40/30/30 by units to training / calibration / test; 4,000 / 3,000 / 3,000 units drawn from them; "
         "100 splits. 'random' is the same pipeline with single units as blocks (the main design). No theorem covers block "
         "randomization; coverage is approximate. s.e.: standard error over splits (per-split coverage, sd / 10); "
         "p10: 10th percentile of per-split coverage.", ""]
    out = []
    for name, blk, title in BLOCK:
        a, r = load(f"block_{name}_{blk}"), load(f"block_{name}_random")
        if a is None:
            L += [f"### {title}", "", "Not run yet.", ""]; continue
        L += [f"### {title}", "", "| rule | method | blocks (s.e.) | random split (s.e.) | blocks: finite only | blocks: infinite | blocks: p10 |",
              "|---|---|---|---|---|---|---|"]
        for rule in [x for x in a.rule.unique()]:
            for m in ORDER:
                g = a[(a.rule == rule) & (a.method == m)]
                if not len(g):
                    continue
                p = pool(g); c = g.n_cov / g.n_sel; p10 = c.quantile(0.1); se = c.std(ddof=1) / np.sqrt(len(c))
                rr = r[(r.rule == rule) & (r.method == m)] if r is not None else g.iloc[:0]
                pr, ser = (pool(rr)["cov"], (rr.n_cov / rr.n_sel).std(ddof=1) / np.sqrt(len(rr))) if len(rr) else (np.nan, np.nan)
                L.append(f"| {rule} | {NAME.get(m, m)} | {p['cov']:.3f} ({se:.3f}) | {pr:.3f} ({ser:.3f}) | {p.cov_fin:.3f} | "
                         f"{p.inf:.2f} | {p10:.3f} |")
                out.append(dict(design=f"block_{name}_{blk}", rule=rule, method=m, cov=p["cov"], se=se, cov_random=pr,
                                se_random=ser, cov_fin=p.cov_fin, inf=p.inf, p10=p10))
        L.append("")
    return L, pd.DataFrame(out)


SI_DESIGNS = [("loco_poverty", "DHS", ("bottom25.2", "bottom5")),
              (f"loso_{OFIA}_siteheight", "FIA, site + height map", ("top20", "top5")),
              (f"loso_{OFIA}_full", "FIA, full-inventory map", ("top20", "top5"))]
RULE = {"bottom25.2": "poorest 25.2%", "bottom5": "poorest 5%", "top20": "highest 20%", "top5": "highest 5%"}


def r2(x):
    """Two decimals, half up, exact for shares such as k/200 (0.155 -> 0.16; plain .2f gives 0.15)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def si_tables():
    """Condensed tables for Supplementary Information [S22] (written to results/RESULTS_OOD_SI.md)."""
    L = ["# Supplementary tables for [S22] (generated by experiments/make_ood.py; do not edit by hand)", "",
         "**Table S22a. Held-out groups: coverage among the selected units.** Pooled over groups, maps and re-splits; "
         "worst group and number of groups below 0.85 in parentheses; +∞: share of infinite JOMI intervals. "
         "R0: map trained with 30% of the held-out group; R1: map from other groups; both calibrated on a random half "
         "of the rest of the group. R2: the R1 map calibrated on as many units from other groups that did not train it. "
         "‡ GeoCP in R2 has almost no kernel mass near the test units.", "",
         "| Data, rule | Arm | JOMI | +∞ | Normalized CP | Split CP | GeoCP | CQR | QRF |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, lab, rules in SI_DESIGNS:
        df = load(tag)
        if df is None:
            continue
        for rule in rules:
            T = held_table(df, rule).set_index(["method", "arm"])
            for arm in ("R0", "R1", "R2"):
                c = lambda m: f"{T.loc[(m, arm), 'cov']:.3f} ({T.loc[(m, arm), 'worst']:.2f}; {T.loc[(m, arm), 'below']})"  # noqa: E731
                L.append(f"| {lab}, {RULE[rule]} ({T.loc[('JOMI stud', arm), 'groups']} groups) | {arm} | {c('JOMI stud')} | "
                         f"{T.loc[('JOMI stud', arm), 'inf']:.2f} | {c('SCP stud')} | {T.loc[('SCP |res|', arm), 'cov']:.3f} | "
                         f"{T.loc[('GeoCP', arm), 'cov']:.3f}{' ‡' if arm == 'R2' else ''} | {T.loc[('CQR', arm), 'cov']:.3f} | "
                         f"{T.loc[('QRF', arm), 'cov']:.3f} |")
    L += ["", "**Table S22b. Paired change in JOMI's coverage, R2 − R1 (DHS rows: pre-specified primary endpoint; FIA rows: "
          "secondary, with only 7–9 states).** Per group, the per-split change averaged over maps and re-splits; mean over "
          "groups; 95% t interval over groups. Finite intervals only: per group, pooled coverage among finite intervals in R2 "
          "minus R1, groups with no finite interval in one arm omitted (not paired by split).", "",
          "| Data, rule | Groups | R2 − R1 | Finite intervals only | Groups with R2 higher | R1 − R0 |", "|---|---|---|---|---|---|"]
    for tag, lab, rules in SI_DESIGNS:
        df = load(tag)
        if df is None:
            continue
        for rule in rules:
            m, ci, G, pos = paired(df, rule); mf, cf, Gf = paired_fin(df, rule); m0, c0, _, _ = paired(df, rule, "R0", "R1")
            L.append(f"| {lab}, {RULE[rule]} | {G} | {m:+.3f} ± {ci:.3f} | {mf:+.3f} ± {cf:.3f} ({Gf} groups) | {pos} of {G} | "
                     f"{m0:+.3f} ± {c0:.3f} |")
    B = pd.read_csv(os.path.join(RES, "ood_loco_poverty_bounds.csv"))
    rs = B[B.bound == "reference-set CLT"]; inf = rs[rs.nR < 2]; arms_inf = sorted(inf.arm.unique())
    g_top = inf.group.value_counts(); name = dict(enumerate(D.dhs_countries()))
    top = D.DHS[name[int(g_top.index[0])]][1] if len(g_top) else ""
    fin2 = rs[(rs.arm == "R2") & (rs.nR >= 2)].holds.mean()
    L += ["", "**Table S22c. DHS, poorest 25.2%: 90% upper bounds on the selected clusters' mean wealth** "
          "(share of the 23 x 5 x 40 surveys in which the bound held; per-country range). An infinite reference-set bound "
          f"(fewer than two reference clusters) counts as holding; this happens only in {', '.join(arms_inf)}, in "
          f"{len(inf)} of {int((rs.arm == arms_inf[0]).sum()) if arms_inf else 0:,} surveys ({int(g_top.iloc[0]) if len(g_top) else 0} "
          f"in {top}); over finite bounds the R2 rate is {fin2:.3f}.", "",
          "| Arm | Survey-mean correction: held (countries < 0.80; range) | Reference-set bound: held (countries < 0.80; range) | "
          "Mean residual, selected / calibration |", "|---|---|---|---|"]
    for arm, g in B.groupby("arm"):
        cell = []
        for bn in ("calibration-mean correction", "reference-set CLT"):
            x = g[g.bound == bn]; c = x.groupby("group").holds.mean()
            cell.append(f"{x.holds.mean():.3f} ({int((c < 0.8).sum())}; {r2(c.min())}–{r2(c.max())})")
        x = g[g.bound == "reference-set CLT"]
        L.append(f"| {arm} | {cell[0]} | {cell[1]} | {x.bias_sel.mean():+.3f} / {x.bias_cal.mean():+.3f} |")
    L += ["", "**Table S22d. Local survey size (R1 map, JOMI, first 20 re-splits; groups that have every size).**", "",
          "| Data, rule | Groups | Calibration units | Coverage | Finite only | +∞ |", "|---|---|---|---|---|---|"]
    for tag, lab, rules in SI_DESIGNS[:2]:
        S_ = survey(load(tag))
        for rule in rules:
            for _, r in S_[S_.rule == rule].iterrows():
                n = "all local units (35% of the group)" if r.n.startswith("all") else r.n
                L.append(f"| {lab}, {RULE[rule]} | {r.groups} | {n} | {r['cov']:.3f} | {r.cov_fin:.3f} | {r.inf:.2f} |")
    L += ["", "**Table S22e. Block-randomized splits against random splits through the same pipeline** "
          "(coverage among the selected units over 100 splits, blocks vs random split; standard errors over splits are at "
          "most 0.03, and at most 0.010 for JOMI and SC-GeoCCP (results/RESULTS_OOD.md); last column: +∞ share of JOMI "
          "under blocks).", "",
          "| Blocks, rule | Split CP | Normalized CP | GeoCP | QRF | JOMI | SC-GeoCCP | JOMI +∞ |", "|---|---|---|---|---|---|---|---|"]
    for name, blk, title in BLOCK:
        a, r = load(f"block_{name}_{blk}"), load(f"block_{name}_random")
        if a is None:
            continue
        for rule in [x for x in a.rule.unique() if x != "all"]:
            cells = []
            for m in ("SCP |res|", "SCP stud", "GeoCP", "QRF", "JOMI stud", "SC-GeoCCP"):
                g = a[(a.rule == rule) & (a.method == m)]
                if not len(g):
                    cells.append("—"); continue
                rr = r[(r.rule == rule) & (r.method == m)] if r is not None else g.iloc[:0]
                rand = f"{pool(rr)['cov']:.3f}" if len(rr) else "not run"
                cells.append(f"{pool(g)['cov']:.3f} vs {rand}")
            inf = pool(a[(a.rule == rule) & (a.method == "JOMI stud")]).inf
            L.append(f"| {title}, {RULE.get(rule, rule)} | " + " | ".join(cells) + f" | {inf:.2f} |")
    open(os.path.join(RES, "RESULTS_OOD_SI.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    L = ["# Out-of-design evaluation (2026-10-29)", "",
         "Design: `paper/OOD_DESIGN.md` (revised version); runs: `experiments/run_ood.py`; tables: `experiments/make_ood.py`.", ""]
    parts = []
    for tag, title in HELD:
        l, t = held_section(tag, title); L += l
        if t is not None:
            parts.append(t)
    l, t = block_section(); L += l; parts.append(t)
    pd.concat(parts, ignore_index=True).to_csv(os.path.join(RES, "ood_summary.csv"), index=False, float_format="%.4f")
    open(os.path.join(RES, "RESULTS_OOD.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    si_tables()
    print("\n".join(L))
