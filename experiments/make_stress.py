"""Tables, decisions and scorecard for the stress test of paper/STRESS_TEST_DESIGN.md (written and frozen before the run).
    ~/miniforge3/envs/favela/bin/python experiments/make_stress.py
reads results/st_twomap_fia.csv, results/st_twomap_fia_post2000.csv (sensitivity) and results/st_signal_<ds>.csv
(experiments/run_stress.py); writes results/RESULTS_STRESS.md (Chinese: decisions by sections 5.1-5.3, scorecard of
section 4, the filled templates of rule 7), results/RESULTS_STRESS_SI.md (Tables S25a-e), results/stress_summary.csv and
results/stress_paired.csv (mean and s.e. of the paired split differences, JOMI minus ten bins of mu-hat, one row per
rule and share or setting).
Decisions use unrounded pooled point estimates (sum n_cov / sum n_sel, +inf counted as covering). The bootstrap intervals
(2,000 resamples of the 100 splits) and the paired split differences are descriptive only.
After the run (execution record of the design): the scope of jinf was corrected to all 11 settings and {r} printed with
two decimals; Tables S25d-e were added; Tables S25a-c gained descriptive columns, endpoint marks, the predictions and
English labels (kappa for the design's tau, which Methods uses for 1 - alpha); the scorecard was written out in English;
no decision changed.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "figures")); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
from make_mondrian_noise import r2, r3                                # noqa: E402

RES = os.environ.get("STRESS_OUT", os.path.join(ROOT, "results"))
METH = [("SCP |res|", "Split CP"), ("SCP stud", "Normalized CP"), ("Mondrian CP", "Mondrian CP, 10 bins of μ̂"),
        ("Mondrian CP, 20 bins", "Mondrian CP, 20 bins of μ̂"), ("Mondrian CP, rule bins 10", "Mondrian CP, 10 bins of the rule's score"),
        ("Mondrian CP, rule bins 20", "Mondrian CP, 20 bins of the rule's score"), ("JOMI stud", "JOMI"), ("JOMI |res|", "JOMI (additive)")]
NV, NV20, I10, I20, JS, JA = "Mondrian CP", "Mondrian CP, 20 bins", "Mondrian CP, rule bins 10", "Mondrian CP, rule bins 20", "JOMI stud", "JOMI |res|"
RULES_A = [("site", "Site map (own prediction)"), ("site+height", "Site + canopy-height map"), ("full", "Full-inventory map")]
RULES_B = [("policy", "Map's prediction (5% rule)"), ("signal tau=2", "Map + field signal, τ = 2"), ("signal tau=1", "Map + field signal, τ = 1"),
           ("signal tau=0.5", "Map + field signal, τ = 0.5"), ("control", "Control: map + ranking noise")]
EN_B = {rn: rl.replace("field signal", "field measurement").replace("τ", "κ") for rn, rl in RULES_B}   # English SI labels
LAB = {nm: l for nm, l, _ in S.DATASETS}; KIND = {nm: k for nm, _, k in S.DATASETS}
MEAS = ["poverty", "poverty_lab", "fia", "camels", "voting", "seattle"]
GAUSS_A = {("site+height", 0.05): 0.79, ("site+height", 0.01): 0.70, ("full", 0.05): 0.61, ("full", 0.01): 0.38}
NB = 2000


def pooled(df, keys):
    g = df.groupby(keys)
    sel, cov, inf = g.n_sel.sum(), g.n_cov.sum(), g.n_inf.sum()
    out = pd.DataFrame({"cov": cov / sel, "inf": inf / sel, "cov_fin": (cov - inf) / (sel - inf).replace(0, np.nan),
                        "up": g.up_miss.sum() / sel, "lo": g.lo_miss.sum() / sel, "width": g.hw_mean_fin.mean(),
                        "resid_sd": g.resid_mean_sd.mean(), "overlap": g.overlap_own.mean(), "n_rep": g.rep.nunique()})
    for c in ("extreme_bin_share", "extreme_bin_cal", "min_bin_cal_sel", "R_mean", "R_min", "cor_signal_resid", "err_ratio",
              "r2_selector", "r2_interval_map", "resid_mean"):
        if c in df.columns:
            out[c] = g[c].mean() if c not in ("min_bin_cal_sel", "R_min") else g[c].min()
    if "extreme_bin_share" in df.columns:
        out["extreme_bin_share_min"] = g["extreme_bin_share"].min()
    return out.reset_index()


def boot(df, keys, seed=20261030):
    """Descriptive 95% interval of the pooled coverage, resampling whole splits."""
    rng = np.random.default_rng(seed); out = []
    for kv, x in df.groupby(keys):
        x = x.sort_values("rep"); c, s = x.n_cov.to_numpy(float), x.n_sel.to_numpy(float)
        idx = rng.integers(0, len(x), (NB, len(x)))
        b = c[idx].sum(1) / s[idx].sum(1)
        out.append(dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)), lo95=np.quantile(b, 0.025), hi95=np.quantile(b, 0.975)))
    return pd.DataFrame(out)


def paired(df, keys, a=JS, b=NV):
    x = df.assign(c=df.n_cov / df.n_sel).pivot_table(index=keys + ["rep"], columns="method", values="c").reset_index()
    x["d"] = x[a] - x[b]; g = x.groupby(keys).d
    return pd.DataFrame({"diff": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.count())}).reset_index()


def wratio(df, keys, a, b=JS):
    """Mean finite half-width of method a relative to method b over the splits in which both were finite."""
    x = df.pivot_table(index=keys + ["rep"], columns="method", values="hw_mean_fin")[[a, b]].dropna()
    g = x.groupby(level=keys)
    return g[a].mean() / g[b].mean()


def boot_b5(B, nb=100000, seed=2026):
    """Chosen after the run, descriptive only: the share of resamples (each setting's 100 splits resampled independently)
    in which ten bins of mu-hat fall below 0.85 in k = 0..5 of the 5 measured data sources at tau = 0.5."""
    rng = np.random.default_rng(seed); below = {}
    for nm in ("poverty", "poverty_lab", "fia", "camels", "voting", "seattle"):
        x = B[(B.dataset == nm) & (B.rule == "signal tau=0.5") & (B.method == NV)].sort_values("rep")
        c, s = x.n_cov.to_numpy(float), x.n_sel.to_numpy(float)
        idx = rng.integers(0, len(x), (nb, len(x)))
        below[nm] = c[idx].sum(1) / s[idx].sum(1) < 0.85
    b5 = (below["poverty"] | below["poverty_lab"]).astype(int) + sum(below[nm].astype(int) for nm in ("fia", "camels", "voting", "seattle"))
    return np.bincount(b5, minlength=6) / nb


def cell(P, **kw):
    m = np.ones(len(P), bool)
    for k, v in kw.items():
        m &= (P[k] == v).to_numpy()
    assert m.sum() == 1, kw
    return P[m].iloc[0]


def andlist(a):
    return a[0] if len(a) == 1 else ", ".join(a[:-1]) + " and " + a[-1]


def fmt(v, inf):
    return r3(v) + (f" ({r2(inf)})" if inf > 0 else "")


def z2(x):
    return "—" if not np.isfinite(x) else r2(0.0 if abs(x) < 0.005 else x)


PRED_B = {nm: "0.81–0.82 (≥ 0.90)" for nm in ("health_obesity", "poverty", "poverty_lab", "voting")}   # design, section 4
PRED_B.update({"seattle": "0.85 (≈ 0.4)", "fia": "0.86 (≈ 0.2)", "camels": "0.87 (≈ 0.2)", "health_bphigh": "0.91 (≈ 0)"})
PRED_B.update({nm: "0.87–0.89 (≤ 0.05)" for nm in ("health_stroke", "health_diabetes", "era5")})
PRIMARY_A = {(0.05, "site+height", NV), (0.05, "site", NV), (0.05, "site+height", JS)}
CARD_EN = {   # design, section 4
    "P-A1": "ten bins of μ̂ below 0.85 in the four cells selected by the site + canopy-height and full-inventory maps at 5% and 1%.",
    "P-A2": "JOMI (normalized and additive) at least 0.87 in all 12 cells.",
    "P-A3": "20 bins of μ̂ below 0.85 in those four cells and within 0.03 of ten bins (also with finite intervals only).",
    "P-A4": "ten bins of μ̂ ordered own rule > site + canopy height > full inventory at both shares.",
    "P-A5": "in those four cells, ten bins of μ̂ < ten bins of the rule's score < JOMI.",
    "P-A6": "at 5%, 20 bins of the rule's score within 0.02 of JOMI (site + canopy height; full inventory).",
    "P-A7": "at 1%, 20 bins of the rule's score more than 0.03 below JOMI (site + canopy height; full inventory), because "
            "their top bin is 5% wide.",
    "P-A8": "ten bins of μ̂ within 0.10 of the finite-sample Gaussian values 0.79, 0.70, 0.61 and 0.38.",
    "P-B1": "at κ = 0.5, ten bins of μ̂ below 0.85 in four or five of the 11 settings (probability about 0.75; 90% range three to six).",
    "P-B2": "at κ = 1, at most one setting below 0.85; at κ = 2 and under the control, none.",
    "P-B3": "JOMI below 0.85 in no setting at any κ or under the control.",
    "P-B4": "in at least 9 settings, ten bins of μ̂ cover strictly less at κ = 0.5 than at κ = 1, and less at κ = 1 than at κ = 2.",
    "P-B5": "in at least 10 settings, ten bins of μ̂ within 0.02 between the control and the map's own rule.",
    "P-B6": "ten and 20 bins of the rule's score below 0.85 in no setting at any κ.",
}


def table_a(P, tag, r2i, L, T, caption, marks=frozenset()):
    hdr = ("| Share | Selecting map (held-out R²) | " + " | ".join(m for _, m in METH) +
           " | Misses above / below (10 bins of μ̂) | Mean residual (error spreads) | Overlap with own selection |")
    T += [caption, "", hdr, "|---|---|" + "---|" * (len(METH) + 3)]
    L += [f"### {tag}", "", hdr, "|---|---|" + "---|" * (len(METH) + 3)]
    for frac in (0.05, 0.01):
        for rn, rl in RULES_A:
            x = P[(P.frac == frac) & (P.rule == rn)].set_index("method")
            row = (f"| {frac:.0%} | {rl} ({r2(x.loc[JS, 'r2_selector'])}) | " +
                   " | ".join(fmt(x.loc[mk, "cov"], x.loc[mk, "inf"]) + ("§" if (frac, rn, mk) in marks else "") for mk, _ in METH) +
                   f" | {r3(x.loc[NV, 'up'])} / {r3(x.loc[NV, 'lo'])} | {r2(x.loc[JS, 'resid_sd'])} | {r2(x.loc[JS, 'overlap'])} |")
            T.append(row); L.append(row)
    T.append(""); L.append("")


def main():
    L = ["# 压力测试：选择规则用到地图之外的信息（2026-10-30）", "",
         "设计（跑前写定并冻结）：`paper/STRESS_TEST_DESIGN.md`。运行：`experiments/run_stress.py`（复现断言通过）。表与判定：`experiments/make_stress.py`。"
         "覆盖为 100 次划分合并的不取整点估计，+∞ 计为覆盖；覆盖值后的括号内为 +∞ 比例（计数表中括号内为最低覆盖）。"
         "区间与配对差只作描述。‡：JOMI 或知情二十箱 +∞ 比例超过 10%，其覆盖不作为成立的证据。", ""]
    T, rows, card, audit = [], [], [], []
    fill = {}
    # ---------------- arm A
    A = pd.read_csv(os.path.join(RES, "st_twomap_fia.csv"))
    PA = pooled(A, ["frac", "rule", "method"]); BA = boot(A, ["frac", "rule", "method"]); DA = paired(A, ["frac", "rule"])
    PA = PA.merge(BA, on=["frac", "rule", "method"])
    r2i = A.r2_interval_map.mean()
    L += ["## 臂 A　两张地图（FIA；区间来自 site 地图）", ""]
    table_a(PA, f"FIA（site 地图留出 R² {r2(r2i)}）", r2i, L, T,
            "Table S25a. Two maps (FIA forest biomass): intervals from the site-attribute map; the rule selects the highest 5% "
            "or 1% of the test plots by the site map itself, by the site + canopy-height map or by the full-inventory map. "
            "Coverage among the selected plots (pooled over 100 splits; no interval was infinite), the share of selected plots "
            "missed above and below the interval with ten bins of μ̂, the selected plots' mean site-map residual in error "
            "spreads ς = √(π/2)·σ̂, and the overlap with the site map's own selection. Held-out R² values are means over the "
            "100 splits on the test plots. § Primary endpoints: ten bins of "
            "μ̂ when the site + canopy-height map selects 5% (N) and when the site map selects its own 5% (N₀), and JOMI when "
            "the site + canopy-height map selects 5% (J). Secondary endpoints: the full-inventory and 1% rows, 20 bins of μ̂, "
            "bins of the rule's score, and normalized and split CP; Table S25a (continued) is a sensitivity analysis. "
            "Expectations written before the run (finite-sample Gaussian model; probability of falling below 0.85 in "
            "parentheses where the design gave one), in the order site + canopy height at 5% and 1%, full inventory at 5% and 1%: ten bins of μ̂ 0.79, "
            "0.70, 0.61 and 0.38 (1.00 each); 20 bins of μ̂ 0.80, 0.71, 0.61 and 0.39; ten bins of the rule's score 0.88 "
            "(0.00), 0.81 (0.90), 0.85 (0.61) and 0.68 (1.00); 20 bins of the rule's score 0.90, 0.84 (0.58), 0.90 and 0.78 "
            "(0.99); JOMI 0.90–0.92; under the site map's own rule about 0.90 throughout (ten bins of μ̂: 0.00 at 5% and 0.03 "
            "at 1%). Descriptive split-bootstrap intervals for every cell, and paired split differences (JOMI minus ten bins of "
            "μ̂) for every row, are in the source data.",
            marks=PRIMARY_A)
    dA = paired(A, ["frac", "rule"])
    xa = {k: cell(PA, frac=k[0], rule=k[1], method=k[2]) for k in PRIMARY_A}
    da = dA[(dA.frac == 0.05) & (dA.rule == "site+height")].iloc[0]
    T += [f"Primary endpoint A (descriptive 95% intervals from 2,000 resamples of the 100 splits): N = "
          f"{r3(xa[(0.05, 'site+height', NV)]['cov'])} [{r3(xa[(0.05, 'site+height', NV)].lo95)}, {r3(xa[(0.05, 'site+height', NV)].hi95)}], "
          f"N₀ = {r3(xa[(0.05, 'site', NV)]['cov'])} [{r3(xa[(0.05, 'site', NV)].lo95)}, {r3(xa[(0.05, 'site', NV)].hi95)}], "
          f"J = {r3(xa[(0.05, 'site+height', JS)]['cov'])} [{r3(xa[(0.05, 'site+height', JS)].lo95)}, {r3(xa[(0.05, 'site+height', JS)].hi95)}]; "
          f"paired split difference J − N {da['diff']:+.3f} (s.e. {da['se']:.3f}).", ""]
    WA = {m: wratio(A, ["frac", "rule"], m) for m in (NV, I10, I20)}
    TE = ["Table S25e. Two maps (FIA): diagnostics, tabulated after the run from the stored per-split results. Where three "
          "values are given, they are for ten bins of μ̂, ten bins of the rule's score and 20 bins of the rule's score. Mean "
          "finite half-width relative to JOMI's; JOMI's reference-set size (mean over splits; smallest in any split); the "
          "extreme (top) bin's calibration units and calibration share (means over splits; smallest share in any split in "
          "brackets); the fewest calibration units in any bin holding a selected plot; the share of selected plots missed above "
          "and below the interval by JOMI and by 20 bins of the rule's score; and the selected plots' mean site-map residual "
          "d·(Y − μ̂) in units of the plot biomass index.", "",
          "| Share | Selecting map | Width (× JOMI) | JOMI reference set | Extreme bin: calibration units | Extreme bin: calibration share [smallest] | Fewest in a selected bin | Misses above / below: JOMI; 20 bins of the rule's score | Mean residual (index units) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for frac in (0.05, 0.01):
        for rn, rl in RULES_A:
            x = PA[(PA.frac == frac) & (PA.rule == rn)].set_index("method")
            bm = (NV, I10, I20)
            wid = " / ".join(r2(WA[m].loc[(frac, rn)]) for m in bm)
            ecal = " / ".join(f"{x.loc[m, 'extreme_bin_cal']:.0f}" for m in bm)
            esh = " / ".join(f"{r3(x.loc[m, 'extreme_bin_share'])} [{r3(x.loc[m, 'extreme_bin_share_min'])}]" for m in bm)
            fewest = " / ".join(str(int(x.loc[m, "min_bin_cal_sel"])) for m in bm)
            TE.append(f"| {frac:.0%} | {rl} | {wid} | {x.loc[JS, 'R_mean']:.1f} ({int(x.loc[JS, 'R_min'])}) | {ecal} | {esh} | {fewest} | "
                     f"{r3(x.loc[JS, 'up'])} / {r3(x.loc[JS, 'lo'])}; {r3(x.loc[I20, 'up'])} / {r3(x.loc[I20, 'lo'])} | {x.loc[JS, 'resid_mean']:.1f} |")
    TE += [""]                                                              # printed after Table S25d
    f2 = os.path.join(RES, "st_twomap_fia_post2000.csv")
    if os.path.exists(f2):
        A2 = pd.read_csv(f2); PA2 = pooled(A2, ["frac", "rule", "method"]).merge(boot(A2, ["frac", "rule", "method"]), on=["frac", "rule", "method"])
        table_a(PA2, "敏感性：fia_post2000（只报告，不判定）", A2.r2_interval_map.mean(), L, T,
                "Table S25a (continued). The same on the FIA plots inventoried in 2000 or later (sensitivity analysis; not used for any decision).")
        for _, rr in PA2.iterrows():
            rows.append(dict(arm="A_post2000", dataset="fia_post2000", frac=rr.frac, rule=rr.rule, method=rr.method, cov=rr["cov"], inf=rr["inf"],
                             cov_fin=rr.cov_fin, up=rr.up, lo=rr.lo, resid_sd=rr.resid_sd, overlap=rr.overlap, lo95=rr.lo95, hi95=rr.hi95,
                             extreme_bin_share=rr.get("extreme_bin_share", np.nan)))
    for _, rr in PA.iterrows():
        rows.append(dict(arm="A", dataset="fia", frac=rr.frac, rule=rr.rule, method=rr.method, cov=rr["cov"], inf=rr["inf"],
                         cov_fin=rr.cov_fin, up=rr.up, lo=rr.lo, resid_sd=rr.resid_sd, overlap=rr.overlap, lo95=rr.lo95, hi95=rr.hi95,
                         extreme_bin_share=rr.get("extreme_bin_share", np.nan)))
    c = lambda frac, rn, mk: cell(PA, frac=frac, rule=rn, method=mk)            # noqa: E731
    N, N0, J = c(0.05, "site+height", NV)["cov"], c(0.05, "site", NV)["cov"], c(0.05, "site+height", JS)["cov"]
    jom_a = PA[PA.method.isin([JS, JA])]
    if f2 and os.path.exists(f2):
        jom_a = pd.concat([jom_a, PA2[PA2.method.isin([JS, JA])]])
    if (jom_a["cov"] < 0.87).any():
        audit.append("臂 A：JOMI 有格低于 0.87：" + ", ".join(f"{r.rule} {r.frac:.0%} {r.method} {r3(r['cov'])}" for _, r in jom_a[jom_a["cov"] < 0.87].iterrows()))
    if J < 0.85:
        outA = "A4"
    elif N < 0.85:
        outA = "A1" if N0 >= 0.85 else "A3"
    else:
        outA = "A2"
    d = DA[(DA.frac == 0.05) & (DA.rule == "site+height")].iloc[0]
    xs = c(0.05, "site+height", NV)
    side = "above" if xs.up >= xs.lo else "below"
    fill.update(N=r2(N), N0=r2(N0), J=r2(J), R2h=r2(c(0.05, "site+height", JS).r2_selector), R2s=r2(r2i),
                r=r2(c(0.05, "site+height", JS).resid_sd), side=side,          # two decimals, as in Table S25a (after the run)
                I10=r2(c(0.05, "site+height", I10)["cov"]), I20=r2(c(0.05, "site+height", I20)["cov"]))
    L += [f"**主终点 A**：N（天真十箱，site+height 选 5%）= {r3(N)}［{r3(xs.lo95)}, {r3(xs.hi95)}］；N₀（site 自身选 5%）= {r3(N0)}；"
          f"J（JOMI）= {r3(J)}；逐划分配对差 J − N = {d['diff']:+.3f}（s.e. {d['se']:.3f}，只作描述）；"
          f"被选单元平均残差 {xs.resid_sd:.2f} 个误差 sd；失效在上侧 {r3(xs.up)}、下侧 {r3(xs.lo)}。"
          f"**判定：{outA}**" + ("（核查触发未解除前不生效）" if audit else "") + "。", ""]
    if outA == "A2" and xs.resid_sd < 0.6:
        L += ["注：被选单元的平均残差低于 0.6 个误差 sd，规则携带的残差信息不足，不作为原则的反证（规则 2）。", ""]
    # scorecard A
    four = [(rn, f) for rn in ("site+height", "full") for f in (0.05, 0.01)]
    card.append(("P-A1", all(c(f, rn, NV)["cov"] < 0.85 for rn, f in four)))
    card.append(("P-A2", bool((PA[PA.method.isin([JS, JA])]["cov"] >= 0.87).all())))
    pa3 = all(c(f, rn, NV20)["cov"] < 0.85 and abs(c(f, rn, NV20)["cov"] - c(f, rn, NV)["cov"]) <= 0.03 and
              c(f, rn, NV20).cov_fin < 0.85 and abs(c(f, rn, NV20).cov_fin - c(f, rn, NV).cov_fin) <= 0.03 for rn, f in four)
    card.append(("P-A3", pa3))
    card.append(("P-A4", all(c(f, "site", NV)["cov"] > c(f, "site+height", NV)["cov"] > c(f, "full", NV)["cov"] for f in (0.05, 0.01))))
    card.append(("P-A5", all(c(f, rn, NV)["cov"] < c(f, rn, I10)["cov"] < c(f, rn, JS)["cov"] for rn, f in four)))
    card.append(("P-A6", all(abs(c(0.05, rn, I20)["cov"] - c(0.05, rn, JS)["cov"]) <= 0.02 for rn in ("site+height", "full"))))
    card.append(("P-A7", all(c(0.01, rn, JS)["cov"] - c(0.01, rn, I20)["cov"] > 0.03 for rn in ("site+height", "full"))))
    card.append(("P-A8（只作记录）", all(abs(c(f, rn, NV)["cov"] - GAUSS_A[(rn, f)]) <= 0.10 for rn, f in four)))
    # ---------------- arm B
    have = [nm for nm, _, _ in S.DATASETS if os.path.exists(os.path.join(RES, f"st_signal_{nm}.csv"))]
    B = pd.concat([pd.read_csv(os.path.join(RES, f"st_signal_{nm}.csv")) for nm in have], ignore_index=True)
    PB = pooled(B, ["dataset", "rule", "method"]).merge(boot(B, ["dataset", "rule", "method"]), on=["dataset", "rule", "method"])
    complete = len(have) == 11
    L += [f"## 臂 B　地图加野外信号（{len(have)} 个设定，5%，政策方向）", ""]
    if not complete:
        L += [f"**设定未全部完成（{len(have)}/11），主终点 B 不判定（规则 6）。**", ""]
    jom_b = PB[PB.method.isin([JS, JA])]
    if (jom_b["cov"] < 0.87).any():
        audit.append("臂 B：JOMI 有格低于 0.87：" + ", ".join(f"{r.dataset} {r.rule} {r.method} {r3(r['cov'])}" for _, r in jom_b[jom_b["cov"] < 0.87].iterrows()))
    cb = lambda nm, rn, mk: cell(PB, dataset=nm, rule=rn, method=mk)            # noqa: E731
    cnt = lambda rn, mk, names: sum(cb(nm, rn, mk)["cov"] < 0.85 for nm in names if nm in have)   # noqa: E731
    b11, b6 = cnt("signal tau=0.5", NV, have), cnt("signal tau=0.5", NV, MEAS)
    dhs = int(any(cb(nm, "signal tau=0.5", NV)["cov"] < 0.85 for nm in ("poverty", "poverty_lab") if nm in have))
    b5 = dhs + cnt("signal tau=0.5", NV, ["fia", "camels", "voting", "seattle"])
    j11, c11 = cnt("signal tau=0.5", JS, have), cnt("control", NV, have)
    counted = [nm for nm in have if cb(nm, "signal tau=0.5", NV)["cov"] < 0.85]
    # rule 3 / template B: the largest JOMI +inf share over the settings in which the reference set was counted (all 11);
    # corrected after the run (the first version took it over the settings where fixed bins failed only; see the record)
    jinf = max([cb(nm, "signal tau=0.5", JS)["inf"] for nm in have], default=0.0)
    jinf_at = max(have, key=lambda nm: cb(nm, "signal tau=0.5", JS)["inf"])
    jinf_counted = max([cb(nm, "signal tau=0.5", JS)["inf"] for nm in counted], default=0.0)
    if b5 >= 2 and j11 == 0:
        outB = "B 阳性" if c11 == 0 else "B 对照失效"
    elif b5 <= 1:
        outB = "B 阴性"
    else:
        outB = "JOMI 低于 0.85（按核查触发处理）"
    fill.update(b5=str(b5), b11=str(b11), c11=str(c11), jinf=r2(jinf))
    L += [f"**主终点 B**（τ = 0.5，天真十箱）：低于 0.85 的设定 11 个中 {b11} 个，6 个实测结局设定中 {b6} 个，5 个实测数据源中 {b5} 个；"
          f"JOMI 11 个中 {j11} 个；对照下天真十箱 {c11} 个。**判定：{outB}**" +
          ("（核查触发未解除前不生效）" if audit else "") + ("（未完成，不判定）" if not complete else "") + "。"
          f"JOMI 的 +∞ 比例：{len(have)} 个设定中最大 {r2(jinf)}（{LAB[jinf_at]}{'，‡' if jinf > 0.10 else ''}），在天真十箱失效的设定中最大 {r2(jinf_counted)}。", ""]
    # scorecard B
    nt = lambda rn, mk=NV: cnt(rn, mk, have)                                    # noqa: E731
    card.append(("P-B1", nt("signal tau=0.5") in (4, 5)))
    card.append(("P-B2", nt("signal tau=1") in (0, 1) and nt("signal tau=2") == 0 and nt("control") == 0))
    card.append(("P-B3", all(nt(rn, JS) == 0 for rn in ("signal tau=0.5", "signal tau=1", "signal tau=2", "control"))))
    mono = sum(cb(nm, "signal tau=0.5", NV)["cov"] < cb(nm, "signal tau=1", NV)["cov"] < cb(nm, "signal tau=2", NV)["cov"] for nm in have)
    card.append(("P-B4", mono >= 9))
    near = sum(abs(cb(nm, "control", NV)["cov"] - cb(nm, "policy", NV)["cov"]) <= 0.02 for nm in have)
    card.append(("P-B5", near >= 10))
    card.append(("P-B6", all(nt(rn, mk) == 0 for rn in ("signal tau=0.5", "signal tau=1", "signal tau=2") for mk in (I10, I20))))
    # tables B
    hdr = "| Rule | Median corr(Z − μ̂, Y − μ̂) | " + " | ".join(m for _, m in METH) + " |"
    T += ["Table S25b. Map plus field measurement (5% in the policy direction, 11 settings, 100 splits): settings with "
          "coverage among the selected units below 0.85 (lowest coverage in parentheses), settings above 0.95, and the largest "
          "share of infinite intervals over the settings (infinite intervals occurred only in CAMELS runoff, whose coverage is "
          "not evidence of holding where that share exceeds 0.10). The rule ranks x_κ = d(κ²μ̂ + Z_κ)/(κ² + 1), with d the "
          "policy direction, Z_κ = Y + κςε, ς = √(π/2)·σ̂ the map's error spread and ε ~ N(0, 1), so nominal κ is the field "
          "measurement's error relative to the map's; the control ranks d(μ̂ + Z₀)/2, the κ = 1 combination, with "
          "Z₀ = μ̂ + ςε. The primary endpoint is ten bins of μ̂ and JOMI at κ = 0.5 (Table S25d); the decision fixed before "
          "the run also required ten bins of μ̂ under the control to fall below 0.85 in no setting. κ = 1 and 2, the "
          "control's other cells and the other methods are secondary endpoints. Expectations written before the run: P-B1 to P-B6 in the scorecard after Table S25e.", "",
          hdr, "|---|---|" + "---|" * len(METH)]
    L += ["### 各规则下低于 0.85 的设定数（最低覆盖）；高于 0.95 的设定数；最大 +∞ 比例", "", hdr, "|---|---|" + "---|" * len(METH)]
    for rn, rl in RULES_B:
        y_ = PB[PB.rule == rn]; cm = y_[y_.method == JS].cor_signal_resid.median() if "cor_signal_resid" in y_ else np.nan
        cells = []
        for mk, _ in METH:
            v = y_[y_.method == mk]["cov"]; vi = y_[y_.method == mk]["inf"]
            cells.append(f"{int((v < 0.85).sum())} ({r3(v.min())}); {int((v > 0.95).sum())}; {z2(vi.max())}")
        row = f"| {rl} | {z2(cm)} | " + " | ".join(cells) + " |"
        T.append(row.replace(f"| {rl} |", f"| {EN_B[rn]} |", 1)); L.append(row)
    T.append(""); L.append("")
    cols = [("SCP stud", "Normalized CP"), (NV, "Mondrian CP, 10 bins of μ̂"), (I10, "Mondrian CP, 10 bins of the rule's score"),
            (I20, "Mondrian CP, 20 bins of the rule's score"), (JS, "JOMI")]
    T += ["Table S25c. Map plus field measurement by setting: coverage among the selected units (share of infinite intervals "
          "in parentheses where nonzero; ‡ on the setting: JOMI or 20 bins of the rule's score with more than 10% infinite "
          "intervals under that rule, whose coverage is not evidence of holding), corr(Z − μ̂, Y − μ̂) with the implied κ_eff = (1/corr² − 1)^{1/2} (¶ would "
          "mark κ_eff outside 0.35–0.65 at nominal κ = 0.5; none is marked; not defined for the map's own rule or the control) "
          "and sd(Z − Y)/sd(Y − μ̂) on the test units. The primary endpoint (κ = 0.5) is given to four decimals in Table S25d. "
          "* PLACES model output (the field measurement is simulated from the modelled value); † reanalysis (likewise).", "",
          "| Setting | Rule | corr (κ_eff); sd ratio | " + " | ".join(cn for _, cn in cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    L += ["### 逐设定（τ = 0.5 及对照）", "", "| 设定 | τ = 0.5：天真十箱 / JOMI | τ_eff | 对照：天真十箱 | policy：天真十箱 |", "|---|---|---|---|---|"]
    for nm, lab, kind in S.DATASETS:
        if nm not in have:
            continue
        lab2 = lab + ("†" if kind == "reanalysis" else "")
        for rn, rl in RULES_B:
            x = PB[(PB.dataset == nm) & (PB.rule == rn)].set_index("method")
            corr = x.loc[JS].get("cor_signal_resid", np.nan)
            te = np.sqrt(1 / corr ** 2 - 1) if np.isfinite(corr) and corr > 0 and rn.startswith("signal") else np.nan
            flag = "¶" if rn == "signal tau=0.5" and np.isfinite(te) and not (0.35 <= te <= 0.65) else ""
            dag = "‡" if (x.loc[JS, "inf"] > 0.10 or x.loc[I20, "inf"] > 0.10) else ""
            er = x.loc[JS].get("err_ratio", np.nan)
            cc = "—" if not np.isfinite(corr) else f"{z2(corr)} ({z2(te)}){flag}; {z2(er)}"
            T.append(f"| {lab2}{dag} | {EN_B[rn]} | {cc} | " + " | ".join(fmt(x.loc[mk, 'cov'], x.loc[mk, 'inf']) for mk, _ in cols) + " |")
            for mk in (NV, I10, I20, JS, JA, "SCP stud", "SCP |res|", NV20):
                rows.append(dict(arm="B", dataset=nm, frac=0.05, rule=rn, method=mk, cov=x.loc[mk, "cov"], inf=x.loc[mk, "inf"],
                                 up=x.loc[mk, "up"], lo=x.loc[mk, "lo"], resid_sd=x.loc[mk, "resid_sd"], lo95=x.loc[mk, "lo95"],
                                 hi95=x.loc[mk, "hi95"], cor_signal_resid=corr, tau_eff=te))
        x5 = PB[(PB.dataset == nm) & (PB.rule == "signal tau=0.5")].set_index("method")
        corr5 = x5.loc[JS].get("cor_signal_resid", np.nan); te5 = np.sqrt(1 / corr5 ** 2 - 1) if np.isfinite(corr5) else np.nan
        dag5 = "‡" if (x5.loc[JS, "inf"] > 0.10 or x5.loc[I20, "inf"] > 0.10) else ""
        L.append(f"| {lab2}{dag5} | {r3(x5.loc[NV, 'cov'])} / {fmt(x5.loc[JS, 'cov'], x5.loc[JS, 'inf'])} | {r2(te5)} | "
                 f"{r3(cb(nm, 'control', NV)['cov'])} | {r3(cb(nm, 'policy', NV)['cov'])} |")
    T.append(""); L.append("")
    DB = paired(B, ["dataset", "rule"]); WB = wratio(B[B.rule == "signal tau=0.5"], ["dataset"], NV)
    T += ["Table S25d. Primary endpoint B (κ = 0.5) by setting, tabulated after the run from the stored per-split results: "
          "coverage of ten bins of μ̂ predicted before the run (Gaussian model shifted to each setting's coverage under the "
          "map's own rule; probability of falling below 0.85 in parentheses) and observed; JOMI; the paired split difference "
          "(JOMI minus ten bins; mean and s.e.); whether the setting counts below 0.85 (decided on the unrounded pooled "
          "estimate; the 95% intervals from 2,000 resamples of the 100 splits are descriptive and were not used to re-judge "
          "values near 0.85); the share of selected units missed above and below the ten-bin interval; the selected units' "
          "mean residual d·(Y − μ̂) in error spreads ς = √(π/2)·σ̂; the overlap with the map's own 5% selection; and the mean "
          "finite half-width of ten bins of μ̂ relative to JOMI's, over the splits in which both were finite. Measured data "
          "sources: the two DHS maps count once. * PLACES model output; † reanalysis; ‡ more than 10% of the intervals of JOMI "
          "or of 20 bins of the rule's score infinite, so their coverage is not evidence of holding.", "",
          "| Setting | Predicted | Ten bins of μ̂ [95% interval] | JOMI [95% interval] (infinite share) | JOMI − ten bins (s.e.) | Below 0.85 | Misses above / below | Mean residual (error spreads) | Overlap | Width (× JOMI) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, lab, kind in S.DATASETS:
        if nm not in have:
            continue
        a = cb(nm, "signal tau=0.5", NV); j = cb(nm, "signal tau=0.5", JS)
        dd = DB[(DB.dataset == nm) & (DB.rule == "signal tau=0.5")].iloc[0]
        lab2 = lab + ("†" if kind == "reanalysis" else "") + ("‡" if (j.inf > 0.10 or cb(nm, "signal tau=0.5", I20).inf > 0.10) else "")
        T.append(f"| {lab2} | {PRED_B[nm]} | {a['cov']:.4f} [{r3(a.lo95)}, {r3(a.hi95)}] | {r3(j['cov'])} [{r3(j.lo95)}, {r3(j.hi95)}]"
                 f"{f' ({r2(j.inf)})' if j.inf > 0 else ''} | {dd['diff']:+.3f} ({dd['se']:.3f}) | {'yes' if a['cov'] < 0.85 else 'no'} | "
                 f"{r3(a.up)} / {r3(a.lo)} | {r2(j.resid_sd)} | {r2(j.overlap)} | {r2(WB.loc[nm])} |")
    T += ["", f"Counts at κ = 0.5 (ten bins of μ̂): {b11} of {len(have)} settings, {b6} of 6 measured settings, {b5} of 5 "
          f"measured data sources; JOMI {j11} of {len(have)}; control (ten bins of μ̂) {c11} of {len(have)}. JOMI's largest share "
          f"of infinite intervals: {r2(jinf)} over the {len(have)} settings ({LAB[jinf_at]}), {r2(jinf_counted)} in the settings "
          "where ten bins fell below 0.85.", ""]
    if complete:
        pk = boot_b5(B); ks = [k for k, p in enumerate(pk) if p > 0]; nb = 100000
        T += [f"Chosen after the run, descriptive only: resampling each setting's 100 splits independently {nb:,} times, ten "
              f"bins of μ̂ fell below 0.85 in {andlist([str(k) for k in ks])} of the 5 measured data sources in "
              f"{andlist([r2(pk[k]) for k in ks])} of the resamples" + (f" (in none in {r2(pk[0])})" if pk[0] > 0 else " (never in none)") +
              f"; Monte Carlo standard error at most {np.sqrt(0.25 / nb):.4f}.", ""]
    T += TE
    # scorecard and templates
    T += ["Scorecard of the predictions written before the run (unrounded pooled estimates; the predictions do not enter the "
          "decisions):", ""]
    T += [f"- {k.replace('（只作记录）', ' (record only)')}: {CARD_EN[k[:4]]} **{'Met' if v else 'Not met'}.**" for k, v in card] + [""]
    pd.concat([DA.assign(arm="A", dataset="fia"), paired(A2, ["frac", "rule"]).assign(arm="A_post2000", dataset="fia_post2000")
               if os.path.exists(f2) else None, DB.assign(arm="B", frac=0.05)], ignore_index=True)[
        ["arm", "dataset", "frac", "rule", "diff", "se"]].to_csv(os.path.join(RES, "stress_paired.csv"), index=False, float_format="%.5f")
    L += ["## 预测记分表", ""] + [f"- {k}：{'符合' if v else '不符合'}" for k, v in card] + [""]
    tpl = {
        "A1": ("When a map with canopy height chose the plots (held-out R² {R2h}) and the site map, which can be applied to unvisited land "
               "(R² {R2s}), supplied the intervals, fixed bins of the site map's prediction covered {N} of the 5% of FIA plots selected, "
               "whose site-map residuals averaged {r} error spreads, and the misses lay mostly {side} the interval; the rule's own "
               "reference set covered {J}, and binning by the selecting map's prediction instead covered {I10} with ten bins and {I20} "
               "with twenty, without a guarantee ([S25]). This is one dataset, and the guarantee assumes that the rule, including the "
               "selecting map's predictions at the audited plots, is declared; a second map built from coordinates alone, less accurate "
               "than the interval map, had left fixed bins below 0.85 in one of 11 settings ([S24])."),
        "A2": ("Fixed bins of the site map's prediction also held when a map with canopy height chose the plots: they covered {N} of the "
               "5% of FIA plots selected, and the rule's own reference set {J} ([S25]), contrary to our prediction."),
        "A3": ("Fixed bins of the site map's prediction fell short when the site map chose its own plots ({N0}), and further when a map "
               "with canopy height chose them ({N}; reference set {J}; [S25])."),
        "A4": "With a map with canopy height choosing the plots, the rule's own reference set covered {J} and fixed bins {N} of the selected FIA plots ([S25]).",
    }
    disc = {"A1": " (when a more accurate map chose the FIA plots, {N} against {J}; [S25])",
            "A2": "; in our tests they nonetheless held when a more accurate map chose the plots ([S25])"}
    tplB = {"B 阳性": ("When the rule also ranked a simulated field measurement whose error was set to half the map's, fixed bins fell "
                      "below 0.85 in {b5} of the 5 data sources with measured outcomes ({b11} of all 11 settings, including modelled and "
                      "reanalysis outcomes), and the reference set in none (up to {jinf} of its intervals infinite; [S25])."),
            "B 阴性": "A simulated field measurement moved fixed bins below 0.85 in {b5} of the 5 measured data sources ([S25]).",
            "B 对照失效": ("A simulated field measurement moved fixed bins below 0.85 in {b5} of the 5 measured data sources, but ranking "
                         "noise alone did so in {c11} of 11 settings ([S25]).")}
    L += ["## 按规则 7 填好的句子", ""]
    if audit:
        L += ["**核查触发**（规则 1）：" + "；".join(audit) + "。核查结束前不写任何结果。", ""]
    L += [f"- Results（{outA}）：" + tpl[outA].format(**fill)]
    if outA in disc:
        L += [f"- Discussion 阶梯句后（{outA}）：" + disc[outA].format(**fill)]
    if outB in tplB and complete:
        L += [f"- Results（{outB}）：" + tplB[outB].format(**fill)]
    L += [""]
    open(os.path.join(RES, "RESULTS_STRESS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    open(os.path.join(RES, "RESULTS_STRESS_SI.md"), "w", encoding="utf-8").write("\n".join(T) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "stress_summary.csv"), index=False, float_format="%.5f")
    print("\n".join(L))


if __name__ == "__main__":
    main()
