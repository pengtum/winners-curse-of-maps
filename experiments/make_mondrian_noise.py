"""Tables for paper/MONDRIAN_NOISE_DESIGN.md: the Mondrian CP baseline, the split-level bootstrap check of the
sampling-noise limit against the independent-binomial limit used in Figs 6c and 8, and the per-split coverage
quantiles of JOMI.
Bootstrap (whole splits resampled, shared by all groups). The version fixed before the run recentres each group's
resampled pooled coverage at 0.90 but keeps that group's own spread, p_g (1 - p_g); a group covered at 1.00 then has
no noise. The version used for the decision ('bootstrap') also rescales each group's deviations to the variance at
0.90, keeping the split-level dependence within and across groups (binomial draws for groups with p_g = 0 or 1).
Design effect: bootstrap variance of a group's pooled coverage over p_g (1 - p_g) / n_g.
Finite intervals (Table S23e / M6; added 2026-10-30): infinite intervals count as covering (n_cov includes them), so per
setting, pooled over splits, coverage among finite intervals = (n_cov - n_inf) / (n_sel - n_inf) and the share of the
selected units covered by a finite interval = (n_cov - n_inf) / n_sel. Width under the quota (Table S23b / M4, last
column): fig_data.pooled widths (mean over splits of the mean finite half-width, as Supplementary Fig. 23), relative to
Mondrian CP with ten bins. mondrian_summary.csv appends the columns cov_finite, finite_covered (policy / budget rows),
width_vs_mondrian10 and inf_selected (+inf share among all units selected under the quota; quota_worst rows).
    ~/miniforge3/envs/favela/bin/python experiments/make_mondrian_noise.py
writes results/RESULTS_MONDRIAN_NOISE.md, results/RESULTS_MONDRIAN_NOISE_SI.md (Tables S23a-e),
results/mondrian_summary.csv, results/noise_limits.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures")); sys.dont_write_bytecode = True
import fig_data as D                                                  # noqa: E402
import style as S                                                     # noqa: E402

RES = os.path.join(ROOT, "results")
MD = ["Mondrian CP", "Mondrian CP, 20 bins", "Mondrian CP |res|"]
REF = ["SCP |res|", "SCP stud", "JOMI |res|", "JOMI stud"]
LAB = {"SCP |res|": "Split CP", "SCP stud": "Normalized CP", "JOMI |res|": "JOMI (additive)", "JOMI stud": "JOMI",
       "Mondrian CP": "Mondrian CP (10 bins)", "Mondrian CP, 20 bins": "Mondrian CP (20 bins)",
       "Mondrian CP |res|": "Mondrian CP (additive, 10 bins)"}
B = 2000


def r3(x):
    from decimal import Decimal, ROUND_HALF_UP
    return "—" if not np.isfinite(x) else str(Decimal(repr(round(float(x), 9))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def r2(x):
    from decimal import Decimal, ROUND_HALF_UP
    return "—" if not np.isfinite(x) else str(Decimal(repr(round(float(x), 9))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# Table S23e / M6: rules (key, English, Chinese), methods, statistics (English, Chinese) in row order
E_RULES = [("policy1", "Highest/lowest 1%", "政策方向 1%"), ("policy5", "Highest/lowest 5%", "政策方向 5%"),
           ("policy20", "Highest/lowest 20%", "政策方向 20%"), ("budget", "Greedy travel-cost budget", "贪心旅行成本预算")]
E_METH = ["JOMI stud", "JOMI |res|", "Mondrian CP", "Mondrian CP, 20 bins", "Mondrian CP |res|", "SCP stud"]
E_LAB = {"JOMI stud": "JOMI", "JOMI |res|": "JOMI, additive", "Mondrian CP": "Mondrian CP", "Mondrian CP, 20 bins": "Mondrian CP, 20 bins",
         "Mondrian CP |res|": "Mondrian CP, additive", "SCP stud": "Normalized CP"}
E_STATS = [("Coverage (median)", "合并覆盖（中位数）"), ("Coverage, finite intervals only (median)", "仅有限区间的覆盖（中位数）"),
           ("Largest share of infinite intervals", "+∞ 比例（最大）"),
           ("Covered by a finite interval: median (minimum)", "有限且覆盖的比例：中位数（最小）"),
           ("Sum of absolute deviations from 0.90", "与 0.90 的绝对偏差之和")]
QW = ("SCP stud", "JOMI stud", "SC-GeoCCP", "Mondrian CP", "Mondrian CP, 20 bins")   # Table S23b / M4 columns


def finite(a, r, m):
    """Pooled over splits: coverage among the selected units that received a finite interval, and the share of the
    selected units that received a finite interval that covers (infinite intervals count as covering, so finite and
    covering = n_cov - n_inf)."""
    x = a[(a.rule == r) & (a.method == m)]; s, c, i = x.n_sel.sum(), x.n_cov.sum(), x.n_inf.sum()
    assert c >= i, (r, m)
    return dict(cov_finite=(c - i) / (s - i) if s > i else np.nan, finite_covered=(c - i) / s)


def finite_cells(o, t, m):
    """Cells of Table S23e / M6 for rule t and method m over the 11 settings, from the full-precision values."""
    x = o[(o.table == t) & (o.method == m)]; assert len(x) == len(S.DATASETS), (t, m, len(x))
    cf = x.cov_finite.dropna(); inf = x.inf.max()
    return [r3(x["cov"].median()), r3(cf.median()) + ("" if len(cf) == len(x) else f" ({len(cf)} of {len(x)})"),
            "<0.01" if 0 < inf < 0.005 else r2(inf), f"{r2(x.finite_covered.median())} ({r2(x.finite_covered.min())})",
            r3((x["cov"] - (1 - D.ALPHA)).abs().sum())]


def quota_widths(nm, r):
    """Width (fig_data.pooled: mean over splits of the mean finite half-width, as Supplementary Fig. 23) and +inf share of
    each Table S23b method under the quota rule r: normalized CP, JOMI and SC-GeoCCP from fig_data.runs, Mondrian CP
    from md_main_<nm>.csv."""
    x = pd.concat([D.runs(nm), pd.read_csv(os.path.join(RES, f"md_main_{nm}.csv"))[D.COLS]], ignore_index=True)
    x = x[(x.rule == r) & x.method.isin(QW)]
    assert not x.duplicated(["method", "rep"]).any() and set(x.method) == set(QW), nm
    return D.pooled(x).xs(r, level="rule")


def quota_ratio(o):
    """Per setting: SC-GeoCCP's width relative to Mondrian CP (10 bins) under the quota, whether the outcome is observed
    (measured), and those of the two methods that gave more than 1% of the selected units an infinite interval."""
    q = o[o.table == "quota_worst"]; rows = []
    for nm, lab, tr in S.DATASETS:
        x = q[q.dataset == nm].set_index("method")
        rows.append(dict(dataset=nm, lab=lab, observed=tr == "observed", ratio=x.loc["SC-GeoCCP", "width_vs_mondrian10"],
                         inf={m: x.loc[m, "inf_selected"] for m in ("SC-GeoCCP", "Mondrian CP") if x.loc[m, "inf_selected"] > 0.01}))
    return pd.DataFrame(rows).set_index("dataset")


def runs(nm):
    a = pd.read_csv(os.path.join(RES, f"main_{nm}.csv")).rename(columns={"hw_mean": "hw_mean_fin"})
    a = a[a.method.isin(REF)][["rule", "method", "rep", "n_sel", "n_cov", "n_inf", "hw_mean_fin"]]
    b = pd.read_csv(os.path.join(RES, f"md_main_{nm}.csv"))[["rule", "method", "rep", "n_sel", "n_cov", "n_inf", "hw_mean_fin"]]
    return pd.concat([a, b], ignore_index=True)


def boot_limit(sg, nmin=D.NMIN, seed=0, null=True):
    """Split-level bootstrap of the expected worst group if every group's coverage were 0.90: resample whole splits
    (shared by all groups), recentre each group's pooled coverage at 0.90 (null=True: deviations rescaled to the
    variance at 0.90; null=False: the version fixed before the run, own spread), take the minimum over groups.
    Returns the limit, the number of groups, its Monte Carlo s.e. and the median and largest design effect."""
    tot = sg.groupby("group")[["n", "c"]].sum(); keep = tot.index[tot.n >= nmin]
    if len(keep) == 0:
        return dict(limit=np.nan, groups=0, mcse=np.nan, deff_med=np.nan, deff_max=np.nan)
    x = sg[sg.group.isin(keep)]
    reps = np.sort(x.rep.unique()); G = list(keep)
    n = x.pivot_table(index="rep", columns="group", values="n", aggfunc="sum").reindex(index=reps, columns=G).fillna(0).values
    c = x.pivot_table(index="rep", columns="group", values="c", aggfunc="sum").reindex(index=reps, columns=G).fillna(0).values
    p, N = c.sum(0) / n.sum(0), n.sum(0).astype(int); t = 1 - D.ALPHA
    inner = (p > 0) & (p < 1); scale = np.sqrt(t * (1 - t) / np.where(inner, p * (1 - p), 1.0))
    rng = np.random.default_rng(seed); w = np.zeros(B); P = np.zeros((B, len(G)))
    for b in range(B):
        idx = rng.integers(0, len(reps), len(reps))
        nb, cb = n[idx].sum(0), c[idx].sum(0)
        P[b] = np.where(nb > 0, cb / np.maximum(nb, 1), p)        # a group absent from a resample keeps its mean
        d = P[b] - p
        if null:
            d = d * scale
            d[~inner] = rng.binomial(N[~inner], t) / N[~inner] - t
        w[b] = (t + d).min()
    de = P[:, inner].var(0) / (p[inner] * (1 - p[inner]) / N[inner])
    return dict(limit=float(w.mean()), groups=len(G), mcse=float(w.std(ddof=1) / np.sqrt(B)),
                deff_med=float(np.median(de)) if len(de) else np.nan, deff_max=float(de.max()) if len(de) else np.nan)


def main():
    L = ["# Mondrian CP 基线、按划分的噪声下限与逐划分分位数（2026-10-29）", "",
         "设计：`paper/MONDRIAN_NOISE_DESIGN.md`（跑前写定）；运行 `experiments/run_mondrian.py`（与 run_main、run_baselines 同种子；"
         "normalized CP 逐划分计数与 JOMI 配额规则下逐州计数逐划分核对一致），表 `experiments/make_mondrian_noise.py`。覆盖为 100 次划分合并；+∞ 计为覆盖。", ""]
    out = []
    for pct in (5, 1, 20):
        L += [f"## 表 M{ {5: 1, 1: 2, 20: 3}[pct] }　政策方向 {pct}% 规则：被选单元上的覆盖（+∞ 比例；有限半宽与 JOMI 之比）", "",
              "| 设定 | " + " | ".join(LAB[m] for m in REF + MD) + " |", "|---|" + "---|" * len(REF + MD)]
        below = {m: 0 for m in REF + MD}
        for nm, lab, _ in S.DATASETS:
            a = runs(nm); P = D.pooled(a); r = D.policy_rule(nm, pct)
            wj = P.loc[(r, "JOMI stud"), "width"]
            cells = []
            for m in REF + MD:
                x = P.loc[(r, m)]
                below[m] += int(x["cov"] < 0.85)
                cells.append(f"{r3(x['cov'])} ({x.inf:.2f}; {x.width / wj:.2f})")
                out.append(dict(table=f"policy{pct}", dataset=nm, rule=r, method=m, cov=x["cov"], inf=x.inf, width_ratio=x.width / wj,
                                **finite(a, r, m)))
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("| **低于 0.85 的设定数（共 11 个）** | " + " | ".join(f"**{below[m]}**" for m in REF + MD) + " |")
        L.append("")
    # greedy travel-cost budget rule
    L += ["## 表 M5　贪心旅行成本预算规则：被选单元上的覆盖（+∞ 比例；宽度与 JOMI 之比）", "",
          "| 设定 | " + " | ".join(LAB[m] for m in REF + MD) + " |", "|---|" + "---|" * len(REF + MD)]
    below = {m: 0 for m in REF + MD}
    for nm, lab, _ in S.DATASETS:
        a = runs(nm); P = D.pooled(a); r = D.budget_rule(nm); wj = P.loc[(r, "JOMI stud"), "width"]; cells = []
        for m in REF + MD:
            x = P.loc[(r, m)]; below[m] += int(x["cov"] < 0.85)
            cells.append(f"{r3(x['cov'])} ({x.inf:.2f}; {x.width / wj:.2f})")
            out.append(dict(table="budget", dataset=nm, rule=r, method=m, cov=x["cov"], inf=x.inf, width_ratio=x.width / wj,
                            **finite(a, r, m)))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["| **低于 0.85 的设定数（共 11 个）** | " + " | ".join(f"**{below[m]}**" for m in REF + MD) + " |", ""]
    # quota worst group, Mondrian vs others; last column (2026-10-30): SC-GeoCCP's width relative to Mondrian CP (10 bins)
    L += ["## 表 M4　配额规则（5%）下的最差州/国：覆盖（该组的 +∞ 比例）；末列为同一规则下 SC-GeoCCP 与 Mondrian CP (10 bins) 的有限半宽之比", "",
          "| 设定 | Normalized CP | JOMI | SC-GeoCCP | Mondrian CP (10 bins) | Mondrian CP (20 bins) | 有限半宽：SC-GeoCCP ÷ Mondrian CP (10 bins) |",
          "|---|---|---|---|---|---|---|"]
    for nm, lab, _ in S.DATASETS:
        g = D.groups(nm, "state"); md = pd.read_csv(os.path.join(RES, f"md_groups_{nm}.csv")); md = md[md.gtype == "state"]
        gg = pd.concat([g, md], ignore_index=True); r = D.quota_rule(nm)
        Pw = quota_widths(nm, r); w10 = Pw.loc["Mondrian CP", "width"]
        cells = []
        for m in QW:
            w, wi, ng, _ = D.worst(gg, r, m)
            cells.append(f"{r3(w)} ({wi:.2f})" if np.isfinite(w) else "—")
            out.append(dict(table="quota_worst", dataset=nm, rule=r, method=m, worst=w, inf_in_worst=wi, groups=ng,
                            width_vs_mondrian10=Pw.loc[m, "width"] / w10, inf_selected=Pw.loc[m, "inf"]))
        cells.append(r2(Pw.loc["SC-GeoCCP", "width"] / w10) + ("‡" if Pw.loc[["SC-GeoCCP", "Mondrian CP"], "inf"].max() > 0.01 else ""))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    W = quota_ratio(pd.DataFrame(out)); w6 = W[W.observed]
    L += ["", f"末列：fig_data.pooled 的宽度（逐划分平均有限半宽的 100 次划分平均，同补充图 23）之比。中位数：11 个设定 {r2(W.ratio.median())}"
          f"（{r2(W.ratio.min())}–{r2(W.ratio.max())}）；{len(w6)} 个实测结局设定（{'、'.join(w6.lab)}）{r2(w6.ratio.median())}"
          f"（{r2(w6.ratio.min())}–{r2(w6.ratio.max())}）。"
          + ("‡ 宽度只计有限区间；超过 1% 的被选单元区间为 +∞：" + "；".join(f"{x.lab}，{m} {v:.1%}" for _, x in W.iterrows() for m, v in x.inf.items())
             + "。" if W.inf.map(len).any() else "")]
    # split-level bootstrap noise limit
    L += ["", "## 表 N1　抽样噪声下限：独立二项与按划分的自助法（配额规则，JOMI 的逐划分计数，被选至少 50 次的组）", "",
          "自助法：按 0.90 方差缩放（决策所用）；事先写定版：只平移到 0.90，保留各组自身的离散度。设计效应：各组合并覆盖的自助方差 ÷ p(1−p)/n。", "",
          "| 设定 | 层级 | 组数 | 独立二项 | 自助法（0.90 方差） | 差 | 事先写定版 | 差 | 设计效应中位数（最大） |",
          "|---|---|---|---|---|---|---|---|---|"]
    lim = []
    for nm, lab, _ in S.DATASETS:
        sg = pd.read_csv(os.path.join(RES, f"md_split_groups_{nm}.csv")); r = D.quota_rule(nm)
        for gt in ("state", "cell"):
            x = sg[(sg.rule == r) & (sg.method == "JOMI stud") & (sg.gtype == gt)]
            tot = x.groupby("group").n.sum(); nn = tot[tot >= D.NMIN].values
            binom = D.null_worst(nn) if len(nn) else np.nan
            bs, pr = boot_limit(x), boot_limit(x, null=False)
            lim.append(dict(dataset=nm, gtype=gt, groups=bs["groups"], binomial=binom, bootstrap=bs["limit"], diff=bs["limit"] - binom,
                            mcse=bs["mcse"], bootstrap_prespec=pr["limit"], diff_prespec=pr["limit"] - binom,
                            deff_median=bs["deff_med"], deff_max=bs["deff_max"]))
            L.append(f"| {lab} | {'州/国' if gt == 'state' else '空间格'} | {bs['groups']} | {r3(binom)} | {r3(bs['limit'])} | "
                     f"{bs['limit'] - binom:+.3f} | {r3(pr['limit'])} | {pr['limit'] - binom:+.3f} | {bs['deff_med']:.2f}（{bs['deff_max']:.2f}） |")
    lim = pd.DataFrame(lim)
    L += ["", f"差（自助法 − 二项）：中位数 {lim['diff'].median():+.4f}，范围 {lim['diff'].min():+.4f} 到 {lim['diff'].max():+.4f}；"
          f"Monte Carlo 标准误至多 {lim.mcse.max():.4f}。事先写定版：中位数 {lim.diff_prespec.median():+.4f}，范围 "
          f"{lim.diff_prespec.min():+.4f} 到 {lim.diff_prespec.max():+.4f}。设计效应：各格中位数的中位数 {lim.deff_median.median():.2f}，"
          f"各组最大 {lim.deff_max.max():.2f}。", ""]
    # per-split quantiles of JOMI at the 5% policy rule
    L += ["## 表 Q1　5% 政策方向规则下逐划分覆盖的分布（被选单元；100 次划分）", "",
          "| 设定 | JOMI：合并 | JOMI：第 10 百分位 | JOMI：低于 0.85 的划分比例 | Normalized CP：第 10 百分位 | Normalized CP：低于 0.85 的划分比例 |",
          "|---|---|---|---|---|---|"]
    for nm, lab, _ in S.DATASETS:
        a = runs(nm); r = D.policy_rule(nm, 5)
        cells = []
        for m in ("JOMI stud", "SCP stud"):
            x = a[(a.rule == r) & (a.method == m)]; c = x.n_cov / x.n_sel
            if m == "JOMI stud":
                cells.append(r3(x.n_cov.sum() / x.n_sel.sum()))
            cells += [r3(c.quantile(0.1)), f"{np.mean(c < 0.85):.2f}"]
            out.append(dict(table="Q1", dataset=nm, rule=r, method=m, p10=c.quantile(0.1), share_below=np.mean(c < 0.85)))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    o = pd.DataFrame(out); new = ["cov_finite", "finite_covered", "width_vs_mondrian10", "inf_selected"]
    o = o[[c for c in o.columns if c not in new] + new]               # added columns last: the earlier ones keep their place
    # Table M6 (= S23e, 2026-10-30): with and without the infinite intervals, summarised over the 11 settings
    labs = {nm: lab for nm, lab, _ in S.DATASETS}
    L += ["", "## 表 M6　1%、5%、20% 与预算规则：计入与不计 +∞ 的覆盖、+∞ 比例、有限且覆盖的比例与绝对偏差之和（11 个设定汇总）", "",
          "每个设定按 100 次划分合并：仅有限区间的覆盖 = 有限且覆盖 ÷ 得到有限区间的被选单元；有限且覆盖的比例 = 有限且覆盖 ÷ 被选单元"
          "（+∞ 计为覆盖，故有限且覆盖 = 覆盖数 − +∞ 数）。中位数、最大与最小取自 11 个设定；合并覆盖与 0.90 之差的绝对值对 11 个设定求和。", "",
          "| 规则 | 统计量 | " + " | ".join(LAB[m] for m in E_METH) + " |", "|---|---|" + "---|" * len(E_METH)]
    for t, _, zh in E_RULES:
        cells = {m: finite_cells(o, t, m) for m in E_METH}
        for k, (_, name) in enumerate(E_STATS):
            L.append(f"| {zh} | {name} | " + " | ".join(cells[m][k] for m in E_METH) + " |")
    p1 = o[o.table == "policy1"]; j = p1[(p1.method == "JOMI stud") & (p1.inf > 0)].dataset.tolist()
    L += ["", f"1% 规则下 JOMI 给出 +∞ 区间的 {len(j)} 个设定（{'、'.join(labs[d] for d in j)}）中，有限且覆盖的比例："
          + "；".join(f"{LAB[m]} {r2(v.min())}–{r2(v.max())}" for m in ("JOMI stud", "Mondrian CP", "Mondrian CP, 20 bins")
                     for v in [p1[(p1.method == m) & p1.dataset.isin(j)].finite_covered]) + "。"]
    open(os.path.join(RES, "RESULTS_MONDRIAN_NOISE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    si_tables(o, lim)
    o.to_csv(os.path.join(RES, "mondrian_summary.csv"), index=False, float_format="%.5f")
    lim.to_csv(os.path.join(RES, "noise_limits.csv"), index=False, float_format="%.5f")
    print("\n".join(L))


def si_tables(o, lim):
    """Tables S23a-e of paper/supplementary_draft.md, from the full-precision values (no second rounding)."""
    def inf2(x):
        return "<0.01" if 0 < x < 0.005 else f"{x:.2f}"
    labs = {nm: lab for nm, lab, _ in S.DATASETS}
    A = [("Split CP", "SCP |res|"), ("Normalized CP", "SCP stud"), ("Mondrian CP", "Mondrian CP"),
         ("Mondrian CP, 20 bins", "Mondrian CP, 20 bins"), ("Mondrian CP, additive", "Mondrian CP |res|"), ("JOMI", "JOMI stud")]
    L = ["Table S23a. Settings (of 11) with coverage among the selected units below 0.85; in parentheses the lowest coverage "
         "and the largest share of infinite intervals across settings.", "",
         "| Rule | " + " | ".join(a for a, _ in A) + " |", "|---|" + "---|" * len(A)]
    for t, lab in (("policy1", "Highest/lowest 1%"), ("policy5", "Highest/lowest 5%"), ("policy20", "Highest/lowest 20%"),
                   ("budget", "Greedy travel-cost budget")):
        cells = []
        for _, m in A:
            x = o[(o.table == t) & (o.method == m)]
            cells.append(f"{int((x['cov'] < 0.85).sum())} ({r3(x['cov'].min())}; {inf2(x.inf.max())})")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    B_ = [("Normalized CP", "SCP stud"), ("JOMI", "JOMI stud"), ("SC-GeoCCP", "SC-GeoCCP"), ("Mondrian CP", "Mondrian CP"),
          ("Mondrian CP, 20 bins", "Mondrian CP, 20 bins")]
    W = quota_ratio(o); w6 = W[W.observed]
    L += ["", "Table S23b. Coverage in the worst state or country under the 5% quota (groups selected at least 50 times), and "
          f"SC-GeoCCP's mean finite half-width relative to Mondrian CP's under the same rule (median {r2(W.ratio.median())} over the "
          f"11 settings and {r2(w6.ratio.median())} over the {({5: 'five', 6: 'six', 7: 'seven'}).get(len(w6), len(w6))} with "
          "measured outcomes).", "",
          "| Setting | " + " | ".join(a for a, _ in B_) + " | SC-GeoCCP half-width (× Mondrian CP) |", "|---|" + "---|" * (len(B_) + 1)]
    notes = []
    for nm, lab, _ in S.DATASETS:
        cells = []
        for a, m in B_:
            x = o[(o.table == "quota_worst") & (o.dataset == nm) & (o.method == m)].iloc[0]
            mark = "†" if x.inf_in_worst >= 0.10 else ""
            if mark:
                notes.append(f"{lab}, {a}: {x.inf_in_worst:.0%}")
            cells.append(r3(x.worst) + mark)
        cells.append(r2(W.loc[nm, "ratio"]) + ("‡" if W.loc[nm, "inf"] else ""))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    if notes:
        L += ["", "† At least 10% of the intervals in that group were infinite (counted as covering): " + "; ".join(notes) + "."]
    wn = [f"{x.lab}, {m}: {v:.0%}" for _, x in W.iterrows() for m, v in x.inf.items()]
    if wn:
        L += ["", "‡ Widths count finite intervals only; more than 1% of the selected units had an infinite interval: " + "; ".join(wn) + "."]
    L += ["", "Table S23c. Sampling-noise limit under the quota rule: independent binomial; split-level bootstrap with deviations "
          "scaled to the variance at 0.90; and the bootstrap fixed before the run, which keeps each group's own spread.", "",
          "| Setting | States or countries: binomial | bootstrap | pre-specified bootstrap | Spatial cells: binomial | bootstrap | pre-specified bootstrap |",
          "|---|---|---|---|---|---|---|"]
    for nm, lab, _ in S.DATASETS:
        cells = []
        for gt in ("state", "cell"):
            x = lim[(lim.dataset == nm) & (lim.gtype == gt)].iloc[0]
            cells += [r3(x.binomial), r3(x.bootstrap), r3(x.bootstrap_prespec)]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "Table S23d. JOMI at the 5% policy-direction rule: distribution of per-split coverage among the selected units (100 splits).", "",
          "| Setting | Pooled coverage | 10th percentile of per-split coverage | Share of splits below 0.85 |", "|---|---|---|---|"]
    for nm, lab, _ in S.DATASETS:
        a = runs(nm); r = D.policy_rule(nm, 5); x = a[(a.rule == r) & (a.method == "JOMI stud")]; c = x.n_cov / x.n_sel
        L.append(f"| {lab} | {r3(x.n_cov.sum() / x.n_sel.sum())} | {r3(c.quantile(0.1))} | {np.mean(c < 0.85):.2f} |")
    L += ["", "Table S23e. Coverage with and without the infinite intervals under the policy-direction and budget rules (11 settings, "
          "100 splits): median coverage among the selected units, counting infinite intervals as covering, and among those with a "
          "finite interval; the largest share of infinite intervals; the share of selected units covered by a finite interval "
          "(median; minimum in parentheses); and the sum over settings of the absolute deviation of coverage from 0.90.", "",
          "| Rule | Statistic | " + " | ".join(E_LAB[m] for m in E_METH) + " |", "|---|---|" + "---|" * len(E_METH)]
    for t, lab, _ in E_RULES:
        cells = {m: finite_cells(o, t, m) for m in E_METH}
        for k, (name, _) in enumerate(E_STATS):
            L.append(f"| {lab} | {name} | " + " | ".join(cells[m][k] for m in E_METH) + " |")
    open(os.path.join(RES, "RESULTS_MONDRIAN_NOISE_SI.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
