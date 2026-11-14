"""WP3 carbon anchor: tables from results/carbon_<map>.csv.gz (experiments/run_carbon.py).

    ~/miniforge3/envs/favela/bin/python experiments/make_carbon.py
writes results/RESULTS_CARBON.md. The figures are drawn in the manuscript style of figures/style.py and reuse
load / load_units / risk / ratio / col from here: Fig. 7 (VCS) is figures/make_fig7.py, Supplementary Fig. 15
(ART TREES, and the declared risk at any level) is figures/make_si_carbon.py.
The colours and markers in SER are those of the old figures (point = black star, JOMI family = blue, model-based =
green, carbon protocols = red); the tables use its labels only.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
from make_figs import md  # noqa: E402  (also sets the old rcParams)
import data as DS  # noqa: E402

RES = DS.FIA_RES                                      # FIA outputs (results/ for the old table, results/fia_db/ now)
MAPS = {"site": "site", "site+height": "siteheight", "full": "full"}
TITLE = {"site": "Site attributes", "site+height": "Site + canopy height", "full": "Full ground inventory"}
SHORT = {"site": "Site", "site+height": "Site +\nheight", "full": "Full\ninventory"}
FAM = {"VCS": dict(delta=1 / 3, rule="VCS MR v5.0", map="VCS map (RF)", ma="VCS model-assisted",
                   ours="ours: VCS formula, sel.-aware", exact="ours: exact, risk VCS"),
       "ART": dict(delta=0.30, rule="ART TREES v2.0", map="ART map (RF)", ma="ART model-assisted",
                   ours="ours: ART formula, sel.-aware", exact="ours: exact, risk ART")}
SER = {  # role -> (label, colour, marker, filled)
    "point": ("Map total, no deduction", "#0b0b0b", "*", True),
    "map": ("Protocol rule, map error layer", "#e34948", "D", True),
    "ma": ("Protocol rule, survey-calibrated estimate", "#1baf7a", "s", True),
    "ours": ("Protocol rule, selection-aware estimate (ours)", "#2a78d6", "o", True),
    "exact": ("Exact selection-aware bound (ours)", "#2a78d6", "o", False)}
ROLES = list(SER)


def load(tag=""):
    fs = [os.path.join(RES, f"carbon_{f}{tag}.csv.gz") for f in MAPS.values()]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["map"] = pd.Categorical(df["map"], list(MAPS), ordered=True)
    return df


def load_units():
    fs = [os.path.join(RES, f"carbon_units_{f}.csv.gz") for f in MAPS.values()]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        return None
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["map"] = pd.Categorical(df["map"], list(MAPS), ordered=True)
    return df


def col(fam, role):
    return "point" if role == "point" else FAM[fam][role]


def risk(df, m, keys):
    """Realized over-crediting risk with a fold-clustered 95% CI (maps are the clusters)."""
    over = (df[m] > df.total).astype(float)
    pf = over.groupby([df[k] for k in keys] + [df.fold], observed=True).mean()
    g = pf.groupby(level=list(range(len(keys))), observed=True)
    return pd.DataFrame({"risk": over.groupby([df[k] for k in keys], observed=True).mean(),
                         "ci": 1.96 * g.std() / np.sqrt(g.size()), "fmin": g.min(), "fmed": g.median(),
                         "fmax": g.max(), "nf": g.size()})


def ratio(df, m, keys):
    g = df.groupby(keys, observed=True)
    return g[m].sum() / g.total.sum()


def pct(f):
    return f"{100 * f:g}%"


def tables(df, rnd=None, units=None):
    out = ["# WP3 碳汇锚点：协议的声明风险在按地图选点时是否成立（正式实验）\n",
           "> 由 `experiments/make_carbon.py` 生成，数据为 `results/carbon_*.csv.gz`，实验脚本为 `experiments/run_carbon.py`。",
           "> 设计、协议原文与解读见 `paper/POLICY_ANCHORS.md`。\n"]
    nf = df.fold.nunique(); ns = df.split.nunique()
    r2 = df.groupby("map", observed=True).r2.agg(["mean", "min", "max"])
    out.append(f"**设计**：每种地图 {nf} 张（训练折）× 每张 {ns} 次审计划分；审计占比 0.2 / 0.5 / 0.8；"
               "选候选样地中地图预测最高的 0.5–10%。除特别说明，下表均为审计占比 0.5。\n")
    out.append("| 地图 | 留出 R²（均值，范围） |\n|---|---|")
    for mp, r in r2.iterrows():
        out.append(f"| {TITLE[mp]} | {r['mean']:.2f}（{r['min']:.2f}–{r['max']:.2f}） |")
    out.append("")
    a = df[df.audit == 0.5]
    fr = sorted(df.frac.unique())
    for fam, F in FAM.items():
        out.append(f"## 表 C{1 if fam == 'VCS' else 2}　实际超认证风险（{F['rule']}，声明风险 {F['delta']:.3g}）\n")
        out.append("每格为全部划分上的频率，括号内为按地图聚类的 95% 置信区间半宽。\n")
        rows = []
        for mp in MAPS:
            for role in ROLES:
                rk = risk(a[a["map"] == mp], col(fam, role), ["frac"])
                rows.append([TITLE[mp], SER[role][0]] + [f"{rk.risk[f]:.3f} (±{rk.ci[f]:.3f})" for f in fr])
        out.append(md(pd.DataFrame(rows, columns=["地图", "方法"] + [f"选 {pct(f)}" for f in fr])))
    out.append("## 表 C3　认证量 / 真实总量（VCS 一族；ART 见 CSV）\n")
    rows = []
    for mp in MAPS:
        for role in ROLES:
            q = ratio(a[a["map"] == mp], col("VCS", role), ["frac"])
            rows.append([TITLE[mp], SER[role][0]] + [f"{q[f]:.3f}" for f in fr])
    out.append(md(pd.DataFrame(rows, columns=["地图", "方法"] + [f"选 {pct(f)}" for f in fr])))
    out.append("## 表 C4　地图之间的差异（选 1%）\n")
    out.append("每张地图（训练折）单独算实际风险，再看各张地图之间的范围。\n")
    rows = []
    b = a[a.frac == 0.01]
    for fam, F in FAM.items():
        for mp in MAPS:
            for role in ("map", "ma", "ours"):
                rk = risk(b[b["map"] == mp], col(fam, role), ["frac"]).iloc[0]
                pf = (b[b["map"] == mp][col(fam, role)] > b[b["map"] == mp].total).groupby(b[b["map"] == mp].fold).mean()
                rows.append([fam, TITLE[mp], SER[role][0], f"{rk.fmin:.2f}–{rk.fmax:.2f}", f"{rk.fmed:.2f}",
                             f"{int((pf > F['delta'] + 1e-9).sum())}/{len(pf)}"])
    out.append(md(pd.DataFrame(rows, columns=["协议", "地图", "方法", "逐图风险范围", "中位", "超过声明值的地图数"])))
    out.append("## 表 C5　超认证有多严重（VCS 一族，选 1%）\n")
    rows = []
    for mp in MAPS:
        x = b[b["map"] == mp]
        for role in ROLES:
            m = col("VCS", role); ex = (x[m] - x.total) / x.total
            rows.append([TITLE[mp], SER[role][0], f"{np.mean(np.maximum(x[m] - x.total, 0)) / x.total.mean():.3f}",
                         f"{ex[ex > 0].mean():.3f}" if (ex > 0).any() else "—", f"{np.quantile(ex, 0.9):.3f}"])
    out.append(md(pd.DataFrame(rows, columns=["地图", "方法", "期望超认证量 / 真值", "超认证时平均超出", "超出比例的 90% 分位"])))
    out.append("## 表 C6　稳健性（选 1% / 5%，实际超认证风险）\n")
    rows = []
    for mp in ("site", "site+height"):
        x = df[df["map"] == mp]
        for lab, m, aud in [("VCS，地图误差层（RF ∣残差∣）", "VCS map (RF)", 0.5), ("VCS，地图误差层（QRF）", "VCS map (QRF)", 0.5),
                            ("VCS，误差层按完全相关合成", "VCS map (RF, corr.)", 0.5), ("ART，地图误差层（QRF）", "ART map (QRF)", 0.5),
                            ("ART，误差层按完全相关合成", "ART map (RF, corr.)", 0.5),
                            ("ours：VCS 公式，含 10% 豁免", "ours: VCS formula + waiver", 0.5)] + \
                [(f"{lab0}，审计占比 {au}", m0, au) for au in (0.2, 0.8)
                 for lab0, m0 in [("VCS，地图误差层", "VCS map (RF)"), ("VCS，调查校正", "VCS model-assisted"),
                                  ("ours：VCS 公式", "ours: VCS formula, sel.-aware"), ("ours：精确界 1/3", "ours: exact, risk VCS")]]:
            y = x[x.audit == aud]; rk = risk(y, m, ["frac"]); q = ratio(y, m, ["frac"])
            rows.append([TITLE[mp], lab] + [f"{rk.risk[f]:.3f}（{q[f]:.2f}）" for f in (0.01, 0.05)])
    out.append("括号内为认证量 / 真值。\n")
    out.append(md(pd.DataFrame(rows, columns=["地图", "变体", "选 1%", "选 5%"])))
    out.append("## 表 C7　声明风险的可靠性（高斯界在任意 δ 下的实际风险，选 1%，不含豁免）\n")
    rows = []
    for mp in MAPS:
        x = b[b["map"] == mp]
        for lab, e, s in [("地图误差层", "point", "se_rf"), ("调查校正", "ma", "se_ma"), ("选择感知（ours）", "sa", "se_sa")]:
            rows.append([TITLE[mp], lab] + [f"{np.mean(x[e] - norm.ppf(1 - d) * x[s] > x.total):.3f}" for d in DGRID])
    out.append(md(pd.DataFrame(rows, columns=["地图", "估计"] + [f"δ={d:.3g}" for d in DGRID])))
    kl = df.assign(kl=df.K / (df.K + df.nR)).groupby("audit").kl.mean()
    out.append("## 表 C8　精确界的诊断\n")
    out.append("精确界只有在 K/ℓ < δ 时才不需要支撑下限（`theory/POOL_LEMMA.md`）。各审计占比下 K/ℓ 的均值："
               + "，".join(f"{a_:g} → {v:.2f}" for a_, v in kl.items()) + "。\n")
    if units is not None:
        out.append("## 表 C10　逐样地发放信用（每个入选样地按其 90% 区间下端发放，审计占比 0.5）\n")
        out.append("脚本 `experiments/run_carbon_units.py`，地图、审计划分与入选样地都与上表相同。项目信用为各样地信用之和。"
                   "SC-GeoCCP 只算了选 1%。\n")
        rows = []
        for mp in MAPS:
            for f in (0.01, 0.05):
                x = units[(units["map"] == mp) & (units.frac == f)]
                for m in x.method.unique():
                    g = x[x.method == m]
                    rows.append([TITLE[mp], pct(f), m, f"{g.n_over.sum() / g.K.sum():.3f}", f"{g.credit.sum() / g.total.sum():.3f}",
                                 f"{g.credited_true.sum() / g.total.sum():.3f}", f"{np.mean(g.credit > g.total):.3f}"])
        out.append(md(pd.DataFrame(rows, columns=["地图", "选点", "方法", "超认证样地比例", "认证量 / 真值", "其中真实的部分 / 真值",
                                                  "项目超认证风险"])))
    if rnd is not None:
        folds = sorted(rnd.fold.unique())
        a_m = a[a.fold.isin(folds)]          # by-map selection on the same maps as the random control (matched)
        out.append("## 表 C9　对照：同样的 K 个样地改为随机挑选（审计占比 0.5，VCS 一族）\n")
        out.append("两列对比，可以把\"按地图选点\"的作用和\"误差层本身是否校准\"分开：随机挑选时的风险反映误差层或估计本身，"
                   "配对差才是选择造成的。注意 VCS 的 10% 豁免：全变量地图选 5% 时，两种挑选方式下几乎所有审计都满足 U ≤ 10%，不做扣减。"
                   f"随机挑选的对照用 {len(folds)} 张地图（训练折 {folds[0]}–{folds[-1]}）× {rnd.split.nunique()} 次划分，审计划分与 K 都与按地图选点相同；"
                   "\"按地图\"一列限定为这几张地图（2026-10-28 起；此前误用了全部 20 张地图）。"
                   "\"差\"是逐张地图的配对差（按地图减随机），括号内为按地图聚类的 95% 置信区间半宽（±1.96 s.e.，与本文其余表格一致）。"
                   f"最后一列是全部 {df.fold.nunique()} 张地图上按地图选点的风险（选 1%），只作参照，不能与随机一列相减。\n")
        rows = []
        for mp in MAPS:
            for role in ("point", "map", "ma", "ours"):
                m = col("VCS", role); cells = []
                for f in (0.01, 0.05):
                    x = a_m[(a_m["map"] == mp) & (a_m.frac == f)]
                    y = rnd[(rnd["map"] == mp) & (rnd.audit == 0.5) & (rnd.frac == f)]
                    if not len(y):
                        cells += ["—"] * 3
                        continue
                    ox = (x[m] > x.total).groupby(x.fold).mean()
                    oy = (y[m] > y.total).groupby(y.fold).mean()
                    d = (ox - oy).dropna()
                    cells += [f"{ox.mean():.3f}", f"{oy.mean():.3f}", f"{d.mean():+.3f} (±{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f})"]
                x_all = a[(a["map"] == mp) & (a.frac == 0.01)]
                rows.append([TITLE[mp], SER[role][0]] + cells + [f"{np.mean(x_all[m] > x_all.total):.3f}"])
        out.append(md(pd.DataFrame(rows, columns=["地图", "方法", "选 1%：按地图", "选 1%：随机", "选 1%：差", "选 5%：按地图", "选 5%：随机",
                                                  "选 5%：差", f"选 1%：按地图（{df.fold.nunique()} 张）"])))
        out.append("\n逐张地图的风险范围（同样这几张地图，选 1%，审计占比 0.5）：每张地图 100 次审计的超认证频率的最小值–最大值。"
                   "\"离散度\"是各张地图风险的方差与二项方差 r(1−r)/100 均值之比：约为 1 表示地图之间的差别只是审计噪声。\n")
        rows = []
        for mp in MAPS:
            for role in ("map", "ma", "ours"):
                m = col("VCS", role); cells = []
                for src in (a_m, rnd[rnd.audit == 0.5]):
                    x = src[(src["map"] == mp) & (src.frac == 0.01)]
                    pf = (x[m] > x.total).groupby(x.fold).mean()
                    nper = x.groupby("fold").size().mean()
                    disp = pf.var(ddof=1) / np.mean(pf * (1 - pf) / nper) if np.mean(pf * (1 - pf)) > 0 else np.nan
                    cells += [f"{pf.min():.2f}–{pf.max():.2f}", f"{disp:.1f}"]
                rows.append([TITLE[mp], SER[role][0]] + cells)
        out.append(md(pd.DataFrame(rows, columns=["地图", "方法", "按地图：范围", "按地图：离散度", "随机：范围", "随机：离散度"])))
    return "\n".join(out)


DGRID = (0.05, 0.10, 0.20, 1 / 3, 0.50)


def main():
    df = load()
    with open(os.path.join(RES, "RESULTS_CARBON.md"), "w") as f:
        f.write(tables(df, load("_random"), load_units()))
    print("wrote results/RESULTS_CARBON.md (figures: figures/make_fig7.py, figures/make_si_carbon.py)")


if __name__ == "__main__":
    main()
