"""Tables for the domain (post-stratified) difference estimator control (paper/CARBON_DOMAIN_DESIGN.md;
experiments/run_carbon_domain.py), next to the series of experiments/run_carbon.py on the same maps, audits and selections.
    ~/miniforge3/envs/favela/bin/python experiments/make_carbon_domain.py
writes results/RESULTS_CARBON_DOMAIN.md and results/carbon_domain_summary.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import make_carbon as MC                                              # noqa: E402

RES = MC.RES                                                          # FIA outputs: experiments/data.py FIA_RES
KEYS = ["map", "fold", "split", "audit", "frac"]
SERIES = [("VCS map (RF)", "Error layer"), ("VCS model-assisted", "Survey-calibrated (all audit plots)"),
          ("VCS domain5", "Domain, 5 strata"), ("VCS domain10", "Domain, 10 strata (main)"),
          ("VCS domain20", "Domain, 20 strata"), ("VCS domain10, formula", "Domain, 10 strata, formula without waiver"),
          ("ours: VCS formula, sel.-aware", "Selection-aware (reference set)")]
ART = [("ART model-assisted", "Survey-calibrated"), ("ART domain10", "Domain, 10 strata"),
       ("ours: ART formula, sel.-aware", "Selection-aware")]


def joined(tag="", folder=RES):
    if folder == RES:
        a = MC.load(tag)
    else:                                       # post-hoc 20-map random control (results/carbon_random20)
        a = pd.concat([pd.read_csv(os.path.join(folder, f"carbon_{f}{tag}.csv.gz")) for f in MC.MAPS.values()], ignore_index=True)
        a["map"] = pd.Categorical(a["map"], list(MC.MAPS), ordered=True)
    fs = [os.path.join(folder, f"carbon_domain_{f}{tag}.csv.gz") for f in MC.MAPS.values()]
    b = pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)], ignore_index=True)
    b["map"] = pd.Categorical(b["map"], list(MC.MAPS), ordered=True)
    keep = [c for c in b.columns if c not in a.columns or c in KEYS]
    m = a.merge(b[keep], on=KEYS, how="inner", validate="one_to_one")
    assert len(m) == len(b), (tag, len(m), len(b))
    return m


def r3(x):
    """Three decimals, half up: risks are counts / 2,000, and plain .3f rounds such halves inconsistently."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def cell(r):
    return f"{r3(r.risk)} (±{r3(r.ci)})"


def main():
    df, rnd = joined(), joined("_random")
    a = df[df.audit == 0.5]
    L = ["# 碳汇：域（后分层）差分估计对照（2026-10-29）", "",
         "设计见 `paper/CARBON_DOMAIN_DESIGN.md`（跑前写定）；运行 `experiments/run_carbon_domain.py`，表 `experiments/make_carbon_domain.py`。"
         "地图、审计划分与选点与 `run_carbon.py` 相同（重拟合的 μ̂ 与已存的 point、survey-calibrated 两列逐项核对一致）。"
         "域估计：按训练折预测值的分位数分层（10 层为主，5、20 层为敏感性），每层审计样地不足 10 个时向下合并；"
         "被选样地按所在层用该层审计样地的平均残差校正，SE = (Σ_h s_h²(K_h + K_h²/n_h))^½；VCS 完整规则（含 10% 豁免与 U > 100% 不予认证）。"
         "风险 = 认证量超过被选样地真实总量的比例；括号内为按地图（训练折）聚类的 95% 区间半宽。", "",
         "**解读（2026-10-29，三路对抗核查之后）。**",
         "- 固定分层只把失效收窄，并没有消除。最高 1% 时，十层的风险在 site 地图为 0.53，canopy-height 地图为 0.64；去掉 VCS 豁免后分别为 0.53、0.59。"
         "同样不含豁免、在同一批审计上与选择感知估计配对，差为 +0.20 ± 0.11 和 +0.26 ± 0.13。",
         "- 原因：顶层占审计样地的 10–12%，被选的只有 1%。选择感知估计其实就是只有一层、层界放在选点阈值上的后分层差分估计。",
         "- 随机选点对照的结果是混合的。canopy-height 地图上，地图选点使风险显著升高（+0.23 ± 0.13）；site 地图（事先写定的主终点）上不显著（+0.14 ± 0.17），"
         "而且随机选点时风险本身就是 0.375 ± 0.041，略高于 1/3。所以在 site 地图上，超出部分不能全归因于选择。这一情形不在事先写定的两条解读规则之内，如实写明。", ""]
    out = []
    L += ["## 表 D1　实际超认证风险，VCS（声明 1/3），审计占比 0.5，按地图选点（20 张地图 × 100 次审计）", "",
          "| 地图 | 选点 | " + " | ".join(lab for _, lab in SERIES) + " |", "|---|---|" + "---|" * len(SERIES)]
    for mp in MC.MAPS:
        for f in (0.005, 0.01, 0.02, 0.05, 0.10):
            x = a[(a["map"] == mp) & (a.frac == f)]
            cells = []
            for m, lab in SERIES:
                r = MC.risk(x, m, ["map"]).iloc[0]; cells.append(cell(r))
                out.append(dict(table="D1", map=mp, frac=f, series=lab, risk=r.risk, ci=r.ci, fmin=r.fmin, fmax=r.fmax,
                                credited=MC.ratio(x, m, ["map"]).iloc[0]))
            L.append(f"| {MC.TITLE[mp]} | {MC.pct(f)} | " + " | ".join(cells) + " |")
    L += ["", "## 表 D2　认证份额（认证量 / 真实总量），同表 D1", "",
          "| 地图 | 选点 | " + " | ".join(lab for _, lab in SERIES) + " |", "|---|---|" + "---|" * len(SERIES)]
    for mp in MC.MAPS:
        for f in (0.01, 0.05):
            x = a[(a["map"] == mp) & (a.frac == f)]
            L.append(f"| {MC.TITLE[mp]} | {MC.pct(f)} | " + " | ".join(f"{MC.ratio(x, m, ['map']).iloc[0]:.3f}" for m, _ in SERIES) + " |")
    folds = sorted(rnd.fold.unique()); a_m = a[a.fold.isin(folds)]; ar = rnd[rnd.audit == 0.5]
    L += ["", f"## 表 D3　随机选点对照（同 {len(folds)} 张地图，训练折 {folds[0]}–{folds[-1]}，审计划分与 K 相同），VCS", "",
          "\"差\"是逐张地图的配对差（按地图选点减随机选点），括号内为按地图聚类的 95% 区间半宽。", "",
          "| 地图 | 方法 | 选 1%：按地图 | 选 1%：随机 | 选 1%：差 | 选 5%：按地图 | 选 5%：随机 | 选 5%：差 |", "|---|---|---|---|---|---|---|---|"]
    for mp in MC.MAPS:
        for m, lab in SERIES:
            cells = []
            for f in (0.01, 0.05):
                x = a_m[(a_m["map"] == mp) & (a_m.frac == f)]; y = ar[(ar["map"] == mp) & (ar.frac == f)]
                ox = (x[m] > x.total).groupby(x.fold).mean(); oy = (y[m] > y.total).groupby(y.fold).mean(); d = (ox - oy).dropna()
                ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                cells += [r3(ox.mean()), r3(oy.mean()), f"{'+' if d.mean() >= 0 else '−'}{r3(abs(d.mean()))} (±{r3(ci)})"]
                out.append(dict(table="D3", map=mp, frac=f, series=lab, risk_map=ox.mean(), risk_random=oy.mean(),
                                paired=d.mean(), paired_ci=ci))
            L.append(f"| {MC.TITLE[mp]} | {lab} | " + " | ".join(cells) + " |")
    L += ["", "## 表 D4　ART TREES（声明 0.30），审计占比 0.5，按地图选点", "",
          "| 地图 | 选点 | " + " | ".join(lab for _, lab in ART) + " |", "|---|---|" + "---|" * len(ART)]
    for mp in MC.MAPS:
        for f in (0.01, 0.05):
            x = a[(a["map"] == mp) & (a.frac == f)]
            L.append(f"| {MC.TITLE[mp]} | {MC.pct(f)} | " + " | ".join(cell(MC.risk(x, m, ['map']).iloc[0]) for m, _ in ART) + " |")
    L += ["", "## 表 D5　机制：每块被选样地的平均残差（真值 − 预测），审计占比 0.5", "",
          "比较被选样地的实际平均残差、全部审计样地的平均残差（survey-calibrated 的校正量）、"
          "所在层审计样地的平均残差（域估计 10 层的校正量）与参考集的平均残差（选择感知估计的校正量），单位同生物量密度。", "",
          "| 地图 | 选点 | 被选样地实际 | 全部审计样地 | 同层审计样地（10 层） | 参考集 | 域估计 U ≤ 10%（不扣减）的比例 |", "|---|---|---|---|---|---|---|"]
    for mp in MC.MAPS:
        for f in (0.01, 0.05):
            x = a[(a["map"] == mp) & (a.frac == f)]; K = x.K
            u = 1.6449 * x.dom10_se / x.dom10_est
            L.append(f"| {MC.TITLE[mp]} | {MC.pct(f)} | {((x.total - x.point) / K).mean():+.2f} | {((x.ma - x.point) / K).mean():+.2f} | "
                     f"{((x.dom10_est - x.point) / K).mean():+.2f} | {((x.sa - x.point) / K).mean():+.2f} | {np.mean(u <= 0.10):.2f} |")
    L += ["", "## 表 D6　其他审计占比（选 1%，VCS）", "", "| 地图 | 审计占比 | Survey-calibrated | Domain, 10 strata | Selection-aware |", "|---|---|---|---|---|"]
    for mp in MC.MAPS:
        for au in (0.2, 0.8):
            x = df[(df["map"] == mp) & (df.frac == 0.01) & (df.audit == au)]
            L.append(f"| {MC.TITLE[mp]} | {au} | " + " | ".join(cell(MC.risk(x, m, ['map']).iloc[0]) for m in
                                                           ("VCS model-assisted", "VCS domain10", "ours: VCS formula, sel.-aware")) + " |")
    R20 = os.path.join(RES, "carbon_random20")
    if os.path.exists(os.path.join(R20, "carbon_domain_site_random.csv.gz")):
        r20 = joined("_random", R20); a20 = r20[r20.audit == 0.5]; folds20 = sorted(r20.fold.unique())
        L += ["", f"## 表 D7　事后补充：随机选点对照扩到 {len(folds20)} 张地图（看过 10 张地图的结果后决定，见设计文件\"事后补充\"一节），VCS", "",
              "与表 D3 同口径（按地图选点减随机选点，逐张地图配对，按地图聚类的 95% 区间半宽），只是地图为训练折 "
              f"{folds20[0]}–{folds20[-1]}。事先写定的终点仍以表 D3（10 张地图）为准。", "",
              "| 地图 | 方法 | 选 1%：按地图 | 选 1%：随机 | 选 1%：差 | 选 5%：按地图 | 选 5%：随机 | 选 5%：差 |", "|---|---|---|---|---|---|---|---|"]
        a_all = a[a.fold.isin(folds20)]
        for mp in MC.MAPS:
            for m, lab in SERIES:
                cells = []
                for f in (0.01, 0.05):
                    x = a_all[(a_all["map"] == mp) & (a_all.frac == f)]; y = a20[(a20["map"] == mp) & (a20.frac == f)]
                    ox = (x[m] > x.total).groupby(x.fold).mean(); oy = (y[m] > y.total).groupby(y.fold).mean(); d = (ox - oy).dropna()
                    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                    cells += [r3(ox.mean()), r3(oy.mean()), f"{'+' if d.mean() >= 0 else '−'}{r3(abs(d.mean()))} (±{r3(ci)})"]
                    out.append(dict(table="D7", map=mp, frac=f, series=lab, risk_map=ox.mean(), risk_random=oy.mean(),
                                    paired=d.mean(), paired_ci=ci, n_maps=len(d)))
                L.append(f"| {MC.TITLE[mp]} | {lab} | " + " | ".join(cells) + " |")
    open(os.path.join(RES, "RESULTS_CARBON_DOMAIN.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    pd.DataFrame(out).to_csv(os.path.join(RES, "carbon_domain_summary.csv"), index=False, float_format="%.4f")
    print("\n".join(L))


if __name__ == "__main__":
    main()
