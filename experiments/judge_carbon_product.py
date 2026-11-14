"""Endpoints and judgement rules of paper/BIOMASS_PRODUCT_DESIGN.md (sections 5 and 7, frozen 2026-10-31) on the
operational-product carbon runs (results/fia_db/product/, experiments/run_carbon_product.py).

Primary cell: window means ('win'), audit share 0.5, highest 1%, VCS.
  E1  survey-calibrated risk (map-clustered 95% interval, clusters = the 20 subsets) and its paired increment over
      random selection (per subset, as Table D3 of make_carbon_domain.py);
  E2  selection-aware risk (formula without the waiver).
  R-P1 positive: E1 > 1/3, its interval excludes 1/3, and the paired increment's interval excludes 0; else R-P0.
  R-P2: E2 within 0.31-0.35.
Secondary (SI only): error layer under map and random selection, other cells, ART, credited shares, tonnes per 100
credits (from biomass sums), the centre-pixel variant. Unrounded pooled estimates decide.
Writes results/fia_db/product/JUDGEMENT_PRODUCT.md and results/fia_db/product/product_summary.csv.
    ~/miniforge3/envs/favela/bin/python -B experiments/judge_carbon_product.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
import make_carbon as MC                                              # noqa: E402

RES = os.path.join(DS.FIA_RES, "product")
MA, SA, EL, DOM = "VCS model-assisted", "ours: VCS formula, sel.-aware", "VCS map", "VCS domain10"
SERIES = {"point": "Map total, no deduction", EL: "Protocol rule, product error layer",
          "VCS map (corr.)": "Protocol rule, error layer fully correlated", MA: "Protocol rule, survey-calibrated",
          DOM: "Protocol rule, ten-stratum domain", "VCS domain10, formula": "Domain, formula without waiver",
          SA: "Selection-aware (ours)", "ours: exact, risk VCS": "Exact bound (ours)",
          "ART model-assisted": "ART, survey-calibrated", "ours: ART formula, sel.-aware": "ART, selection-aware (ours)"}


def load(v, rnd=False):
    return pd.read_csv(os.path.join(RES, f"carbon_product_{v}{'_random' if rnd else ''}.csv.gz"))


def cell(d, audit=0.5, frac=0.01):
    return d[(d.audit == audit) & (d.frac == frac)]


def risk(d, m):
    r = MC.risk(d.assign(k=0), m, ["k"]).iloc[0]
    return float(r.risk), float(r.ci)


def paired(x, y, m):
    ox = (x[m] > x.total).groupby(x.fold).mean(); oy = (y[m] > y.total).groupby(y.fold).mean()
    d = (ox - oy).dropna()
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d)), len(d)


def per100(d, m):
    c, t = d[m], d.total
    return 100 * np.maximum(c - t, 0).mean() / c.mean(), 100 * np.maximum(t - c, 0).mean() / c.mean()


def main():
    L = ["# 业务化生物量产品检验：判读（`paper/BIOMASS_PRODUCT_DESIGN.md` 第 5、7 节，冻结 2026-10-31）", "",
         "由 `experiments/judge_carbon_product.py` 生成。地图为 ESA CCI Biomass v7.0（2020），真值为 FIA 地上活立木生物量；"
         "20 个 6,000 样地子集 × 100 次审计；区间为按子集聚类的 95% 区间。判定用不取整的点估计。", ""]
    rows = []
    for v in ("win", "ctr"):
        if not os.path.exists(os.path.join(RES, f"carbon_product_{v}.csv.gz")):
            continue
        m, r = load(v), load(v, True)
        for a in (0.2, 0.5, 0.8):
            for f in sorted(m.frac.unique()):
                x, y = cell(m, a, f), cell(r, a, f)
                for s, lab in SERIES.items():
                    rr, ci = risk(x, s); rr0, ci0 = risk(y, s); dd, dci, _ = paired(x, y, s)
                    rows.append(dict(variant=v, audit=a, frac=f, series=lab, risk=rr, ci=ci, risk_random=rr0,
                                     ci_random=ci0, paired=dd, paired_ci=dci,
                                     credited_share=x[s].mean() / x.total.mean(), point_share=x.point.mean() / x.total.mean()))
    S = pd.DataFrame(rows); S.to_csv(os.path.join(RES, "product_summary.csv"), index=False, float_format="%.6g")
    m, r = cell(load("win")), cell(load("win", True))
    e1, e1ci = risk(m, MA); d1, d1ci, nd = paired(m, r, MA); e2, e2ci = risk(m, SA)
    p1 = e1 > 1 / 3 and e1 - e1ci > 1 / 3 and d1 - d1ci > 0
    p2 = 0.31 <= e2 <= 0.35
    r2 = m.groupby("fold").r2.first()
    L += ["## 主终点（窗口均值、审计 0.5、最高 1%、VCS）", "",
          f"- 产品在子集上的 R²：中位 {r2.median():.3f}（{r2.min():.3f}–{r2.max():.3f}）。地图总量 / 真值：{m.point.mean() / m.total.mean():.3f}。",
          f"- **E1** 调查校正估计风险 {e1:.4f}（±{e1ci:.3f}）；随机选点 {risk(r, MA)[0]:.4f}（±{risk(r, MA)[1]:.3f}）；"
          f"配对增量 {d1:+.4f}（±{d1ci:.3f}，{nd} 个子集）。",
          f"- **E2** 选择感知估计风险 {e2:.4f}（±{e2ci:.3f}）。",
          f"- 每 100 个信用：调查校正多发 {per100(m, MA)[0]:.1f}、少发 {per100(m, MA)[1]:.1f}；选择感知多发 {per100(m, SA)[0]:.1f}、少发 {per100(m, SA)[1]:.1f}（t CO₂e，按生物量比值）。",
          "",
          f"**R-P1：{'阳性' if p1 else '不可检出（R-P0）'}。** **R-P2：{'满足（0.31–0.35）' if p2 else '不满足'}。**", ""]
    L += ["## 次要终点（只作描述）", "", "| 口径 | 审计 | 选 | 序列 | 地图选点风险 | 随机选点风险 | 配对增量 | 认证/真值 |", "|---|---|---|---|---|---|---|---|"]
    for t in S[(S.audit == 0.5) & S.frac.isin([0.01, 0.05])].itertuples():
        L.append(f"| {t.variant} | {t.audit} | {t.frac:.0%} | {t.series} | {t.risk:.3f} (±{t.ci:.3f}) | {t.risk_random:.3f} | "
                 f"{t.paired:+.3f} (±{t.paired_ci:.3f}) | {t.credited_share:.3f} |")
    open(os.path.join(RES, "JUDGEMENT_PRODUCT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
