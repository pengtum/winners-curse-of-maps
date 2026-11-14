"""VCS 10% waiver at the map-selected plots (post hoc, added 2026-11-02 after a simulated review).
The frozen endpoints judge the survey-calibrated estimate under the VCS rule, waiver included. This table reports
how often the waiver applied and the risk if the 0.4307 SE deduction is always applied (U > 100% still not creditable).
Reads the stored estimate and SE columns; nothing is rerun.
    ~/miniforge3/envs/favela/bin/python experiments/make_carbon_waiver.py
writes results/fia_db/RESULTS_CARBON_WAIVER.md
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "fia_db")
Z_VCS, RULE, EST, SE = 0.4307, "VCS model-assisted", "ma", "se_ma"
COLS = ["fold", "audit", "frac", "total", RULE, EST, SE, "ours: VCS formula + waiver", "ours: VCS formula, sel.-aware"]
SOURCES = [("Site attributes", "carbon_site.csv.gz", "carbon_site_random.csv.gz"),
           ("Site attributes, 20-map control", "carbon_site.csv.gz", "carbon_random20/carbon_site_random.csv.gz"),
           ("Site + canopy height", "carbon_siteheight.csv.gz", "carbon_siteheight_random.csv.gz"),
           ("Full ground inventory", "carbon_full.csv.gz", "carbon_full_random.csv.gz"),
           ("ESA CCI Biomass v7.0", "product/carbon_product_win.csv.gz", "product/carbon_product_win_random.csv.gz")]


def load(f):
    d = pd.read_csv(os.path.join(RES, f), usecols=COLS)
    u = 1.6449 * d[SE] / d[EST].where(d[EST] > 0)
    d["U"] = u.fillna(np.inf)
    d["waived"] = d.U <= 0.10
    d["always"] = np.where(d.U > 1, 0.0, d[EST] - Z_VCS * d[SE]).clip(min=0)
    return d


def clustered(x, by):
    m = x.groupby(by).mean()
    return m.mean(), 1.96 * m.std(ddof=1) / np.sqrt(len(m))


def fmt(p):
    return f"{p[0]:.3f} (±{p[1]:.3f})"


def main():
    L = ["# VCS 豁免与总是扣减（事后补充，2026-11-02）", "",
         "由 `experiments/make_carbon_waiver.py` 从已存的估计值与标准误算出，没有新运行。冻结的判定（R1、R-P1）用含豁免的 VCS 规则，不变。"
         "U = 1.6449 SE / 估计值；U ≤ 0.10 时规则不扣减。'总是扣减' 指 U ≤ 1 时一律减 0.4307 SE。区间为按地图（或子集）聚类的 95% 区间。", "",
         "| 地图 | 审计 | 选 | 选点 | 豁免触发 | 中位 U | VCS 规则风险 | 总是扣减风险 | 选择感知（含豁免 / 不含） |", "|---|---|---|---|---|---|---|---|---|"]
    P = ["", "配对增量（地图选点 − 随机选点，同一批地图；审计 0.5）：", "",
         "| 地图 | 选 | 地图数 | VCS 规则 | 总是扣减 |", "|---|---|---|---|---|"]
    for name, fm, fr in SOURCES:
        dm, dr = load(fm), load(fr)
        for a, fr_ in [(0.5, 0.01), (0.5, 0.05), (0.2, 0.01), (0.8, 0.01)]:
            for lab, d in (("地图", dm), ("随机", dr)):
                if "20-map" in name and lab == "地图":
                    continue
                x = d[(d.audit == a) & (d.frac == fr_)]
                sa_w = (x["ours: VCS formula + waiver"] > x.total).mean()
                sa = (x["ours: VCS formula, sel.-aware"] > x.total).mean()
                L.append(f"| {name} | {a} | {fr_:.0%} | {lab} | {x.waived.mean():.3f} | {x.U.median():.3f} | "
                         f"{fmt(clustered(x[RULE] > x.total, x.fold))} | {fmt(clustered(x.always > x.total, x.fold))} | {sa_w:.3f} / {sa:.3f} |")
        for fr_ in (0.01, 0.05):
            xm, xr = dm[(dm.audit == 0.5) & (dm.frac == fr_)], dr[(dr.audit == 0.5) & (dr.frac == fr_)]
            out = []
            for c in (RULE, "always"):
                g = ((xm[c] > xm.total).groupby(xm.fold).mean() - (xr[c] > xr.total).groupby(xr.fold).mean()).dropna()
                out.append(f"{g.mean():+.3f} (±{1.96 * g.std(ddof=1) / np.sqrt(len(g)):.3f})")
            P.append(f"| {name} | {fr_:.0%} | {len(g)} | {out[0]} | {out[1]} |")
    with open(os.path.join(RES, "RESULTS_CARBON_WAIVER.md"), "w") as f:
        f.write("\n".join(L + P) + "\n")


if __name__ == "__main__":
    main()
