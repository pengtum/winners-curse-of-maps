"""Judgement rules R1 and R2 of paper/FIA_REBUILD_DESIGN.md section 6 (frozen 2026-10-30) on the fia_db reruns.

R1 (abstract sentences 6-7), cell: site map, audit share 0.5, highest 1%, VCS family.
  1  random-selection control: survey-calibrated risk within 1/3 +- 0.03;
  2  map-based selection: survey-calibrated risk above 1/3, map-clustered 95% interval excluding 1/3, and the paired
     increment over random selection (per map, as Table D3 of make_carbon_domain.py) with its 95% interval excluding 0;
  3  selection-aware estimate between 0.31 and 0.35.
  The random control is the one the paper reports ("a random-selection control on ten of the maps", folds 0-9,
  results/fia_db/carbon_site_random.csv.gz); the 20-map control (results/fia_db/carbon_random20/) is printed beside it.
R2 (abstract sentences 2 and 4): FIA split CP and normalized CP at the policy 5% (pooled sum n_cov / sum n_sel, unrounded).
Risks and intervals use make_carbon.risk (fold = map clusters). Writes results/fia_db/JUDGEMENT_R1_R2.md.
    ~/miniforge3/envs/favela/bin/python -B experiments/judge_fia_db.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
import make_carbon as MC                                              # noqa: E402

assert DS.FIA == "fia_db", DS.FIA
MA, SA = "VCS model-assisted", "ours: VCS formula, sel.-aware"


def cell(df, mp="site", audit=0.5, frac=0.01):
    return df[(df["map"] == mp) & (df.audit == audit) & (df.frac == frac)]


def paired(x, y, m):
    ox = (x[m] > x.total).groupby(x.fold).mean(); oy = (y[m] > y.total).groupby(y.fold).mean()
    d = (ox - oy).dropna()
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d)), len(d)


def main():
    L = ["# FIA 重建：判读规则 R1、R2（`paper/FIA_REBUILD_DESIGN.md` 第 6 节，冻结 2026-10-30）", "",
         "由 `experiments/judge_fia_db.py` 生成；数据 `fia_db`（FIADB 9.5）。风险 = 认证量超过被选样地真实总量的比例，"
         "区间为按地图聚类的 95% 区间。判定用不取整的点估计。", ""]
    mp_ = cell(MC.load())
    ok = {}
    for tag, folder, label in (("", MC.RES, "10 张地图（论文所用，主判定）"),
                               ("_random", os.path.join(MC.RES, "carbon_random20"), "20 张地图（补充，不判定）")):
        f = os.path.join(folder, "carbon_site_random.csv.gz")
        rnd = cell(pd.read_csv(f))
        folds = sorted(rnd.fold.unique())
        x = mp_[mp_.fold.isin(folds)]
        rr = MC.risk(rnd.assign(k=0), MA, ["k"]).iloc[0]
        rm = MC.risk(mp_.assign(k=0), MA, ["k"]).iloc[0]                     # map-based: all 20 maps
        rmf = MC.risk(x.assign(k=0), MA, ["k"]).iloc[0]                       # map-based on the same maps
        ds, dci, nd = paired(x, rnd, MA)
        rs = MC.risk(mp_.assign(k=0), SA, ["k"]).iloc[0]
        c1 = abs(rr.risk - 1 / 3) <= 0.03
        c2 = rm.risk > 1 / 3 and rm.risk - rm.ci > 1 / 3 and ds - dci > 0
        c3 = 0.31 <= rs.risk <= 0.35
        if tag == "":
            ok = dict(c1=c1, c2=c2, c3=c3)
        L += [f"## R1　碳汇（site 地图、审计占比 0.5、最高 1%、VCS）：随机对照 {label}", "",
              f"- ① 随机选点（{len(folds)} 张地图，折 {folds[0]}–{folds[-1]}）下调查校正估计的风险 {rr.risk:.4f}（±{rr.ci:.3f}）；"
              f"要求在 1/3 ± 0.03（0.3033–0.3633）以内：**{'满足' if c1 else '不满足'}**。",
              f"- ② 地图选点（20 张地图）下调查校正估计的风险 {rm.risk:.4f}（±{rm.ci:.3f}，区间下端 {rm.risk - rm.ci:.4f}）；"
              f"同一批 {nd} 张地图上为 {rmf.risk:.4f}；配对增量（地图选点 − 随机选点）{ds:+.4f}（±{dci:.3f}，下端 {ds - dci:+.4f}）；"
              f"要求风险高于 1/3、区间不含 1/3、配对增量区间不含 0：**{'满足' if c2 else '不满足'}**。",
              f"- ③ 选择感知估计的风险 {rs.risk:.4f}（±{rs.ci:.3f}）；要求在 0.31–0.35：**{'满足' if c3 else '不满足'}**。", ""]
    L += [f"**R1 判定（以 10 张地图的随机对照为准）：{'三条都满足，保留摘要第 6–7 句、只换数字' if all(ok.values()) else '有条件不满足，按实际结果改写第 6–7 句'}**"
          f"（① {'✓' if ok['c1'] else '✗'}，② {'✓' if ok['c2'] else '✗'}，③ {'✓' if ok['c3'] else '✗'}）。", ""]
    # R2
    d = pd.read_csv(os.path.join(ROOT, "results", f"main_{DS.FIA}.csv")); d = d[d.rule == "top5"]
    g = d.groupby("method"); cov = g.n_cov.sum() / g.n_sel.sum()
    scp, ncp = cov["SCP |res|"], cov["SCP stud"]
    L += ["## R2　覆盖审计（FIA，被选 5%）", "",
          f"- split CP {scp:.5f}：{'仍低于 0.85，摘要 eight of 11 与 four of six 不变' if scp < 0.85 else '升到 0.85 及以上，两处按实际个数改写'}。",
          f"- normalized CP {ncp:.5f}：{'升到 0.85 及以上，第 4 句 spread-scaling left two 改为 one，Results 第 3 小节与 Discussion 第 2 段的 0.82 一并改' if ncp >= 0.85 else '仍低于 0.85，只换数字'}。",
          f"- 参照：JOMI {cov['JOMI stud']:.5f}。", ""]
    out = os.path.join(MC.RES, "JUDGEMENT_R1_R2.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
