"""Summaries of the GeoCERT pilots -> results/PILOT_GeoCERT.md

    ~/miniforge3/envs/favela/bin/python experiments/make_pilot.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import data as DS  # noqa: E402
from make_figs import md  # noqa: E402

RES = os.path.join(ROOT, "results")
ORDER2 = ["SCP stud", "JOMI", "JOMI|state", "SC-GeoCCP coarse", "SC-GeoCCP l1=1.0", "SC-GeoCCP l1=1.5"]
ORDER3 = ["Point sum mu", "Model iid CLT", "Cal iid CLT (no selection)", "Ref-set SRSWOR CLT",
          "SC-Total exact (raw)", "SC-Total exact (stud)", "Sum unit lower (SCP stud)", "Sum unit lower (JOMI stud)"]


def l2(name):
    f = os.path.join(RES, f"pilot_l2_{name}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    rows = []
    for mth in [m for m in ORDER2 if m in set(d.method)]:
        x = d[d.method == mth]
        g = x.groupby("state").agg(n=("cov", "size"), c=("cov", "sum"), fin=("finite", "mean"))
        g = g[g.n >= 50]; g["cv"] = g.c / g.n
        if len(g) == 0:
            continue
        sims = np.random.default_rng(0).binomial(g.n.values[None, :], 0.9, (4000, len(g))) / g.n.values[None, :]
        rows.append({"method": mth, "cov|sel": x["cov"].mean(), "+inf share": 1 - x.finite.mean(),
                     "median half-width (finite)": x.hw[x.finite].median(),
                     "states (n>=50)": len(g), "worst state": g.cv.min(),
                     "states <0.85": int((g.cv < 0.85).sum()),
                     "worst-state +inf": 1 - g.fin.min(),
                     "null worst (exact 0.9)": sims.min(1).mean()})
    return pd.DataFrame(rows), d.groupby(["fold", "split"]).ngroups


def l3(name, part="l3"):
    f = os.path.join(RES, f"pilot_{part}_{name}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    sd = DS.load(name)["y"].std()
    rows = []
    for frac in sorted(d.frac.unique()):
        for mth in ORDER3:
            x = d[(d.frac == frac) & (d.method == mth)]
            c = x.covered.mean()
            rows.append({"K/m": f"{frac:.0%}", "K": int(x.K.iloc[0]), "method": mth,
                         "P(total >= bound)": c, "MC se": np.sqrt(c * (1 - c) / len(x)),
                         "median (bound - total)/K, in sd(y)": float(np.median((x.lb - x.total) / x.K / sd)),
                         "median bound/total": float(np.median(x.lb / x.total)) if (x.total > 0).all() else np.nan})
    return pd.DataFrame(rows), d.groupby(["fold", "split"]).ngroups


def main():
    L = ["# GeoCERT 快速 pilot 结果（自动生成：experiments/make_pilot.py）\n",
         "设计：3 个固定训练折 × 多次随机校准/测试划分（保证正是对划分取的）。α = 0.1（层 2），δ = 0.05（层 3）。\n\n",
         "## 层 2　SC-GeoCCP：配额规则下的逐区域覆盖\n",
         "规则：FIA 州配额（top，按州单元数比例配额，共 5%）；PovertyMap（无影像版）国配额（bottom，援助给预测最穷的簇）。"
         "SC-GeoCCP 的基为 截距 + 粗区域（州/国质心 k-means：FIA 3 个、非洲 5 个，不惩罚）[+ 州/国指示，L1 惩罚]，"
         "在每个被选单元的 JOMI 参考集上做增广分位数回归（GCC）。`worst state` 为合并全部划分后被选次数 ≥ 50 的州/国中覆盖最低者，"
         "`null worst` 为每州覆盖恰为 0.9 时二项抽样下最差州的期望值。\n\n"]
    for nm in ["fia", "poverty_lab"]:
        r = l2(nm)
        if r is not None:
            L += [f"### {DS.LABEL[nm]}（{r[1]} 次划分）\n\n", md(r[0]), "\n"]
    L += ["## 层 3　SC-Total：被选集合总量的下界（δ = 0.05）\n",
          "FIA：top-K 碳汇样地，认证总生物量不得超过真实总量（下界）。PovertyMap（无影像版）：bottom-K 援助，"
          "预算充足 ⟺ 被选簇财富之和的下界有效。支撑下限：FIA 用 Y ≥ 0，财富指数用训练折最小值（显式假设）。"
          "`P(total >= bound)` 应 ≥ 0.95。\n\n"]
    for nm in ["fia", "poverty_lab"]:
        r = l3(nm)
        if r is not None:
            L += [f"### {DS.LABEL[nm]}（{r[1]} 次划分；SC-Total 用 Monte Carlo 包络，近似有效）\n\n", md(r[0]), "\n"]
    L += ["## 层 3（精确版）　SC-Total 改用精确 DP 包络（`excp_select/totals.py::envelope_exact`，给定池时违反概率恰好 ≤ δ）\n\n"
          "其余方法与上表相同（同一种子）。PovertyMap 的支撑下限取训练折最小值，为显式假设。\n\n"]
    for nm in ["fia", "poverty_lab"]:
        r = l3(nm, "l3x")
        if r is not None:
            L += [f"### {DS.LABEL[nm]}（{r[1]} 次划分，精确包络）\n\n", md(r[0]), "\n"]
    open(os.path.join(RES, "PILOT_GeoCERT.md"), "w").write("".join(L))
    print("".join(L))


if __name__ == "__main__":
    main()
