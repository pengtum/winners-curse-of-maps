"""WP2 summaries: results/dec_<target>.csv -> results/RESULTS_DECISIONS.md, figs/nc_fig_decisions.png

    ~/miniforge3/envs/favela/bin/python experiments/make_decisions.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
from make_figs import md, INK, INK2, GRID  # noqa: E402

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
METH = ["point", "Normalized CP", "GeoCP", "QRF", "Bootstrap", "JOMI", "SC-GeoCCP"]
STY = {"point": ("#0b0b0b", "*", True), "Normalized CP": ("#7d7b74", "s", True), "GeoCP": ("#eb6834", "D", True),
       "QRF": ("#1baf7a", "o", False), "Bootstrap": ("#1baf7a", "s", False), "JOMI": ("#2a78d6", "o", True),
       "SC-GeoCCP": ("#6250d6", "P", True)}
FIA = os.environ.get("QW2_FIA", "fia_db")               # == experiments/data.py FIA
LAB = {FIA: "FIA biomass (ground plots)", "m3leo_resnet": "Biomass ESA-CCI, ResNet", "m3leo_dofa": "Biomass ESA-CCI, DOFA",
       "poverty_lab": "DHS wealth, no imagery", "poverty": "DHS wealth (PovertyMap)", "poverty_resnet": "DHS wealth, ResNet",
       "poverty_dofa": "DHS wealth, DOFA", "floods_resnet": "Floods, ResNet", "floods_dofa": "Floods, DOFA"}


def load(t):
    f = os.path.join(RES, f"dec_{t}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def carbon(ts):
    rows = []
    for t in ts:
        d = load(t)
        if d is None:
            continue
        for (rn, mn), x in d.groupby(["rule", "method"]):
            rows.append({"data": LAB[t], "rule": rn, "method": mn,
                         "over-credit rate (plots)": x.n_over.sum() / x.n_sel.sum(),
                         "over-credited share of credits": x.sum_over.sum() / x.sum_credit.sum(),
                         "true carbon credited": x.sum_credited_true.sum() / x.sumY.sum(),
                         "P(total credit > true total)": x.total_overclaim.mean(),
                         "coverage": x.n_cov.sum() / x.n_sel.sum() if "n_cov" in x and x.n_cov.notna().any() else np.nan})
    return pd.DataFrame(rows)


def aid(ts):
    rows = []
    for t in ts:
        d = load(t)
        if d is None:
            continue
        for (rn, mn), x in d.groupby(["rule", "method"]):
            nf = x.n_fund.sum(); nd = x.n_drop.sum()
            rows.append({"data": LAB[t], "rule": rn, "method": mn,
                         "eligible among selected": x.n_elig.sum() / x.n_sel.sum(),
                         "fund-now share": nf / x.n_sel.sum(),
                         "inclusion error (fund-now)": x.n_fund_inelig.sum() / nf if nf else np.nan,
                         "drop share": nd / x.n_sel.sum(),
                         "exclusion error (drop)": x.n_drop_elig.sum() / nd if nd else np.nan,
                         "survey share": x.n_survey.sum() / x.n_sel.sum(),
                         "eligible reached now": x.n_fund_elig.sum() / x.n_elig.sum()})
    return pd.DataFrame(rows)


def floods(ts):
    rows = []
    for t in ts:
        d = load(t)
        if d is None:
            continue
        for (rn, mn), x in d.groupby(["rule", "method"]):
            rows.append({"data": LAB[t], "rule": rn, "method": mn,
                         "tiles short": x.n_short.sum() / x.n_sel.sum(),
                         "shortage share of true need": x.sum_short.sum() / x.sumY.sum(),
                         "supplies / true need": x.sum_supply.sum() / x.sumY.sum(),
                         "coverage": x.n_cov.sum() / x.n_sel.sum() if x.n_cov.notna().any() else np.nan})
    return pd.DataFrame(rows)


def cfbh_table():
    """cfBH inside the selected set vs interval triage (results/cfbh_<target>.csv)."""
    rows = []
    for tg in ["poverty_lab", "poverty", "poverty_resnet", "poverty_dofa"]:
        f = os.path.join(RES, f"cfbh_{tg}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        for (rn, mn), x in d.groupby(["rule", "method"], sort=False):
            rows.append({"data": LAB[tg], "rule": rn, "method": mn, "FDR (mean FDP)": x.fdp.mean(),
                         "funded now (mean)": x.n_fund.mean(), "eligible reached": x.n_fund_elig.sum() / x.n_elig.sum()})
    if not rows:
        return []
    return ["## 援助：在被选村中用 cfBH 确认合格（FDR ≤ q）vs 区间三分类（U < c）\n\n",
            "cfBH 的校准集为 JOMI 参考集；bottom 5% 下参考集为公共集合（结构引理）；配额下每村参考集不同，为启发式版本。"
            "FDR = 100 次划分上 FDP 的平均。\n\n", md(pd.DataFrame(rows)), "\n"]


def order(df):
    df = df.copy(); df["_m"] = df.method.map({m: i for i, m in enumerate(METH)})
    return df.sort_values(["data", "rule", "_m"]).drop(columns="_m")


def main():
    C = carbon([FIA, "m3leo_resnet", "m3leo_dofa"])
    A = aid(["poverty_lab", "poverty", "poverty_resnet", "poverty_dofa"])
    Fl = floods(["floods_resnet", "floods_dofa"])
    L = ["# WP2 决策后果（自动生成：experiments/make_decisions.py）\n\n",
         "K = 测试单元的 5%（援助另有按国配额）。point = 按点预测选点；cautious = 按训练折尺度构造的可信界选点"
         "（碳汇 μ−1.645σ̂，援助 μ+1.645σ̂，洪水 μ+1.645σ̂）。区间方法都在被选单元上使用。\n\n",
         "## 碳汇：按下界发放信用（point = 按 μ 发放）\n\n", md(order(C)), "\n",
         "## 援助：U ≤ c 直接资助，L > c 不予资助，其余补做调查（point：μ ≤ c 即资助）；c = 训练折该国财富第一四分位数\n\n",
         md(order(A)), "\n",
         "## 洪水：物资按上界准备（point = 按 μ）\n\n", md(order(Fl)), "\n"]
    L += cfbh_table()
    open(os.path.join(RES, "RESULTS_DECISIONS.md"), "w").write("".join(L))
    fig(C, A, Fl)
    print("".join(L)[:3000])


def fig(C, A, Fl):
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.4))
    # (a) carbon, FIA point 5%
    ax = axs[0]; x = C[(C.data == LAB[FIA]) & (C.rule == "point 5%")]
    for _, r in x.iterrows():
        c, mk, fill = STY[r.method]
        ax.scatter(r["true carbon credited"], r["over-credit rate (plots)"], s=60, marker=mk,
                   color=c if fill else "white", edgecolor=c, lw=1.4, zorder=3, label=r.method)
    ax.axhline(0.05, color=INK2, lw=1, ls="--", zorder=0)
    ax.set_xlabel("share of true carbon credited"); ax.set_ylabel("plots over-credited")
    ax.set_title("a  Carbon credits (FIA, top 5%)", fontsize=9, loc="left")
    # (b) aid triage, no-imagery DHS, point quota
    ax = axs[1]; x = A[(A.data == LAB["poverty_lab"]) & (A.rule == "point quota")]
    ms = [m for m in METH if m in set(x.method)]
    vals = [x[x.method == m]["inclusion error (fund-now)"].values[0] for m in ms]
    share = [x[x.method == m]["fund-now share"].values[0] for m in ms]
    for i, (m, v, s_) in enumerate(zip(ms, vals, share)):
        c, mk, fill = STY[m]
        ax.barh(i, v, color=c if fill else "white", edgecolor=c, lw=1.2, height=0.6)
        ax.text(v + 0.005, i, f"{s_:.0%} funded now", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=8); ax.invert_yaxis()
    ax.axvline(0.05, color=INK2, lw=1, ls="--")
    ax.set_xlabel("ineligible among villages funded now")
    ax.set_title("b  Aid triage (DHS, country quotas)", fontsize=9, loc="left")
    # (c) floods, DOFA point 5%
    ax = axs[2]; x = Fl[(Fl.data == LAB["floods_dofa"]) & (Fl.rule == "point 5%")]
    for _, r in x.iterrows():
        c, mk, fill = STY[r.method]
        ax.scatter(r["supplies / true need"], r["shortage share of true need"], s=60, marker=mk,
                   color=c if fill else "white", edgecolor=c, lw=1.4, zorder=3)
    ax.set_xlabel("supplies prepared / true need"); ax.set_ylabel("shortage (share of true need)")
    ax.set_title("c  Flood relief (DOFA GeoFM, top 5%)", fontsize=9, loc="left")
    for a in axs:
        a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
    axs[0].legend(frameon=False, fontsize=7, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "nc_fig_decisions.png")); plt.close(fig)


if __name__ == "__main__":
    main()
