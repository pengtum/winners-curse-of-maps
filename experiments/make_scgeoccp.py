"""Tables + figure for SC-GeoCCP (results/scg_*.csv) -> results/RESULTS_SCGeoCCP.md, figs/fig6_scgeoccp.png

    ~/miniforge3/envs/favela/bin/python experiments/make_scgeoccp.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import data as DS  # noqa: E402
from make_figs import md, INK, INK2, GRID  # noqa: E402

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
METH = ["SCP stud", "ExCP-default", "JOMI stud", "JOMI|state stud", "SC-GeoCCP", "SC-GeoCCP (l1=0.95)"]
FIX = "SC-GeoCCP (l1=0.95)"
COL = {"SCP stud": ("#7d7b74", "s", True), "ExCP-default": ("#eb6834", "D", True),
       "JOMI stud": ("#2a78d6", "o", False), "JOMI|state stud": ("#1baf7a", "^", True),
       "SC-GeoCCP": ("#6250d6", "P", True), "SC-GeoCCP (l1=0.95)": ("#6250d6", "P", False)}
NMIN = 50


def quota_rule(nm):
    return "quota_bottom" if nm.startswith("poverty") else "quota"


def worst(G, rule, mth, gtype, nmin=NMIN):
    g = G[(G.rule == rule) & (G.method == mth) & (G.gtype == gtype) & (G.n >= nmin)]
    if g.empty:
        return np.nan, np.nan, 0, np.nan, np.nan
    cv = g.c / g.n
    sims = np.random.default_rng(0).binomial(g.n.values[None, :], 0.9, (4000, len(g))) / g.n.values[None, :]
    k = cv.values.argmin()
    return cv.min(), (g.inf / g.n).values[k], int((cv < 0.85).sum()), sims.min(1).mean(), len(g)


def main():
    names = [n for n in DS.DISPLAY if os.path.exists(os.path.join(RES, f"scg_main_{n}.csv"))]
    def load(kind, n):
        x = pd.read_csv(os.path.join(RES, f"scg_{kind}_{n}.csv"))
        f = os.path.join(RES, f"scg_{kind}_{n}_l1fix.csv")
        if os.path.exists(f):                      # "smallest l1 > tau" rule, SC-GeoCCP only
            y = pd.read_csv(f); y["method"] = FIX
            x = pd.concat([x, y], ignore_index=True)
        return x
    M = {n: load("main", n) for n in names}
    G = {n: load("groups", n) for n in names}
    L = ["# SC-GeoCCP 结果（自动生成：experiments/make_scgeoccp.py）\n",
         "α = 0.1；与 run_main.py 相同的 100 次随机划分（μ、σ 与被选单元完全一致）。SC-GeoCCP 的基为 截距（不惩罚）+ 3 个尺度的高斯 bump"
         "+ 州/国指示（L1 惩罚，l1 在训练折上选）。逐组统计合并 100 次划分，只计被选 ≥ 50 次的组；"
         "`null worst` 为每组覆盖恰为 0.9 时二项抽样下最差组的期望值。* 为 PLACES 模型输出。\n\n"]
    # ---- Table 6: quota rules, per-state
    rows = []
    for n in names:
        r = quota_rule(n); d = M[n][M[n].rule == r]
        for mth in METH:
            x = d[d.method == mth]
            if x.empty:
                continue
            w, wi, nb, nul, ng = worst(G[n], r, mth, "state")
            rows.append({"data": DS.LABEL[n], "rule": r, "method": mth,
                         "cov|sel": x.n_cov.sum() / x.n_sel.sum(), "+inf": x.n_inf.sum() / x.n_sel.sum(),
                         "mean half-width (finite)": x.hw_mean_fin.mean(), "states": ng, "worst state": w,
                         "worst-state +inf": wi, "states <0.85": nb, "null worst": nul})
    T6 = pd.DataFrame(rows)
    L += ["## 表 6　配额规则：逐州/逐国的选择条件覆盖\n\n", md(T6), "\n"]
    # ---- Table 7: all rules, JOMI vs SC-GeoCCP (marginal + worst state + worst cell)
    rows = []
    for n in names:
        for r in M[n].rule.unique():
            d = M[n][M[n].rule == r]
            row = {"data": DS.LABEL[n], "rule": r, "|S|": int(d.n_sel.mean())}
            for mth, tag in [("ExCP-default", "ExCP"), ("JOMI stud", "JOMI"), ("SC-GeoCCP", "SCG"), (FIX, "SCG.95")]:
                x = d[d.method == mth]
                if x.empty:
                    continue
                row[f"cov {tag}"] = x.n_cov.sum() / x.n_sel.sum()
                row[f"worst cell {tag}"] = worst(G[n], r, mth, "cell")[0]
                row[f"worst state {tag}"] = worst(G[n], r, mth, "state")[0]
            j = d[d.method == "JOMI stud"]; s = d[d.method == "SC-GeoCCP"]
            row["+inf SCG"] = s.n_inf.sum() / s.n_sel.sum(); row["+inf JOMI"] = j.n_inf.sum() / j.n_sel.sum()
            row["width SCG/JOMI"] = s.hw_mean_fin.mean() / j.hw_mean_fin.mean()
            s95 = d[d.method == FIX]
            if len(s95):
                row["width SCG.95/JOMI"] = s95.hw_mean_fin.mean() / j.hw_mean_fin.mean()
                row["+inf SCG.95"] = s95.n_inf.sum() / s95.n_sel.sum()
            rows.append(row)
    T7 = pd.DataFrame(rows)
    L += ["## 表 7　全部规则：ExCP 默认 vs JOMI vs SC-GeoCCP（被选覆盖、最差格网、最差州）\n\n", md(T7), "\n"]
    ok = T7[T7["+inf SCG"] < 0.05]
    L += [f"SC-GeoCCP 被选覆盖（+∞ < 5% 的 {len(ok)} 格）：最小 {ok['cov SCG'].min():.3f}，中位 {ok['cov SCG'].median():.3f}。"
          f"最差州：SC-GeoCCP 优于 JOMI 的格 {(T7['worst state SCG'] > T7['worst state JOMI'] + 0.005).sum()} / {T7['worst state SCG'].notna().sum()}，"
          f"劣于 JOMI 的格 {(T7['worst state SCG'] < T7['worst state JOMI'] - 0.005).sum()}；"
          f"最差格网：优于 {(T7['worst cell SCG'] > T7['worst cell JOMI'] + 0.005).sum()} / {T7['worst cell SCG'].notna().sum()}，"
          f"劣于 {(T7['worst cell SCG'] < T7['worst cell JOMI'] - 0.005).sum()}。宽度比 SCG/JOMI 中位 {T7['width SCG/JOMI'].median():.3f}。\n\n"]
    if "worst state SCG.95" in T7:
        L += [f"l1 = 0.95 版：最差州优于 JOMI 的格 {(T7['worst state SCG.95'] > T7['worst state JOMI'] + 0.005).sum()} / {T7['worst state SCG.95'].notna().sum()}，"
              f"劣于 {(T7['worst state SCG.95'] < T7['worst state JOMI'] - 0.005).sum()}；最差格网优于 {(T7['worst cell SCG.95'] > T7['worst cell JOMI'] + 0.005).sum()}，"
              f"劣于 {(T7['worst cell SCG.95'] < T7['worst cell JOMI'] - 0.005).sum()}；被选覆盖最小 {T7['cov SCG.95'].min():.3f}；"
              f"宽度比 SCG.95/JOMI 中位 {T7['width SCG.95/JOMI'].median():.3f}（范围 {T7['width SCG.95/JOMI'].min():.2f}–{T7['width SCG.95/JOMI'].max():.2f}）；"
              f"+∞ 份额与 JOMI 相同的格 {(np.abs(T7['+inf SCG.95'] - T7['+inf JOMI']) < 1e-9).sum()} / {len(T7)}。\n\n"]
    # ---- Table 8: l1 chosen
    l1 = pd.DataFrame({DS.LABEL[n]: M[n][M[n].method != FIX].drop_duplicates("rep").l1.value_counts(normalize=True)
                       for n in names}).T.fillna(0)
    L += ["## 表 8　训练折上选出的 l1 分布（各数据集 100 次划分中的比例）\n\n", md(l1.reset_index().rename(columns={"index": "data"}), 2), "\n"]
    open(os.path.join(RES, "RESULTS_SCGeoCCP.md"), "w").write("".join(L))
    fig(T6, names)
    print("".join(L))


def fig(T6, names):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = np.arange(len(names))[::-1]; off = np.linspace(-0.3, 0.3, len(METH))
    for k, mth in enumerate(METH):
        c, mk, fill = COL[mth]
        v = [T6[(T6.data == DS.LABEL[n]) & (T6.method == mth)] for n in names]
        w = [x["worst state"].values[0] if len(x) else np.nan for x in v]
        fi = [x["+inf"].values[0] if len(x) else np.nan for x in v]
        axs[0].scatter(w, y + off[k], s=26, marker=mk, color=c if fill else "none", edgecolor=c, lw=1.2, label=mth, zorder=3)
        axs[1].scatter(fi, y + off[k], s=26, marker=mk, color=c if fill else "none", edgecolor=c, lw=1.2, zorder=3)
    nul = [T6[(T6.data == DS.LABEL[n])]["null worst"].values[0] for n in names]
    axs[0].scatter(nul, y, marker="|", s=160, color=INK2, label="exact 0.9 + sampling noise", zorder=2)
    axs[0].set_xlabel("worst state/country coverage | selected (quota rule)")
    axs[1].set_xlabel("share of +∞ intervals")
    axs[0].set_yticks(y); axs[0].set_yticklabels([DS.LABEL[n] for n in names])
    for a in axs:
        a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
    axs[0].axvline(0.9, color=INK2, lw=1, ls="--", zorder=0)
    fig.legend(ncol=3, frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    fig.savefig(os.path.join(FIG, "fig6_scgeoccp_quota.png")); plt.close(fig)


if __name__ == "__main__":
    main()
