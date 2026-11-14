"""Tables and figures for QW2 from results/main_*.csv, state_*.csv, units_*_rep0.npz.

    ~/miniforge3/envs/favela/bin/python experiments/make_figs.py
writes results/RESULTS.md and figs/fig{1..4}_*.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "figures")); sys.dont_write_bytecode = True
import data as DS  # noqa: E402
from style import INK, INK2, GRID  # noqa: E402  (chrome colours: one source, figures/style.py)

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
ALPHA = 0.1
STYLE = {  # method -> (color, marker, filled); pre-NC method set, NC figures use figures/style.py
    "SCP |res|": ("#b5b3ac", "x", True), "SCP stud": ("#7d7b74", "s", True),
    "GeoCP v2": ("#1baf7a", "^", True), "ExCP-default": ("#eb6834", "D", True),
    "BY stud": ("#b5b3ac", "v", False), "JOMI |res|": ("#2a78d6", "o", False),
    "JOMI stud": ("#2a78d6", "o", True), "JOMI-ExCP": ("#6250d6", "P", True),
}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load():
    out = []
    for nm in DS.DISPLAY:
        f = os.path.join(RES, f"main_{nm}.csv")
        if os.path.exists(f):
            out.append(pd.read_csv(f))
    return pd.concat(out, ignore_index=True)


def pooled(df, keys):
    g = df.groupby(keys)
    t = pd.DataFrame({"cov": g.n_cov.sum() / g.n_sel.sum(), "n_sel": g.n_sel.mean(),
                      "inf": g.n_inf.sum() / g.n_sel.sum(), "hw_med": g.hw_med.median(),
                      "reps": g.rep.nunique()})
    if "R_mean" in df:
        t["R"] = g.R_mean.mean(); t["bound"] = g.ub_sum.sum() / g.n_sel.sum()
    # split-level spread of the selected-unit coverage
    t["cov_sd"] = g.apply(lambda x: (x.n_cov / x.n_sel).std())
    return t


def md(df, floatfmt=3):
    esc = lambda v: str(v).replace("|", "∣")
    cols = list(df.columns)
    s = "| " + " | ".join(map(esc, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else esc(v) for v in r) + " |\n"
    return s


QUOTA_RULE = {"poverty": "quota_bottom", "poverty_lab": "quota_bottom"}      # aid quotas fund the predicted-poorest per country
KNAP_RULE = {"poverty": "knapsack_need", "poverty_lab": "knapsack_need"}


def policy_rule(nm, frac=5):
    return f"{DS.POLICY[nm]}{frac}"


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load(); df["cov"] = df.n_cov / df.n_sel
    names = [n for n in DS.DISPLAY if n in set(df.dataset)]
    P = pooled(df, ["dataset", "rule", "method"])
    lines = ["# QW2 结果（自动生成：experiments/make_figs.py）\n",
             f"α = {ALPHA}；随机划分 40/30/30；每个数据集 {int(P.reps.max())} 次划分。覆盖 = 被选单元上的合并覆盖率 Σ覆盖/Σ被选"
             "（即 P(Y∈C | 被选) 的估计）；`sd` 为逐次划分被选覆盖率的标准差；`inf` 为 +∞ 区间份额；"
             "`bound` = 1−α+E[1/(1+|R|)]（仅不加权 JOMI）。* 为 PLACES 模型输出，不是实测真值。\n"]

    # ---------- Table 1: headline (policy direction, 5%)
    meths = ["SCP |res|", "SCP stud", "GeoCP v2", "ExCP-default", "BY stud", "JOMI |res|", "JOMI stud", "JOMI-ExCP"]
    rows = []
    for nm in names:
        r = policy_rule(nm); x = P.loc[(nm, r)]
        row = {"data": DS.LABEL[nm], "rule": r}
        for mth in meths:
            row[mth] = x.loc[mth, "cov"]
        j = x.loc["JOMI stud"]
        row["sd(ExCP)"] = x.loc["ExCP-default", "cov_sd"]; row["sd(JOMI)"] = j.cov_sd
        row["|R|"] = j.R; row["bound"] = j.bound
        row["width JOMI/ExCP"] = j.hw_med / x.loc["ExCP-default", "hw_med"]
        row["inf JOMI"] = j.inf; row["inf ExCP"] = x.loc["ExCP-default", "inf"]
        row["inf GeoCPv2"] = x.loc["GeoCP v2", "inf"]
        rows.append(row)
    T1 = pd.DataFrame(rows)
    lines += ["## 表 1　政策方向 5% 选择：被选单元上的覆盖\n", md(T1), "\n"]

    # ---------- Table 2: all rules, ExCP-default vs JOMI stud
    rules = ["top1", "top5", "top20", "bottom1", "bottom5", "bottom20", "quota", "knapsack",
             "quota_bottom", "knapsack_need", "mask_base", "mask_h2"]
    rows = []
    for nm in names:
        for r in rules:
            if (nm, r) not in P.index.droplevel(2):
                continue
            x = P.loc[(nm, r)]
            rows.append({"data": DS.LABEL[nm], "rule": r, "|S|": x.loc["JOMI stud", "n_sel"],
                         "SCP |res|": x.loc["SCP |res|", "cov"], "SCP stud": x.loc["SCP stud", "cov"],
                         "GeoCP v2": x.loc["GeoCP v2", "cov"], "ExCP": x.loc["ExCP-default", "cov"],
                         "JOMI stud": x.loc["JOMI stud", "cov"], "JOMI-ExCP": x.loc["JOMI-ExCP", "cov"],
                         "|R|": x.loc["JOMI stud", "R"], "bound": x.loc["JOMI stud", "bound"],
                         "inf JOMI": x.loc["JOMI stud", "inf"], "inf ExCP": x.loc["ExCP-default", "inf"],
                         "w ratio": x.loc["JOMI stud", "hw_med"] / x.loc["ExCP-default", "hw_med"]})
    T2 = pd.DataFrame(rows)
    T2["|S|"] = T2["|S|"].round(0).astype(int)
    lines += ["## 表 2　全部规则：默认方法 vs JOMI\n", md(T2), "\n"]
    # validity summary
    j = T2[(T2["inf JOMI"] < 0.05)]
    lines += [f"ExCP-default 在被选单元上的 +∞ 份额：中位 {T2['inf ExCP'].median():.3f}，最大 {T2['inf ExCP'].max():.3f}"
              f"（{T2.loc[T2['inf ExCP'].idxmax(), 'data']} {T2.loc[T2['inf ExCP'].idxmax(), 'rule']}）；"
              f"> 5% 的格 {(T2['inf ExCP'] > 0.05).sum()} / {len(T2)}。全体测试单元上它被调到 ≤ 0.5%。\n\n"]
    lines += [f"JOMI stud 覆盖（+∞ < 5% 的格）：最小 {j['JOMI stud'].min():.3f}，最大 {j['JOMI stud'].max():.3f}，"
              f"超出上界的格数 {(j['JOMI stud'] > j['bound'] + 0.01).sum()} / {len(j)}；"
              f"ExCP-default < 0.85 的格数 {(T2['ExCP'] < 0.85).sum()} / {len(T2)}，"
              f"SCP stud < 0.85：{(T2['SCP stud'] < 0.85).sum()}，SCP |res| < 0.85：{(T2['SCP |res|'] < 0.85).sum()}，"
              f"GeoCP v2 < 0.85：{(T2['GeoCP v2'] < 0.85).sum()}。\n\n"]

    # ---------- Table 3: state quota, per-state conditional coverage
    rows = []
    for nm in names:
        f = os.path.join(RES, f"state_{nm}.csv")
        if not os.path.exists(f):
            continue
        s = pd.read_csv(f)
        qrule = QUOTA_RULE.get(nm, "quota")
        s = s[s.rule == qrule] if "rule" in s else s
        g = s.groupby(["method", "state"]).agg(n=("n_sel", "sum"), c=("n_cov", "sum"), i=("n_inf", "sum"))
        g = g[g.n >= 50]
        g["cov"] = g.c / g.n
        row = {"data": DS.LABEL[nm], "rule": qrule, "states(n≥50)": g.loc["JOMI stud"].shape[0]}
        for mth in ["SCP |res|", "ExCP-default", "JOMI stud", "JOMI|state stud"]:
            row[f"min {mth}"] = g.loc[mth, "cov"].min()
        row["share states <0.85 (ExCP)"] = (g.loc["ExCP-default", "cov"] < 0.85).mean()
        row["share states <0.85 (JOMI)"] = (g.loc["JOMI stud", "cov"] < 0.85).mean()
        gs = g.loc["JOMI|state stud"]; row["inf JOMI|state"] = gs.i.sum() / gs.n.sum()
        # binomial null: expected minimum over these states if every state had exactly 0.90
        nn = g.loc["JOMI stud", "n"].values
        sims = np.random.default_rng(0).binomial(nn[None, :], 0.9, (4000, len(nn))) / nn[None, :]
        row["null min (exact 0.9)"] = sims.min(1).mean(); row["null 5% min"] = np.quantile(sims.min(1), 0.05)
        rows.append(row)
    T3 = pd.DataFrame(rows)
    lines += ["## 表 3　州配额：逐州的选择条件覆盖（合并 100 次划分，每州被选 ≥ 50 次）\n",
              "弱版 JOMI 只保证\"被选\"条件下的覆盖，不保证逐州；JOMI|state 保证\"被选且在州 s\"，代价是州内参考集很小（+∞）。\n\n",
              md(T3), "\n"]

    # ---------- Table 4: policy quantities
    rows = []
    positive = {nm: DS.load(nm)["y"].min() > 0 for nm in names}     # ratios need a positive outcome
    for nm in names:
        for r in [policy_rule(nm), KNAP_RULE.get(nm, "knapsack"), QUOTA_RULE.get(nm, "quota")]:
            d = df[(df.dataset == nm) & (df.rule == r)]
            for mth in ["SCP |res|", "SCP stud", "GeoCP v2", "ExCP-default", "JOMI stud"]:
                x = d[d.method == mth]
                fin = np.isfinite(x.sumU)
                rows.append({"data": DS.LABEL[nm], "rule": r, "method": mth,
                             "upper breach": x.up_breach.sum() / x.n_sel.sum(),
                             "lower breach": x.lo_breach.sum() / x.n_sel.sum(),
                             "ΣU/ΣY": float(np.median(x.sumU[fin] / x.sumY[fin])) if fin.any() else np.inf,
                             "ΣL/ΣY": float(np.median((x.sumL / x.sumY)[np.isfinite(x.sumL)])) if np.isfinite(x.sumL).any() else -np.inf,
                             "Σμ/ΣY": float(np.median(x.sumMu / x.sumY))})
                if not positive[nm]:
                    rows[-1].update({"ΣU/ΣY": np.nan, "ΣL/ΣY": np.nan, "Σμ/ΣY": np.nan})
    T4 = pd.DataFrame(rows)
    lines += ["## 表 4　政策量：被选集合上的上界/下界之和与真实总量\n",
              "`upper breach` = P(Y > U | 被选)（预留不足的单元比例）；`lower breach` = P(Y < L | 被选)（认证过量的单元比例）；"
              "ΣU/ΣY、ΣL/ΣY、Σμ/ΣY 取 100 次划分的中位数（结果可取负值的数据集，如 DHS 财富指数，不报比值）。\n\n", md(T4), "\n"]

    # ---------- Table 5: JOMI-ExCP kappa
    e = df[df.method == "JOMI-ExCP"].groupby(["dataset", "rule"]).kappa_sel.apply(lambda v: (v > 0).mean()).unstack()
    lines += ["## 表 5　JOMI-ExCP 实际保留局部化（κ_sel 非空）的划分比例\n", md(e.reset_index(), 2), "\n"]
    open(os.path.join(RES, "RESULTS.md"), "w").write("".join(lines))

    fig1(P, names); fig2(P, names); fig4(T4, names)
    fig3(DS.FIA, "quota", "state", "fig3_fia_quota_mask_maps.png")
    fig3("poverty", "quota_bottom", "country", "fig5_povertymap_country_quota_maps.png")
    print("".join(lines[:4]))


def fig1(P, names):
    meths = ["SCP |res|", "SCP stud", "GeoCP v2", "ExCP-default", "JOMI stud", "JOMI-ExCP"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(names))[::-1]
    off = np.linspace(-0.3, 0.3, len(meths))
    ax.axvline(1 - ALPHA, color=INK2, lw=1, ls="--", zorder=0)
    ax.text(1 - ALPHA, len(names) - 0.35, "target 0.90", ha="center", va="bottom", color=INK2, fontsize=8)
    for k, mth in enumerate(meths):
        c, mk, fill = STYLE[mth]
        v = [P.loc[(nm, policy_rule(nm), mth), "cov"] for nm in names]
        ax.scatter(v, y + off[k], s=28, marker=mk, color=c if fill else "none", edgecolor=c,
                   lw=1.2, label=mth, zorder=3)
        for nm, vv, yy in zip(names, v, y + off[k]):     # coverage partly bought by abstention
            fi = P.loc[(nm, policy_rule(nm), mth), "inf"]
            if fi > 0.05 and mth in ("ExCP-default", "JOMI stud"):
                lab = f"{fi:.0%} +∞" + (" (JOMI)" if mth == "JOMI stud" else " (ExCP)")
                ax.annotate(lab, (vv, yy), xytext=(6, -4 if mth == "JOMI stud" else 4), textcoords="offset points",
                            va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([f"{DS.LABEL[n]} ({DS.POLICY[n]} 5%)" for n in names])
    ax.set_xlabel("coverage among selected units  P(Y ∈ C | selected)")
    ax.set_xlim(0.15, 1.02); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.savefig(os.path.join(FIG, "fig1_selected_coverage.png")); plt.close(fig)


def fig2(P, names):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fr = [1, 5, 20]
    for ax, d in zip(axs, ["top", "bottom"]):
        ax.axhline(1 - ALPHA, color=INK2, lw=1, ls="--", zorder=0)
        for nm in names:
            v = [P.loc[(nm, f"{d}{f}", "ExCP-default"), "cov"] for f in fr]
            jv = [P.loc[(nm, f"{d}{f}", "JOMI stud"), "cov"] for f in fr]
            ax.plot(fr, v, color=STYLE["ExCP-default"][0], lw=1.2, alpha=0.75, marker="D", ms=3)
            ax.plot(fr, jv, color=STYLE["JOMI stud"][0], lw=1.2, alpha=0.75, marker="o", ms=3)
            if min(v) < 0.89:
                ax.annotate(DS.LABEL[nm], (5, v[1]), xytext=(0, -7), textcoords="offset points",
                            ha="center", va="center", fontsize=6.5, color=INK2)
        ax.set_xscale("log"); ax.set_xticks(fr); ax.set_xticklabels([f"{f}%" for f in fr])
        ax.set_xlim(0.6, 25); ax.set_ylim(0.82, 1.01)
        ax.set_title(f"{d}-K selection", fontsize=9, color=INK); ax.set_xlabel("K / m")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_ylabel("coverage | selected")
    axs[0].plot([], [], color=STYLE["ExCP-default"][0], marker="D", ms=3, label="ExCP-default (stud × RLCP+)")
    axs[0].plot([], [], color=STYLE["JOMI stud"][0], marker="o", ms=3, label="JOMI stud (excp_select)")
    fig.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.savefig(os.path.join(FIG, "fig2_selectivity.png")); plt.close(fig)


def fig3(nm, qrule, unit, fname):
    f = os.path.join(RES, f"units_{nm}_rep0.npz")
    if not os.path.exists(f) or f"{qrule}|idx" not in np.load(f).files:
        return
    z = np.load(f); s = z["coords"]; T = z["T"]
    sp = pd.read_csv(os.path.join(RES, f"state_{nm}.csv"))
    if "rule" in sp:
        sp = sp[sp.rule == qrule]
    gs = sp.groupby(["method", "state"]).agg(n=("n_sel", "sum"), c=("n_cov", "sum")); gs["cov"] = gs.c / gs.n
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#d95926", "#ece9e2", "#2a78d6"])
    norm = TwoSlopeNorm(vcenter=0.9, vmin=0.4, vmax=1.0)
    fig = plt.figure(figsize=(7.4, 2.9))
    gsp = fig.add_gridspec(2, 3, height_ratios=[1, 0.05], hspace=0.08, wspace=0.05)
    axs = [fig.add_subplot(gsp[0, k]) for k in range(3)]
    for ax, mth, ttl in [(axs[0], "ExCP-default", f"{unit} quota: ExCP-default"), (axs[1], "JOMI stud", f"{unit} quota: JOMI")]:
        ax.scatter(s[T, 0], s[T, 1], s=0.4, color=GRID, rasterized=True)
        J = z[f"{qrule}|idx"]; st = z["state"][J]
        col = [gs.loc[(mth, u), "cov"] if (mth, u) in gs.index else np.nan for u in st]
        sc = ax.scatter(s[J, 0], s[J, 1], s=9, c=col, cmap=cmap, norm=norm, edgecolor=INK2, lw=0.2)
        ax.set_title(ttl, fontsize=8.5)
    cax = fig.add_subplot(gsp[1, :2])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label(f"coverage of selected units in the {unit} (100 splits pooled)", fontsize=7.5)
    cb.ax.axvline(0.9, color=INK, lw=1)
    ax = axs[2]
    Jm = z["mask_base|idx"]; ab = np.setdiff1d(T, Jm)
    ax.scatter(s[Jm, 0], s[Jm, 1], s=0.5, color="#2a78d6", rasterized=True, label=f"published ({len(Jm)/len(T):.0%})")
    ax.scatter(s[ab, 0], s[ab, 1], s=0.5, color="#eb6834", rasterized=True, label="abstain (GeoCP v2 = +∞)")
    ax.set_title("GeoCP-v2 finite mask (h = 10-NN scale)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", markerscale=6, bbox_to_anchor=(0, -0.22))
    for a_ in axs:
        a_.set_xticks([]); a_.set_yticks([]); a_.set_aspect("equal")
        for sp_ in a_.spines.values():
            sp_.set_visible(False)
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def fig4(T4, names):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    meths = ["SCP |res|", "SCP stud", "GeoCP v2", "ExCP-default", "JOMI stud"]
    y = np.arange(len(names))[::-1]; off = np.linspace(-0.28, 0.28, len(meths))
    for ax, col, ttl, ref in [(axs[0], "upper breach", "P(Y > upper | selected)", ALPHA / 2),
                              (axs[1], "ΣU/ΣY", "reserved upper bounds / true total", 1.0)]:
        ax.axvline(ref, color=INK2, lw=1, ls="--", zorder=0)
        for k, mth in enumerate(meths):
            c, mk, fill = STYLE[mth]
            v = [T4[(T4.data == DS.LABEL[nm]) & (T4.rule == policy_rule(nm)) & (T4.method == mth)][col].values[0] for nm in names]
            ax.scatter(v, y + off[k], s=24, marker=mk, color=c if fill else "none", edgecolor=c, lw=1.1, label=mth)
        ax.set_title(ttl, fontsize=8.5); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_yticks(y); axs[0].set_yticklabels([f"{DS.LABEL[n]} ({DS.POLICY[n]} 5%)" for n in names])
    axs[0].legend(ncol=5, frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.savefig(os.path.join(FIG, "fig4_policy.png")); plt.close(fig)


if __name__ == "__main__":
    main()
