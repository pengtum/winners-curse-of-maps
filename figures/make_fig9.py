"""Fig. 9 | The audit a decision needs can be computed in advance (paper/AUDIT_PLANNER_DESIGN.md).

a  audited units for a finite 90% interval and for a regional floor, against the selected share (excp_select/planner.py).
b  a 30-plot project among 6,000 FIA plots: planned chance of a finite interval by audit size and where the audit is drawn;
   markers are realized shares (site maps, ESA CCI; experiments/run_audit_planner.py carbon).
c  the same project: VCS over-crediting risk of the survey-calibrated (rule) and selection-aware (always deducted) estimates,
   with map-clustered 95% intervals, and the selection-aware credit as a share of the true total.
d  regional floors against audit size (planned curves) and the lowest regional coverage realized (DHS countries, FIA states).
Data: results/fia_db/planner/summary_carbon.csv, summary_floors.csv (experiments/make_audit_planner.py).
    ~/miniforge3/envs/favela/bin/python figures/make_fig9.py
writes figs/nc_fig9_planner.{pdf,svg,png}, figs/source_data/nc_fig9{a,b,c,d}.csv
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
from excp_select import planner as P                                  # noqa: E402

D = os.path.join(ROOT, "results", "fia_db", "planner")
W, H = 183, 118
MK = {"site": "o", "cci": "s"}
STRATA = [(1.0, "Whole map", S.GREY_M), (0.10, "Highest 10%", S.AQUA), (0.05, "Highest 5%", S.BLUE)]
ARMS = [(1.0, 3000, "Whole map, 3,000"), (1.0, 1869, "Whole map, 1,869"), (0.10, 186, "Highest 10%, 186"), (0.05, 92, "Highest 5%, 92")]


def panel_a(fig):
    ax = S.axes_mm(fig, 15, 70, 66, 42)
    sh = np.geomspace(0.005, 0.30, 40); rows = []
    for s in sh:
        K = int(round(s * 100000))
        rows.append(dict(share=s, n90=P.n_for_finite(100000, K, 0.9), n50=P.n_for_finite(100000, K, 0.5),
                         floor_w10=0.95 / (0.10 * s * 0.10), floor_w2=0.95 / (0.10 * s * 0.02)))
    T = pd.DataFrame(rows)
    ax.plot(T.share, T.floor_w2, color=S.VIOLET, lw=S.LW["line"], ls="--")
    ax.plot(T.share, T.floor_w10, color=S.VIOLET, lw=S.LW["line"])
    ax.plot(T.share, T.n90, color=S.BLUE, lw=S.LW["line"])
    ax.plot(T.share, T.n50, color=S.BLUE, lw=S.LW["line"], ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.005, 0.30); ax.set_ylim(20, 400000)   # headroom: legend
    ax.minorticks_off()
    ax.set_xticks([0.005, 0.01, 0.05, 0.2]); ax.set_xticklabels(["0.5%", "1%", "5%", "20%"])
    ax.set_yticks([100, 1000, 10000, 100000]); ax.set_yticklabels(["100", "1,000", "10,000", "100,000"])
    ax.set_xlabel("Share of candidates selected"); ax.set_ylabel("Audited units needed")
    S.clean(ax); ax.spines["left"].set_bounds(20, 200000)
    ax.legend(handles=[Line2D([], [], color=S.VIOLET, ls="--", label="Floor 0.80, region with 2%"),
                       Line2D([], [], color=S.VIOLET, label="Floor 0.80, region with 10%"),
                       Line2D([], [], color=S.BLUE, label="Finite interval, 90% chance"),
                       Line2D([], [], color=S.BLUE, ls="--", label="Finite interval, 50% chance")],
              loc="upper right", frameon=False, fontsize=S.FS["legend"], handlelength=2, borderaxespad=0.1, labelspacing=0.25)
    return ax, T


def panel_b(fig, C):
    ax = S.axes_mm(fig, 101, 70, 78, 42)
    rows = []
    for q, lab, col in STRATA:
        Ms = int(round(q * 6000)); ns = np.unique(np.geomspace(10, min(Ms - 31, 3000), 120).astype(int))
        pf = [P.p_finite(int(n), Ms - int(n), 30) for n in ns]
        ax.plot(ns, pf, color=col, lw=S.LW["line"])
        rows += [dict(stratum=lab, n=int(n), planned=p) for n, p in zip(ns, pf)]
        for kind, s, z in (("cci", 14, 4), ("site", 6, 5)):          # same values: the circle sits inside the square
            g = C[(C["map"] == kind) & (C.q == q)]
            ax.scatter(g.n, g.fin, marker=MK[kind], s=s, facecolor="white" if kind == "cci" else col, edgecolor=col,
                       lw=S.LW["marker"], zorder=z)
    S.target(ax, 0.9, orient="h")
    for n, y, s, col in ((92, 0.40, "92", S.BLUE), (186, 0.25, "186", S.AQUA), (1869, 0.40, "1,869", S.GREY_D)):
        ax.annotate(s, (n, 0.9), (n, y), ha="center", fontsize=S.FS["note"], color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=S.LW["ref"]))
    ax.set_xscale("log"); ax.set_xlim(10, 3300); ax.set_ylim(-0.03, 1.03); ax.minorticks_off()
    ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10", "100", "1,000"])
    ax.set_xlabel("Audited plots"); ax.set_ylabel("Chance of a finite interval")
    S.clean(ax)
    for x, y, s, col, ha in ((54, 0.52, "Highest 5%", S.BLUE, "right"), (215, 0.62, "Highest 10%", S.AQUA, "left"),
                             (880, 0.22, "Whole map", S.GREY_D, "right")):
        ax.text(x, y, s, color=col, fontsize=S.FS["note"], ha=ha, va="center")
    ax.legend(handles=[Line2D([], [], color=S.INK2, label="Planned"),
                       Line2D([], [], marker=MK["site"], ls="", mfc=S.INK2, mec=S.INK2, ms=2.4, label="Realized, site maps"),
                       Line2D([], [], marker=MK["cci"], ls="", mfc="white", mec=S.INK2, ms=3.4, label="Realized, ESA CCI")],
              loc="upper left", bbox_to_anchor=(0.0, 0.87), frameon=False, fontsize=S.FS["legend"], handlelength=1.6,
              borderaxespad=0.2, labelspacing=0.25)
    return ax, pd.DataFrame(rows)


def panel_c(fig, C):
    a1, a2 = S.axes_mm(fig, 30, 11, 46, 42), S.axes_mm(fig, 79, 11, 16, 42)
    ys = {arm[:2]: len(ARMS) - 1 - i for i, arm in enumerate(ARMS)}; rows = []
    for (q, n), y in ys.items():
        for kind, dy in (("site", 0.17), ("cci", -0.17)):
            r = C[(C["map"] == kind) & (C.q == q) & (C.n == n)].iloc[0]
            face = lambda c: "white" if kind == "cci" else c
            a1.errorbar(r.ma_rule, y + dy + 0.06, xerr=r.ma_rule_ci, marker=MK[kind], ms=3.0, color=S.RED, mfc=face(S.RED), lw=S.LW["err"])
            a1.errorbar(r.sa_always, y + dy - 0.06, xerr=r.sa_ci, marker=MK[kind], ms=3.5, color=S.VIOLET, mfc=face(S.VIOLET), lw=S.LW["err"])   # ours
            a2.scatter(r.share_sa, y + dy, marker=MK[kind], s=12, facecolor=face(S.VIOLET), edgecolor=S.VIOLET, lw=S.LW["marker"])
            rows.append(dict(map=kind, q=q, n=n, survey_rule=r.ma_rule, survey_ci=r.ma_rule_ci, aware_always=r.sa_always,
                             aware_ci=r.sa_ci, credit_share=r.share_sa))
    a1.axvline(1 / 3, color=S.INK2, lw=S.LW["ref"], ls=":")
    a1.text(1 / 3 + 0.01, 3.55, "Declared 1/3", fontsize=S.FS["note"], color=S.INK2, va="center")
    for a in (a1, a2):
        a.set_ylim(-1.0, 3.75); S.clean(a)
    a1.set_yticks(list(ys.values())); a1.set_yticklabels([x[2] for x in ARMS])
    a2.set_yticks(list(ys.values())); a2.set_yticklabels([])
    a1.set_xlim(0.15, 1.0); a1.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); a1.set_xlabel("Over-crediting risk")
    a2.set_xlim(0.85, 1.0); a2.set_xticks([0.9, 1.0]); a2.set_xlabel("Credit / true total")
    a1.legend(handles=[Line2D([], [], color=S.RED, marker="o", ms=3, lw=S.LW["err"], label="Survey-calibrated"),
                       Line2D([], [], color=S.VIOLET, marker="o", ms=3.5, lw=S.LW["err"], label="Selection-aware (ours)")],
              loc="lower right", frameon=False, fontsize=S.FS["legend"], handlelength=1.4, borderaxespad=0.1, labelspacing=0.25)
    return [a1, a2], pd.DataFrame(rows)


def panel_d(fig):
    F = pd.read_csv(os.path.join(D, "summary_floors.csv")); F = F[(F.kind == "unit") & (F.sel >= 50)]
    axs, rows = [], []
    for i, (name, lab) in enumerate((("poverty", "DHS, 23 countries"), ("fia_db", "FIA, ten states"))):
        ax = S.axes_mm(fig, 113 + i * 34, 11, 31, 42); axs.append(ax)
        g = F[F.dataset == name]; b = g[g.n == 3000]
        lo, med = b.M_u.min(), b.M_u.median()
        ns = np.geomspace(100, 100000, 100)
        for mu, ls, l in ((med, "-", "median"), (lo, "--", "smallest")):
            fl = 0.9 - 0.95 * 3000 / ((ns + 1) * mu)
            ax.plot(ns, np.where(fl > 0, fl, np.nan), color=S.GREY_D, lw=S.LW["line"], ls=ls)
            rows += [dict(dataset=name, region=l, n=n, value=f) for n, f in zip(ns, fl)]
        for key, col, mk in (("cov_jomi", S.BLUE, "o"), ("cov_scg", S.VIOLET, "D")):
            w = g.groupby("n")[key].min()
            ax.plot(w.index, w.values, color=col, marker=mk, ms=3, lw=S.LW["line"])
            rows += [dict(dataset=name, region=f"lowest {key}", n=int(n), value=v) for n, v in w.items()]
        ax.axhline(0.9, color=S.INK2, lw=S.LW["ref"], ls=":")
        ax.set_xscale("log"); ax.set_xlim(100, 50000); ax.set_ylim(0, 1.3); ax.minorticks_off()   # headroom: legends
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticks([100, 1000, 10000]); ax.set_xticklabels(["100", "1,000", "10,000"])
        ax.set_title(lab, fontsize=S.FS["title"], pad=2); S.clean(ax); ax.spines["left"].set_bounds(0, 1.0)
        ax.set_xlabel("Audited units")
        if i == 0:
            ax.set_ylabel("Floor or coverage")
        else:
            ax.set_yticklabels([])
    kw = dict(loc="upper left", frameon=False, fontsize=S.FS["legend"], handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)
    axs[0].legend(handles=[Line2D([], [], color=S.VIOLET, marker="D", ms=3, label="RegionCP (ours)"),
                           Line2D([], [], color=S.BLUE, marker="o", ms=3, label="JOMI")], **kw)
    axs[1].legend(handles=[Line2D([], [], color=S.GREY_D, label="Median floor"),
                           Line2D([], [], color=S.GREY_D, ls="--", label="Smallest floor")], **kw)
    return axs, pd.DataFrame(rows)


def main():
    S.apply()
    C = pd.read_csv(os.path.join(D, "summary_carbon.csv"))
    fig = S.figure(W, H)
    A, Ta = panel_a(fig); B, Tb = panel_b(fig, C); Cx, Tc = panel_c(fig, C); Dx, Td = panel_d(fig)
    for x, y, l in ((2.0, H - 1.5, "a"), (88.0, H - 1.5, "b"), (2.0, 57.5, "c"), (101.0, 57.5, "d")):
        S.letter_mm(fig, x, y, l)
    for k, T in zip("abcd", (Ta, Tb, Tc, Td)):
        S.source_data(f"nc_fig9{k}", T)
    S.save(fig, "nc_fig9_planner", axes=[A, B] + Cx + Dx, panel_ids=["a", "b", "c1", "c2", "d1", "d2"],
           row_groups=[["a", "b"], ["c1", "c2"], ["d1", "d2"]])


if __name__ == "__main__":
    main()
