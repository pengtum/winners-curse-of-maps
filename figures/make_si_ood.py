"""Supplementary Fig. 22 | Outside the random-split design (paper/OOD_DESIGN.md; paper/FIGURE_PLAN.md, S22).

a-d  held-out groups (each DHS country, or each FIA state with >= 300 plots, in turn; maps from other groups): coverage
     among the selected units in each group for JOMI calibrated on a random half of the 70% of the group not used by
     the R0 map, i.e. 35% of the group (R1, filled), JOMI calibrated
     on as many units from other groups that did not train the map (R2, hollow) and normalized CP calibrated on the
     group's own units (R1, grey); right: share of infinite intervals for JOMI. Groups sorted by R2. FIA uses the
     site + height map. The 5% panels keep the groups with >= 571 units (K >= 10), fixed before the run.
e    DHS, poorest-predicted 25.2%: how often a 90% upper bound on the selected clusters' mean wealth held, per
     country (small) and pooled (large), for the survey-mean correction and the reference-set central-limit bound,
     with the map trained with (R0) or without (R1) part of the country, and calibration from other countries (R2).
f    local survey size: share of infinite JOMI intervals with n of the local units (35% of the group) as calibration
     or all of them (R1 map, first 20 re-splits, groups that have every n); circles 25.2% / 20%, squares 5%.
g    block-randomized splits (whole countries or k-means blocks of FIA plots allotted to training, calibration or
     test; 100 splits): coverage at the 5% rule (markers) against the same pipeline with single units as blocks
     (dashes; one FIA comparison run, drawn under both block sizes).
Data: results/ood_*.csv through experiments/make_ood.py (tables in results/RESULTS_OOD.md).
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_si_ood.py
writes figs/si_fig_ood.{pdf,svg,png}, figs/source_data/si_fig_ood_{a_d,e,f,g}.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402
OFIA = "fia_post2000" if D.DS.FIA == "fia" else D.DS.FIA         # old table: held-out states on the 2000+ plots
import make_ood as O                                                  # noqa: E402

W, H = 183, 141
PITCH = 2.45                                    # mm per group row in a-d
TOP = 128.0                                     # top edge of a-d
COLX = (2.0, 63.5, 125.0)                       # panel letters
CX, CW, IW, GAP = (22.0, 83.0, 144.0), 27.0, 9.0, 2.0
ROW2 = (11.0, 34.0)                             # e-g: bottom, height
EX, EW = (24.0, 83.0, 142.0), 38.0
# group code -> name: pd.factorize order of the FIA table (experiments/data.py); checked against sizes
FIA = ({0: ("MO", 4367), 1: ("ME", 3247), 2: ("CA", 3292), 3: ("AZ", 490), 4: ("FL", 3650), 5: ("WA", 5129),     # fia_post2000
        6: ("MT", 1490), 7: ("VT", 1000), 8: ("NM", 388), 9: ("CO", 15)} if D.DS.FIA == "fia" else
       {0: ("AZ", 2843), 1: ("CA", 3861), 2: ("CO", 3267), 3: ("FL", 2379), 4: ("ME", 2595), 5: ("MO", 1832),     # fia_db
        6: ("MT", 3696), 7: ("NM", 3351), 8: ("VT", 618), 9: ("WA", 2617)})
NHELD = sum(n >= 300 for _, n in FIA.values())  # experiments/run_ood.py MIN_GROUP
PANELS = [("a", "loco_poverty", "bottom25.2", 0, "DHS, poorest 25.2%: each country held out"),
          ("b", "loco_poverty", "bottom5", 1, "DHS, poorest 5% (countries ≥ 571 clusters)"),
          ("c", f"loso_{OFIA}_siteheight", "top20", 2, f"FIA, highest 20%: {['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'][NHELD]} states held out"),
          ("d", f"loso_{OFIA}_siteheight", "top5", 2, "FIA, highest 5% (states ≥ 571 plots)")]
GREY_SQ = "SCP stud"


def names(tag):
    if tag.startswith("loco"):
        c = D.dhs_countries()
        return {i: D.DHS[k][1] for i, k in enumerate(c)}
    return {i: s for i, (s, _) in FIA.items()}


def data_ad(dfs):
    rows = []
    for l, tag, rule, _, _ in PANELS:
        df = dfs[tag]; nm = names(tag)
        J, N = O.per_group(df, rule, "JOMI stud"), O.per_group(df, rule, GREY_SQ)
        if tag.startswith("loso"):
            for gi, r in J.iterrows():
                assert FIA[gi][1] == int(r[("n_group", "")] if ("n_group", "") in J.columns else r["n_group"]), gi
        for gi in J.index:
            for arm in ("R0", "R1", "R2"):
                rows.append(dict(panel=l, design=tag, rule=rule, group=nm[gi], arm=arm, method="JOMI",
                                 coverage=J.loc[gi, (arm, "cov")], se=J.loc[gi, (arm, "se")],
                                 inf_share=J.loc[gi, (arm, "inf")], order=J.loc[gi, ("R2", "cov")]))
            rows.append(dict(panel=l, design=tag, rule=rule, group=nm[gi], arm="R1", method="Normalized CP",
                             coverage=N.loc[gi, ("R1", "cov")], se=N.loc[gi, ("R1", "se")],
                             inf_share=N.loc[gi, ("R1", "inf")], order=J.loc[gi, ("R2", "cov")]))
    return pd.DataFrame(rows)


def panel_ad(fig, T):
    A = []
    for l, tag, rule, col, _ in PANELS:
        x = T[T.panel == l]; grp = x.drop_duplicates("group").sort_values("order", ascending=False).group.tolist()
        n = len(grp); h = n * PITCH
        top = TOP if l != "d" else TOP - 9 * PITCH - 11.5
        a1 = S.axes_mm(fig, CX[col], top - h, CW, h); a2 = S.axes_mm(fig, CX[col] + CW + GAP, top - h, IW, h)
        y = {g: i for i, g in enumerate(grp)}
        for ax, xl in ((a1, (0.55, 1.02)), (a2, (-0.06, 1.06))):
            ax.set_ylim(n - 0.5, -0.5); ax.set_xlim(*xl); S.hgrid(ax, range(n))
            S.clean(ax, left=False)
        S.target(a1)
        for g in grp:
            v = x[x.group == g].set_index(["method", "arm"])
            a1.plot([v.loc[("JOMI", "R1"), "coverage"], v.loc[("JOMI", "R2"), "coverage"]], [y[g]] * 2,
                    color=S.GRID, lw=0.6, zorder=1)
        j1, j2 = x[(x.method == "JOMI") & (x.arm == "R1")], x[(x.method == "JOMI") & (x.arm == "R2")]
        nc = x[x.method == "Normalized CP"]
        S.dot(a1, nc.coverage, nc.group.map(y), GREY_SQ, s=7, z=2)
        S.dot(a1, j1.coverage, j1.group.map(y), "JOMI stud", s=9, z=4)
        S.dot(a1, j2.coverage, j2.group.map(y), "JOMI stud", s=9, z=3, hollow=True)
        S.dot(a2, j1.inf_share, j1.group.map(y) - 0.18, "JOMI stud", s=7, z=4)
        S.dot(a2, j2.inf_share, j2.group.map(y) + 0.18, "JOMI stud", s=7, z=3, hollow=True)
        a1.set_yticks(range(n)); a1.set_yticklabels(grp, fontsize=S.FS["tick"]); a2.set_yticks([])
        a1.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0]); a1.set_xticklabels(["0.6", "0.7", "0.8", "0.9", "1"])
        a2.set_xticks([0, 1]); a2.set_xticklabels(["0", "1"])
        if l in ("a", "b", "d"):
            a1.set_xlabel("Selected-unit coverage")
        a2.set_title("+∞", fontsize=S.FS["tick"], fontweight="normal", loc="center", pad=1.5, color=S.INK2)
        A += [a1, a2]
    return A


def data_e():
    B = pd.read_csv(os.path.join(O.RES, "ood_loco_poverty_bounds.csv"))
    c = B.groupby(["arm", "bound", "group"]).holds.mean().rename("holds").reset_index()
    p = B.groupby(["arm", "bound"]).holds.mean().rename("pooled").reset_index()
    return c.merge(p, on=["arm", "bound"])


BOUND = {"calibration-mean correction": ("Survey-mean correction", S.GREY_M, "s", -0.17),
         "reference-set CLT": ("Reference-set bound (ours)", S.VIOLET, "o", 0.17)}      # this paper: violet (2026-11-08)


def panel_e(fig, T):
    ax = S.axes_mm(fig, EX[0], *ROW2[:1], EW, ROW2[1])
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.03, 1.03); S.hgrid(ax, [0.25, 0.5, 0.75]); S.target(ax, orient="h")
    rng = np.random.default_rng(0)
    for bn, (lab, c, m, off) in BOUND.items():
        for k, arm in enumerate(("R0", "R1", "R2")):
            x = T[(T.arm == arm) & (T.bound == bn)]
            ax.scatter(k + off + rng.uniform(-0.06, 0.06, len(x)), x.holds, s=4, marker=m, facecolor=c,
                       edgecolor="none", alpha=0.55, zorder=2)
            ax.scatter([k + off], [x.pooled.iloc[0]], s=22, marker=m, facecolor=c, edgecolor=S.INK, linewidths=0.6,
                       zorder=4)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["R0", "R1", "R2"])
    ax.set_yticks([0, 0.5, 0.9]); ax.set_yticklabels(["0", "0.5", "0.9"])
    ax.set_ylabel("Share of surveys in which the bound held"); ax.set_xlabel("Map and calibration")
    hs = [Line2D([], [], ls="none", marker=m, ms=3.4, mfc=c, mec=S.INK, mew=0.5, label=lab)
          for lab, c, m, _ in BOUND.values()]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3,
              handletextpad=0.2, columnspacing=0.8)
    return ax


SURVEY_LINES = [("loco_poverty", "bottom25.2", "DHS 25.2%", "-", "o", -0.06), ("loco_poverty", "bottom5", "DHS 5%", "-", "s", -0.06),
                (f"loso_{OFIA}_siteheight", "top20", "FIA 20%", (0, (3, 1.5)), "o", 0.06),
                (f"loso_{OFIA}_siteheight", "top5", "FIA 5%", (0, (3, 1.5)), "s", 0.06)]
NX = {"50": 0, "100": 1, "200": 2, "all (half of the group)": 3}      # 'all': every local unit (35% of the group)


def data_f(dfs):
    out = []
    for tag, rule, lab, *_ in SURVEY_LINES:
        s = O.survey(dfs[tag]); s = s[s.rule == rule].assign(design=tag, label=lab)
        out.append(s)
    return pd.concat(out, ignore_index=True)


def panel_f(fig, T):
    ax = S.axes_mm(fig, EX[1], *ROW2[:1], EW, ROW2[1])
    ax.set_xlim(-0.3, 3.3); ax.set_ylim(-0.03, 1.03); S.hgrid(ax, [0.25, 0.5, 0.75])
    for tag, rule, lab, ls, mk, dx in SURVEY_LINES:     # dodged so curves that coincide (at 0 or 1) stay visible
        x = T[(T.design == tag) & (T.rule == rule)].assign(xx=lambda d: d.n.map(NX) + dx).sort_values("xx")
        ax.plot(x.xx, x.inf, color=S.BLUE, lw=S.LW["line"], ls=ls, zorder=2)
        ax.scatter(x.xx, x.inf, s=9, marker=mk, facecolor=S.BLUE, edgecolor="white", linewidths=0.4, zorder=3)
    hs = [Line2D([], [], color=S.BLUE, lw=S.LW["line"], ls=ls, marker=mk, ms=3.0, mfc=S.BLUE, mec="white", mew=0.4,
                 label=lab) for _, _, lab, ls, mk, _ in SURVEY_LINES]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3, handlelength=2.2,
              labelspacing=0.25, columnspacing=1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["50", "100", "200", "All"])
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("Local calibration units"); ax.set_ylabel("Share of +∞ JOMI intervals")
    return ax


BLOCKS = [("poverty", "country", "bottom5", "DHS\ncountries"), (OFIA, "k100", "top5", "FIA\n100 blocks"),
          (OFIA, "k25", "top5", "FIA\n25 blocks")]
G_METH = ["SCP |res|", "SCP stud", "GeoCP", "JOMI stud", "SC-GeoCCP"]


def data_g():
    rows = []
    for name, blk, rule, lab in BLOCKS:
        for arm, tag in (("blocks", f"block_{name}_{blk}"), ("random", f"block_{name}_random")):
            df = O.load(tag)
            if df is None:
                continue
            for m in G_METH:
                x = df[(df.rule == rule) & (df.method == m)]
                if len(x):
                    p = O.pool(x)
                    rows.append(dict(design=f"{name}_{blk}", label=lab.replace("\n", " "), rule=rule, split=arm,
                                     method=S.SHORT[m], key=m, coverage=p["cov"], inf_share=p.inf,
                                     p10=(x.n_cov / x.n_sel).quantile(0.1)))
    return pd.DataFrame(rows)


def panel_g(fig, T):
    ax = S.axes_mm(fig, EX[2], *ROW2[:1], EW, ROW2[1])
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.15, 1.03); S.hgrid(ax, [0.4, 0.6, 0.8]); S.target(ax, orient="h")
    off = dict(zip(G_METH, np.linspace(-0.32, 0.32, len(G_METH))))
    for k, (name, blk, *_ ) in enumerate(BLOCKS):
        for m in G_METH:
            x = T[(T.design == f"{name}_{blk}") & (T.key == m)].set_index("split").coverage
            if "random" in x:
                ax.plot([k + off[m]] * 2, [x["blocks"], x["random"]], color=S.GRID, lw=0.6, zorder=1)
                ax.scatter([k + off[m]], [x["random"]], s=22, marker="_", color=S.METHODS[m][1], linewidths=1.1, zorder=2)
            S.dot(ax, [k + off[m]], [x["blocks"]], m, s=9, z=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([b[3] for b in BLOCKS])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 0.9]); ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "0.9"])
    ax.set_ylabel("Coverage among the selected units")
    return ax


def main():
    S.apply()
    dfs = {t: O.load(t) for t in ("loco_poverty", f"loso_{OFIA}_siteheight")}
    Tad, Te, Tf, Tg = data_ad(dfs), data_e(), data_f(dfs), data_g()
    fig = S.figure(W, H)
    A = panel_ad(fig, Tad); E, F, G = panel_e(fig, Te), panel_f(fig, Tf), panel_g(fig, Tg)
    hs = [S.handle("JOMI stud", label="JOMI, local calibration (R1)"),
          S.handle("JOMI stud", label="JOMI, calibration from other groups (R2)", hollow=True),
          S.handle(GREY_SQ, label="Normalized CP, local calibration (R1)"),
          Line2D([], [], color=S.INK2, lw=S.LW["ref"], ls=(0, (3, 2)), label="0.90")]
    fig.legend(handles=hs, ncol=4, loc="upper left", bbox_to_anchor=(CX[0] / W, (H - 1.2) / H), borderaxespad=0,
               handletextpad=0.3, columnspacing=1.4)
    hs_g = [S.handle(m, label=S.SHORT[m].replace(" (ours)", "")) for m in G_METH]
    fig.legend(handles=hs_g, ncol=3, loc="lower left", bbox_to_anchor=(135.0 / W, (ROW2[0] + ROW2[1] + 1.4) / H),
               borderaxespad=0, handletextpad=0.2, columnspacing=0.7, labelspacing=0.25)
    for (l, _, _, col, t) in PANELS:
        y = TOP + 4.5 if l != "d" else TOP - 9 * PITCH - 11.5 + 4.5
        S.letter_mm(fig, COLX[col], y, l)
        S.text_mm(fig, COLX[col] + 4.5, y - 0.35, t, fontsize=S.FS["title"], fontweight="bold", ha="left", va="top")
    ye = ROW2[0] + ROW2[1] + 10.5
    for x, l, t in ((COLX[0], "e", "Bound on mean wealth, DHS 25.2%"), (COLX[1], "f", "Size of the local survey"),
                    (COLX[2], "g", "Block-randomized splits, 5% rule")):
        S.letter_mm(fig, x, ye, l)
        S.text_mm(fig, x + 4.5, ye - 0.35, t, fontsize=S.FS["title"], fontweight="bold", ha="left", va="top")
    S.text_mm(fig, EX[2] + EW, ROW2[0] - 8.6, "markers: blocks; dashes: single units", fontsize=S.FS["note"],
              color=S.INK2, ha="right", va="top")
    S.source_data("si_fig_ood_a_d", Tad); S.source_data("si_fig_ood_e", Te)
    S.source_data("si_fig_ood_f", Tf); S.source_data("si_fig_ood_g", Tg)
    ids = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e", "f", "g"]
    S.save(fig, "si_fig_ood", axes=A + [E, F, G], panel_ids=ids,
           row_groups=[["a1", "a2"], ["b1", "b2"], ["c1", "c2"], ["d1", "d2"], ["e", "f", "g"]],
           column_groups=[["c1", "d1"], ["c2", "d2"]])
    print(Tg.pivot_table(index=["label", "method"], columns="split", values="coverage").round(3).to_string())


if __name__ == "__main__":
    main()
