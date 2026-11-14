"""Fig. 5 | Other uncertainty methods and score functions (paper/FIGURE_PLAN.md, result 2).

a  coverage among the selected units against interval width relative to JOMI (policy-direction 5%), one small
   point per setting and method, large marker = median over the 11 settings; hollow = reference-set
   calibration on a QRF score (CQR form).
b  the guarantee is an average: FIA state quota, coverage in each state for JOMI with the normalized score
   and with the additive QRF score, and for RegionCP (ours; added 2026-11-08), states ordered by mean biomass.
The deep-backbone panel (ResNet vs frozen DOFA; data_b, panel_b, kept here for import) moved on 2026-10-28 to
Supplementary Fig. 9 (figures/make_si_backbones.py): its heads were chosen after inspecting the evaluation units.
Data: results/main_*, bl_main_*, scg_main_*_l1fix, sf_main_* (a); scg_groups_fia, sf_groups_fia (b).
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_fig5.py
writes figs/nc_fig5_alternatives.{pdf,svg,png}, figs/source_data/nc_fig5{a,b}.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402

W, H = 183, 92
A_METH = ["SCP |res|", "SCP stud", "CQR", "GeoCP", "Regression kriging", "QRF", "Bootstrap ensemble + residual",
          "JOMI stud", "SC-GeoCCP", "JOMI CQR-QRF", "SC-GeoCCP CQR-QRF"]
B_METH = ["SCP stud", "GeoCP", "JOMI stud", "SC-GeoCCP"]
LAB_B = {"poverty": "DHS wealth", "m3leo": "Biomass†", "floods": "Flooded fraction", "usavars": "Tree cover†"}
FIA_STATE = D.fia_states()                                            # group code -> state (fig_data.fia_states)
# layout (mm)
A = (12.0, 11.0, 86.0, 72.0)
B = (122.0, 55.5, 59.0, 27.5)                  # Supplementary Fig. 9 resets this before calling panel_b
C = (122.0, 11.0, 59.0, 72.0)


def data_a():
    rows = []
    for nm, *_ in S.DATASETS:
        df = D.runs(nm); rule = D.policy_rule(nm); d = df[df.rule == rule]
        jw = d[d.method == "JOMI stud"].hw_mean_fin.mean()
        for mth in A_METH:
            x = d[d.method == mth]
            rows.append(dict(dataset=nm, rule=rule, method=mth, label=S.SHORT[mth],
                             width_rel_jomi=x.hw_mean_fin.mean() / jw, coverage=x.n_cov.sum() / x.n_sel.sum(),
                             inf_share=x.n_inf.sum() / x.n_sel.sum()))
    return pd.DataFrame(rows)


def panel_a(fig, T):
    ax = S.axes_mm(fig, *A)
    ax.set_xscale("log"); ax.set_xlim(0.11, 2.75); ax.set_ylim(0.15, 1.035)
    S.target(ax, orient="h"); ax.plot([1, 1], [0.55, 1.035], color=S.GRID, lw=S.LW["ref"], zorder=0)
    S.hgrid(ax, [0.2, 0.4, 0.6, 0.8, 1.0])
    for mth in A_METH:
        x = T[T.method == mth]
        S.dot(ax, x.width_rel_jomi, x.coverage, mth, s=6, z=2, alpha=0.45)
    for mth in A_METH:                  # medians; the selection-conditional ones drawn last (on top)
        x = T[T.method == mth]; top = mth in ("JOMI stud", "SC-GeoCCP")
        z = 6 if top else 5.5 if mth == "SCP |res|" else 5   # split CP's cross sits on GeoCP's diamond: draw it above
        S.dot(ax, [x.width_rel_jomi.median()], [x.coverage.median()], mth, s=34 if top else 30, z=z)
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.125×", "0.25×", "0.5×", "1×", "2×"])
    ax.minorticks_off()
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 0.9, 1.0]); ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "0.9", "1.0"])
    ax.set_xlabel("Mean interval width relative to JOMI (log scale)")
    ax.set_ylabel("Coverage among selected units")
    kw = dict(fontsize=S.FS["note"], color=S.INK2, fontstyle="italic")
    ax.text(0.115, 0.17, "Narrower, misses the target", ha="left", va="bottom", **kw)
    ax.text(2.7, 0.97, "Wider,\nno guarantee", ha="right", va="top", linespacing=1.15, **kw)
    ax.annotate("Selection-conditional (JOMI, RegionCP):\n0.90 at 1–1.2× the width", (0.975, 0.91),
                xytext=(0.115, 0.99), ha="left", va="top", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.MUTED, shrinkA=1, shrinkB=3), **kw)
    hs = [S.handle(m, ms=4.0) for m in A_METH]
    ax.legend(handles=hs, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.02), handletextpad=0.2,
              columnspacing=0.8, labelspacing=0.35, title="Large marker: median of 11 settings",
              title_fontproperties=dict(size=S.FS["note"], style="italic"))._legend_box.align = "left"
    return ax


def data_b():
    rows = []
    for ds, (lab, pol) in D.FM.items():
        for bb in ("resnet", "dofa"):
            d = D.geofm(ds, bb)
            if d is None:
                continue
            P = D.pooled(d.rename(columns={"hw_mean_fin": "hw_mean_fin"})[D.COLS])
            for mth in B_METH:
                r = P.loc[(f"{pol}5", mth)]
                rows.append(dict(dataset=ds, label=lab, backbone=bb, rule=f"{pol}5", method=S.SHORT[mth], key=mth,
                                 coverage=r["cov"], ci_lo=r.lo, ci_hi=r.hi, inf_share=r.inf))
    return pd.DataFrame(rows)


def panel_b(fig, T):
    ax = S.axes_mm(fig, *B)
    ds = list(D.FM); y = {d: i for i, d in enumerate(ds)}
    ax.set_ylim(len(ds) - 0.5, -0.5); ax.set_xlim(0.15, 1.02)
    S.hgrid(ax, range(len(ds))); S.target(ax)
    off = dict(zip(B_METH, np.linspace(-0.27, 0.27, len(B_METH))))
    for mth in B_METH:
        for d in ds:
            x = T[(T.dataset == d) & (T.key == mth)].set_index("backbone")
            yy = y[d] + off[mth]
            if {"resnet", "dofa"} <= set(x.index):
                ax.plot([x.loc["resnet", "coverage"], x.loc["dofa", "coverage"]], [yy, yy], color=S.METHODS[mth][1],
                        lw=0.7, zorder=2, alpha=0.8)
            for bb in x.index:
                S.dot(ax, [x.loc[bb, "coverage"]], [yy], mth, s=9, z=3, hollow=(bb == "dofa"))
    ax.set_yticks(range(len(ds))); ax.set_yticklabels([LAB_B[d] + (" ↓" if D.FM[d][1] == "bottom" else " ↑")
                                                        for d in ds], fontsize=S.FS["tick"])
    ax.tick_params(axis="y", length=0); S.clean(ax, left=False)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 0.9]); ax.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "0.9"])
    ax.set_xlabel("Coverage among selected units", labelpad=1.5)
    hs = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=S.INK2, mec=S.INK2, label="ResNet"),
          Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=S.INK2, mew=0.75, label="Frozen DOFA (GeoFM)"),
          Line2D([], [], ls="none", label="† remote-sensing product label")]
    ax.legend(handles=hs, ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.2,
              columnspacing=1.0, borderaxespad=0.2)
    return ax


def data_c():
    g = D.groups(D.DS.FIA, "state"); g = g[(g.rule == "quota") & (g.n >= D.NMIN)]
    z = D.DS.load(D.DS.FIA); mean_y = pd.Series(z["y"]).groupby(z["state"]).mean()
    rows = []
    for mth in ("JOMI stud", "JOMI CQR-QRF", "SC-GeoCCP", "SC-GeoCCP CQR-QRF"):
        for _, r in g[g.method == mth].iterrows():
            rows.append(dict(state=FIA_STATE[int(r.group)], code=int(r.group), method=S.SHORT[mth], key=mth,
                             coverage=r.c / r.n, n_selected_pooled=int(r.n), mean_biomass=mean_y[int(r.group)]))
    T = pd.DataFrame(rows)
    for mth in ("JOMI stud", "JOMI CQR-QRF"):
        x = g[g.method == mth]
        print(f"{mth}: overall quota coverage over states {x.c.sum() / x.n.sum():.3f}")
    return T


def panel_c(fig, T):
    ax = S.axes_mm(fig, *C)
    st = T.drop_duplicates("state").sort_values("mean_biomass").state.tolist()
    y = {s: i for i, s in enumerate(st)}
    ax.set_ylim(len(st) - 0.5, -0.5); ax.set_xlim(-0.02, 1.02)
    S.target(ax)
    meths = ["JOMI stud", "JOMI CQR-QRF", "SC-GeoCCP"]                    # RegionCP (ours) beside JOMI
    off = dict(zip(meths, (-0.24, 0.0, 0.24)))
    for mth in meths:
        x = T[T.key == mth].set_index("state")
        S.dot(ax, x.loc[st, "coverage"], [y[s] + off[mth] for s in st], mth, s=10, z=3)
    lo = T[(T.key == "JOMI CQR-QRF")].set_index("state")
    low = lo.coverage.idxmin(); r = lo.loc[low]                           # the lowest state, from the data
    ax.annotate(f"{r.coverage:.3f}", (r.coverage, y[low] + off["JOMI CQR-QRF"]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=S.FS["note"], color=S.INK2)
    ax.set_yticks(range(len(st))); ax.set_yticklabels(st, fontsize=S.FS["tick"])
    ax.tick_params(axis="y", length=0); S.clean(ax, left=False)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9]); ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "0.9"])
    ax.set_xlabel("Coverage among selected plots in the state", labelpad=1.5)
    ax.set_ylabel("State, low → high mean biomass", fontsize=S.FS["note"], color=S.INK2, labelpad=2)
    ax.text(0.02, len(st) - 2.0, "JOMI, pooled over all states:\n0.90 with either score", ha="left", va="center",
            fontsize=S.FS["note"], color=S.INK2, fontstyle="italic", linespacing=1.15)
    hs = [S.handle("JOMI stud", label="JOMI, normalized score"), S.handle("JOMI CQR-QRF", label="JOMI, additive QRF score"),
          S.handle("SC-GeoCCP", label="RegionCP (ours), normalized score")]
    rc = T[T.key == "SC-GeoCCP"]
    print(f"RegionCP over states: {rc.coverage.min():.3f}-{rc.coverage.max():.3f} (lowest {rc.loc[rc.coverage.idxmin(), 'state']})")
    ax.legend(handles=hs, ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.2,
              columnspacing=1.0, borderaxespad=0.2)
    return ax


def main():
    S.apply()
    Ta, Tc = data_a(), data_c()
    fig = S.figure(W, H)
    axa, axc = panel_a(fig, Ta), panel_c(fig, Tc)
    S.letter_mm(fig, 2.0, H - 1.5, "a"); S.letter_mm(fig, 101.0, H - 1.5, "b")
    S.source_data("nc_fig5a", Ta); S.source_data("nc_fig5b", Tc[Tc.key.isin(["JOMI stud", "JOMI CQR-QRF", "SC-GeoCCP"])])
    stale = os.path.join(S.SRC, "nc_fig5c.csv") if hasattr(S, "SRC") else None
    if stale and os.path.exists(stale):
        os.remove(stale)                                              # the old panel c is now b
    S.save(fig, "nc_fig5_alternatives", axes=[axa, axc], panel_ids=["a", "b"], row_groups=[["a", "b"]],
           column_groups=[])
    med = Ta.groupby("label")[["width_rel_jomi", "coverage"]].median().round(3)
    print(med.to_string())


if __name__ == "__main__":
    main()
