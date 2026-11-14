"""Fig. 4 | How common is the winner's curse? Coverage among the selected units (paper/FIGURE_PLAN.md, result 1).

a  11 random-forest settings and, below them under their own header, the 4 settings with frozen-DOFA maps
   (paper/DOFA_FORMAL_DESIGN.md, added 2026-11-08) x 12 published or selective methods at the policy-direction 5% rule
   (top or bottom), one facet per method family; (n | m) after a method = random-forest | DOFA settings with
   coverage < 0.85. Supplementary Fig. 2 (make_si_budget.py) reuses panel_a without the DOFA block.
b  coverage as selection grows stricter (K/m = 20, 5, 1%), top-K and bottom-K, grouped by method: Normalized CP,
   GeoCP, JOMI and RegionCP (ours; 1% and 5% only, on a light violet band); median, middle half and range over the
   11 random-forest settings.
Bars in a: 95% intervals over the 100 random splits. Data: results/main_*, bl_main_*, scg_main_*_l1fix.
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_fig4.py
writes figs/nc_fig4_prevalence.{pdf,svg,png}, figs/source_data/nc_fig4{a,b}.csv, figs/qa/nc_fig4_prevalence.*
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

W, H = 183, 130
FACETS = [("Standard conformal", ["SCP |res|", "SCP stud", "CQR", "BY stud"]),
          ("Local conformal", ["GeoCP", "LSCP", "RLCP"]),
          ("Model-based", ["Regression kriging", "QRF", "Bootstrap ensemble + residual"]),
          ("Selection-conditional", ["JOMI stud", "SC-GeoCCP"])]
SHORT = {"BY stud": "BY", "Regression kriging": "Kriging", "SC-GeoCCP": "RegionCP"}
B_METH = ["SCP stud", "GeoCP", "JOMI stud"]
# layout (mm)
AX0, AW, AGAP, AY0, AH = 31.0, 35.75, 2.5, 62.0, 54.0          # AH: height of the 11-row panel; rows keep its pitch
BY0, BH, BW, BX = 12.0, 29.0, 77.0, (14.0, 104.5)
TOP = 14.0                                                     # mm above panel a (facet titles and legends)
# the formal DOFA settings (paper/DOFA_FORMAL_DESIGN.md): below the random-forest rows, own header; †product label
DOFA_ROWS = [("dofa_poverty", "DHS wealth, 13 countries", "dofa"), ("dofa_m3leo", "M3LEO biomass†", "dofa"),
             ("dofa_floods", "UrbanSARFloods", "dofa"), ("dofa_usavars", "USAVars tree cover†", "dofa")]
DOFA_HEAD = "Frozen DOFA map + linear probe"


def policy_arrow(nm):
    return "↓" if D.DS.POLICY[nm] == "bottom" else "↑"


def rows_layout(rows):
    """D.dataset_rows() for any list of (setting, label, truth) rows: a header gap before each ground-truth group,
    and one for the DOFA block. Returns ({setting: y}, [(y, header)], total height)."""
    y, out, heads, last = 0.0, {}, [], None
    for nm, _, tr in rows:
        if tr != last:
            heads.append((y + 0.45, DOFA_HEAD if tr == "dofa" else S.TRUTH[tr])); y += 0.9; last = tr
        out[nm] = y + 0.5; y += 1.0
    return out, heads, y


def height_a(extra=()):
    """Height (mm) of panel a: the 11-row panel is AH high, and added rows keep its row pitch."""
    return AH * rows_layout(list(S.DATASETS) + list(extra))[2] / rows_layout(S.DATASETS)[2]


def panel_a(fig, P, rule=D.policy_rule, arrow=policy_arrow,
            key="↑ / ↓  top / bottom 5% selected\n(n)  settings below 0.85, of 11", extra=()):
    """Coverage among the selected units by setting and method; `rule` maps a setting to its selection rule
    (default: the policy-direction 5% rule; Supplementary Fig. 2 passes the travel-cost budget). `extra`: rows
    added below the random-forest settings under their own header (Fig. 4a: the DOFA settings), counted separately."""
    rows = list(S.DATASETS) + list(extra)
    Y, heads, ymax = rows_layout(rows)
    AH = height_a(extra)                                                    # noqa: N806 (shadows the 11-row height)
    axs, src = [], []
    for k, (title, meths) in enumerate(FACETS):
        ax = S.axes_mm(fig, AX0 + k * (AW + AGAP), AY0, AW, AH); axs.append(ax)
        ax.set_xlim(0.15, 1.02); ax.set_ylim(ymax, 0)
        S.hgrid(ax, Y.values()); S.target(ax)
        off = np.linspace(-0.27, 0.27, len(meths)) if len(meths) > 1 else [0.0]
        hs = []
        for o, mth in zip(off, meths):
            c, vals, vals_x = S.METHODS[mth][1], [], []
            for nm, _, tr in rows:
                r = P[nm].loc[(rule(nm), mth)]
                ax.plot([r.lo, r.hi], [Y[nm] + o] * 2, color=c, lw=S.LW["err"], zorder=2, solid_capstyle="butt")
                S.dot(ax, [r["cov"]], [Y[nm] + o], mth, s=10)   # same marker for both maps: several methods are open
                (vals_x if tr == "dofa" else vals).append(r["cov"])
                row = dict(dataset=nm, rule=rule(nm), method=S.SHORT[mth], coverage=r["cov"],
                           ci_lo=r.lo, ci_hi=r.hi, inf_share=r.inf, n_selected=r.K, splits=r.splits)
                src.append({"map": "DOFA probe" if tr == "dofa" else "random forest", **row} if extra else row)
            n_bad = int((np.array(vals) < 0.85).sum())
            n_x = f"|{int((np.array(vals_x) < 0.85).sum())}" if extra else ""
            hs.append(S.handle(mth, label=f"{SHORT.get(mth, S.SHORT[mth])} ({n_bad}{n_x})"))
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 0.9]); ax.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "0.9"])
        ax.tick_params(axis="y", length=0)
        S.clean(ax, left=False)
        S.text_mm(fig, AX0 + k * (AW + AGAP), AY0 + AH + 8.0, title, fontsize=S.FS["title"], fontweight="bold",
                  ha="left", va="bottom")
        ax.legend(handles=hs, ncol=2, loc="upper left", bbox_to_anchor=(-0.03, 1 + 7.4 / AH), handletextpad=0.15,
                  columnspacing=0.7, borderaxespad=0.15, labelspacing=0.3)
        if k == 0:
            ax.set_yticks(list(Y.values()))
            ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=S.FS["label"])
            ax.tick_params(axis="y", pad=6.5)
            for nm, *_ in rows:              # selection direction beside the name
                ax.text(-0.015, Y[nm], arrow(nm), transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=S.FS["tick"], color=S.MUTED)
            for yy, lab in heads:
                S.text_mm(fig, 2.0, AY0 + AH * (1 - yy / ymax), lab, ha="left", va="center",
                          fontsize=S.FS["note"], color=S.INK2, fontstyle="italic")
        else:
            ax.set_yticks([])
    S.text_mm(fig, AX0 + (4 * AW + 3 * AGAP) / 2, AY0 - 6.6, "Coverage among selected units", ha="center",
              va="center", fontsize=S.FS["label"])
    S.text_mm(fig, 2.0, AY0 + AH + 1.0, key,
              ha="left", va="bottom", fontsize=S.FS["note"], color=S.MUTED, linespacing=1.25)
    return axs, pd.DataFrame(src)


B_GROUPS = B_METH + ["SC-GeoCCP"]                        # RegionCP (ours), last, on a light violet band
B_SHARES = (20, 5, 1)                                    # left to right: selection grows stricter within each method
B_OFF = dict(zip(B_SHARES, (-0.27, 0.0, 0.27)))           # column offsets inside a method group (group centres 0, 1, 2)


def panel_b(fig, P):
    """Coverage as selection grows stricter (20% -> 5% -> 1% of the test units), grouped by method: for each method and
    share, the median of the 11 random-forest settings (the method's marker), the middle half (thick bar) and the range
    (thin line). No lines join the shares. Redrawn 2026-11-08 (was three methods dodged inside each share, every setting
    as a jittered dot and medians joined by lines, which hid the differences between methods)."""
    axs, src = [], []
    for k, d in enumerate(("top", "bottom")):
        ax = S.axes_mm(fig, BX[k], BY0, BW, BH); axs.append(ax)
        ng = len(B_GROUPS)
        S.target(ax, orient="h"); ax.set_xlim(-0.55, ng - 0.45); ax.set_ylim(0.45, 1.07)   # room for the ∞ labels
        ax.axvspan(ng - 1.5, ng - 0.45, color=S.VIOLET_BAND, lw=0, zorder=0)        # this paper's method
        S.hgrid(ax, [0.5, 0.6, 0.7, 0.8, 1.0])
        for g in np.arange(0.5, ng - 1):                                           # separators between methods
            ax.axvline(g, color=S.GRID, lw=0.5, zorder=0)
        for gi, mth in enumerate(B_GROUPS):
            c = S.METHODS[mth][1]
            for pct in B_SHARES:
                if mth == "SC-GeoCCP" and pct == 20:                             # RegionCP ran at 1% and 5% only
                    ax.text(gi + B_OFF[pct], 0.75, "not\nrun", ha="center", va="center", fontsize=S.FS["note"],
                            color=S.MUTED, linespacing=1.25, zorder=4)              # between the 0.7 and 0.8 lines
                    continue
                v, n_inf = [], 0
                for nm, *_ in S.DATASETS:
                    r = P[nm].loc[(f"{d}{pct}", mth)]
                    v.append(r["cov"]); n_inf += int(mth in ("JOMI stud", "SC-GeoCCP") and r.inf > 0.05)
                    src.append(dict(direction=d, pct=pct, dataset=nm, method=S.SHORT[mth], coverage=r["cov"],
                                    ci_lo=r.lo, ci_hi=r.hi, inf_share=r.inf, n_selected=r.K))
                v = np.array(v); x = gi + B_OFF[pct]
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([x, x], [v.min(), v.max()], color=c, lw=0.8, zorder=2, solid_capstyle="butt")
                ax.plot([x, x], [q1, q3], color=c, lw=4.2, zorder=3, solid_capstyle="butt", alpha=0.55)
                S.dot(ax, [x], [med], mth, s=20, z=5, edge=S.INK)
                if n_inf:                                                          # JOMI's infinite intervals at 1%
                    ax.text(x, 1.008, f"∞ {n_inf}", ha="center", va="bottom", fontsize=S.FS["note"],   # above the 1.0 line
                            color=S.INK2)
        ax.set_xticks([gi + B_OFF[p] for gi in range(ng) for p in B_SHARES])
        ax.set_xticklabels([f"{p}%" for _ in B_GROUPS for p in B_SHARES])
        ax.tick_params(axis="x", length=2, pad=1.5)
        for gi, mth in enumerate(B_GROUPS):                                       # method names under each group
            ax.text(gi, -0.2, S.SHORT[mth], transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=S.FS["label"], color=S.INK)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
        S.text_mm(fig, BX[k], BY0 + BH + 1.2, "Top-K: highest predictions" if d == "top" else
                  "Bottom-K: lowest predictions", fontsize=S.FS["title"], fontweight="bold", ha="left", va="bottom")
        if k == 0:
            ax.set_ylabel("Coverage among selected")
            r = P[D.DS.FIA].loc[("top1", "GeoCP")]
            ax.annotate("FIA forest\nbiomass", (1 + B_OFF[1], r["cov"]), xytext=(1 + B_OFF[1] - 0.09, 0.55),
                        textcoords="data", fontsize=S.FS["note"], color=S.INK2, ha="right", va="center",
                        linespacing=1.1, arrowprops=dict(arrowstyle="-", lw=0.4, color=S.MUTED, shrinkA=0.5, shrinkB=1.0))
                        # inside the GeoCP group, between the 0.5 and 0.6 grid lines (no line crosses the label)
    hs = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=S.INK2, mec=S.INK, mew=S.LW["marker"],
                 label="Median of 11 random-forest settings"),
          Line2D([], [], color=S.INK2, lw=4.2, alpha=0.55, solid_capstyle="butt", label="Middle half"),
          Line2D([], [], color=S.INK2, lw=0.8, label="Range"),
          Line2D([], [], ls="none", label="∞ k: settings (of 11) with > 5% infinite intervals")]
    fig.legend(handles=hs, ncol=4, loc="lower left", bbox_to_anchor=(BX[0] / W, (BY0 + BH + 4.6) / H),
               handletextpad=0.35, handlelength=1.4, columnspacing=1.4, borderaxespad=0)
    return axs, pd.DataFrame(src)


def main():
    global H
    S.apply()
    P = {nm: D.pooled(D.runs(nm)) for nm, *_ in list(S.DATASETS) + DOFA_ROWS}
    H = AY0 + height_a(DOFA_ROWS) + TOP                 # 130 mm without the DOFA block
    fig = S.figure(W, H)
    A, srcA = panel_a(fig, P, extra=DOFA_ROWS,
                      key="↑ / ↓  top / bottom 5%\n(n|m)  settings below 0.85,\n"
                          "of 11 random-forest | 4 DOFA")
    B, srcB = panel_b(fig, P)
    S.letter_mm(fig, 2.0, H - 1.5, "a"); S.letter_mm(fig, 2.0, BY0 + BH + 10.5, "b")
    S.source_data("nc_fig4a", srcA); S.source_data("nc_fig4b", srcB)
    S.save(fig, "nc_fig4_prevalence", axes=A + B, panel_ids=["a1", "a2", "a3", "a4", "b1", "b2"],
           row_groups=[["a1", "a2", "a3", "a4"], ["b1", "b2"]], column_groups=[])
    for m, t in srcA.groupby("map"):
        pr = t.pivot_table(index="dataset", columns="method", values="coverage")
        print(f"{m}: settings with coverage < 0.85 (of {len(pr)}):", (pr < 0.85).sum().to_dict())


if __name__ == "__main__":
    main()
