"""Supplementary Fig. 24 | Carbon crediting with an operational biomass map (paper/BIOMASS_PRODUCT_DESIGN.md).

The map is ESA CCI Biomass v7.0 (2020) at the FIA plots; truth is FIA above-ground live-tree biomass; VCS rule (declared
risk 1/3). Realized over-crediting risk of the project total against the top share of the candidate plots, same series
and markers as Fig. 7a-c (the product's AGB_SD is the error layer):
a  the product selects, values averaged over a 1 km window around the public plot coordinates (main);
b  the same K candidates chosen at random (control);
c  the product selects, centre pixel (sensitivity).
Audit share 0.5; 20 subsets of 6,000 plots x 100 audits (experiments/run_carbon_product.py). Subset-clustered 95%
intervals are in the source data (results/fia_db/product/product_summary.csv).
    FIGURE_QA_HELPERS=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_si_carbon_product.py
writes figs/si_fig_carbon_product.{pdf,svg,png}, figs/source_data/si_fig_carbon_product.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import make_fig7 as F7                                                # noqa: E402

MC = F7.MC
RES = os.path.join(MC.DS.FIA_RES, "product")
W, H = 183, 66
PANELS = [(14.0 + 58.0 * i, 9.5, 50.0, 36.0) for i in range(3)]
COL = {"point": "point", "map": "VCS map", "ma": "VCS model-assisted", "ours": "ours: VCS formula, sel.-aware",
       "exact": "ours: exact, risk VCS"}                               # product run columns, styled as Fig. 7's roles
SETS = [("win", "", "Product selects (1 km window)"), ("win", "_random", "Random selection, same K"),
        ("ctr", "", "Product selects (centre pixel)")]


def data():
    rows = []
    for v, tag, title in SETS:
        d = pd.read_csv(os.path.join(RES, f"carbon_product_{v}{tag}.csv.gz")); d = d[d.audit == 0.5]
        for role, c in list(COL.items()) + [("fixed", F7.FIXED["key"])]:
            for f, g in d.groupby("frac"):
                rows.append(dict(panel=title, variant=v, selection="random" if tag else "product", role=role, column=c,
                                 top_share=f, risk=float(np.mean(g[c] > g.total)), credited_over_true=g[c].sum() / g.total.sum(),
                                 heldout_r2=float(g.groupby("fold").r2.first().median())))
    return pd.DataFrame(rows)


def main():
    matplotlib.rcdefaults(); S.apply()
    T = data(); fig = S.figure(W, H); key = F7.KEYS["VCS"]
    axs = [S.axes_mm(fig, *r) for r in PANELS]
    x = np.sort(T.top_share.unique()) * 100
    for i, (ax, (v, tag, title)) in enumerate(zip(axs, SETS)):
        t = T[T.panel == title]
        r = t[t.role == "fixed"].sort_values("top_share")
        ax.plot(x, r.risk.values, zorder=2.5, gid=f"fixed-deciles-{'abc'[i]}", **F7.fixed_kw())
        for role in COL:
            r = t[t.role == role].sort_values("top_share")
            ax.plot(x, r.risk.values, lw=S.LW["line"], zorder={"ours": 4, "point": 3.5}.get(role, 3),
                    **F7.marker_kw(key[role]))
        S.target(ax, v=1 / 3, orient="h"); S.hgrid(ax, [0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xlim(0.4, 12.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{u:g}" for u in x])
        ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        if i:
            ax.set_yticklabels([])
        S.text_mm(fig, PANELS[i][0], PANELS[i][1] + PANELS[i][3] + 1.2,
                  f"{title} (R² = {t.heldout_r2.iloc[0]:.2f})", fontsize=S.FS["label"], ha="left", va="bottom")
        S.letter_mm(fig, PANELS[i][0] - 7.0, PANELS[i][1] + PANELS[i][3] + 1.2, "abc"[i])
    axs[0].set_ylabel("Realized over-crediting risk")
    axs[1].set_xlabel("Top share of the candidate plots (%)")
    hs = [Line2D([], [], lw=S.LW["line"], label=F7.label(r).replace("Map", "Product") if r == "point" else
                 ("Protocol rule, product error layer" if r == "map" else F7.label(r)), **F7.marker_kw(key[r])) for r in COL]
    hs.append(Line2D([], [], color=S.INK2, lw=S.LW["ref"], ls=(0, (3, 2)), label=F7.DECLARED["VCS"]))
    hs.append(Line2D([], [], label=F7.FIXED["label"], **F7.fixed_kw()))
    fig.legend(handles=hs, ncol=4, loc="lower left", bbox_to_anchor=(PANELS[0][0] / W, 52.5 / H),
               handlelength=2.0, columnspacing=1.4, borderaxespad=0)
    S.source_data("si_fig_carbon_product", T)
    S.save(fig, "si_fig_carbon_product", axes=axs, panel_ids=list("abc"), row_groups=[list("abc")], column_groups=[])
    p = T[np.isclose(T.top_share, 0.01)].pivot(index="role", columns="panel", values="risk")
    print("top 1% risk:\n" + p.round(3).to_string())


if __name__ == "__main__":
    main()
