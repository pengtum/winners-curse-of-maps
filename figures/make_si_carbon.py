"""Supplementary Fig. 15 | Carbon anchor under ART TREES, and the declared risk at any level (paper/FIGURE_PLAN.md, S15).

ART TREES v2.0 deducts 0.524417 standard errors and declares an over-crediting risk of 0.30 (Fig. 7 is the VCS version).
a-c  Fig. 7a-c under ART TREES, drawn by make_fig7.panel_ac: realized over-crediting risk of the project total against
     the selection strength (top 0.5-10% of the candidate plots by the map) for the three maps, same five series.
d-f  top 1%: the protocol formula at any declared risk delta, credit = estimate - Phi^-1(1 - delta) SE (ART TREES is
     delta = 0.30; the VCS deduction of 0.4307 SE is delta = 1/3, leaving aside its 10% waiver and its no-credit rule
     for U > 100%), for the (estimate, SE) pairs of the three protocol series of a-c (map error layer;
     survey-calibrated; selection-aware, ours). Dashed diagonal: realized = declared. At delta = 0.30 (dotted) the
     curves meet the 1% points of a-c (Phi^-1(0.7) = 0.524401 vs the protocol's 0.524417).
Audit share 0.5; 20 maps x 100 audits per map type (experiments/run_carbon.py). Risks are frequencies over all maps and
audits; map-clustered 95% intervals are in the source data (a-c as table C2 of results/RESULTS_CARBON.md; d-f extend
table C7 to every delta).
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_si_carbon.py
writes figs/si_fig_carbon_protocol.{pdf,svg,png}, figs/source_data/si_fig_carbon_protocol_{a_c,d_f}.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import make_fig7 as F7                                                # noqa: E402  (Fig. 7a-c panels; imports make_carbon)

MC = F7.MC
W, H = 183, 116
FAM = "ART"
TOP = [(14.0 + 58.0 * i, 66.0, 50.0, 36.0) for i in range(3)]       # a-c, the geometry of Fig. 7a-c
BOT = [(14.0 + 58.0 * i, 9.5, 50.0, 36.0) for i in range(3)]        # d-f
# (estimate, SE) columns behind each protocol series (experiments/run_carbon.py, credits) and their labels in d-f
EST = {"map": ("point", "se_rf"), "ma": ("ma", "se_ma"), "ours": ("sa", "se_sa")}
ELABEL = {"map": "Map error layer", "ma": "Survey-calibrated estimate", "ours": "Selection-aware estimate (ours)"}
DGRID = np.round(np.arange(1, 101) * 0.005, 3)                       # declared risk 0.005-0.5, so that Phi^-1(1 - delta) >= 0
MARKS = (0.1, 0.2, 0.3, 0.4, 0.5)                                     # markers on the curves


# ------------------------------------------------------------------ data
def data_df(df):
    b, rows, z = df[(df.audit == 0.5) & (df.frac == 0.01)], [], norm.ppf(1 - DGRID)
    for mp in MC.MAPS:
        y = b[b["map"] == mp]
        for role, (e, s) in EST.items():
            with np.errstate(invalid="ignore"):                       # SE = inf when |R| < 2: never over-credited
                over = pd.DataFrame(y[e].values[:, None] - z[None, :] * y[s].values[:, None] > y.total.values[:, None])
            pf = over.groupby(y.fold.values).mean()                   # risk of each map (training fold)
            ci = 1.96 * pf.std() / np.sqrt(len(pf))
            rows += [dict(map=MC.TITLE[mp], code=mp, series=ELABEL[role], estimate=e, se=s, delta=d, z=z[j],
                          risk=over[j].mean(), ci95_halfwidth=ci[j], maps=len(pf), selections=len(y))
                     for j, d in enumerate(DGRID)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ panels
def panel_df(fig, T):
    axs = [S.axes_mm(fig, *r) for r in BOT]
    for i, (ax, mp) in enumerate(zip(axs, MC.MAPS)):
        t = T[T.code == mp]
        for role in EST:
            r = t[t.series == ELABEL[role]].sort_values("delta")
            marks = list(np.flatnonzero(np.isin(r.delta.round(3), MARKS)))
            ax.plot(r.delta, r.risk, lw=S.LW["line"], markevery=marks, zorder={"ours": 4}.get(role, 3),
                    **F7.marker_kw(F7.KEYS[FAM][role]))
        ax.plot([0, 0.5], [0, 0.5], color=S.INK2, lw=S.LW["ref"], ls=(0, (3, 2)), zorder=1)
        ax.axvline(MC.FAM[FAM]["delta"], color=S.INK2, lw=S.LW["ref"], ls=(0, (1, 1.5)), zorder=1)
        S.hgrid(ax, [0.2, 0.4, 0.6, 0.8])
        ax.set_xlim(0, 0.515); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
        ax.set_ylim(-0.03, 0.85); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        if i:
            ax.set_yticklabels([])
        S.text_mm(fig, BOT[i][0], BOT[i][1] + BOT[i][3] + 1.2, f"{MC.TITLE[mp]}, top 1%", fontsize=S.FS["label"],
                  ha="left", va="bottom")
    axs[0].set_ylabel("Realized over-crediting risk")
    axs[1].set_xlabel(r"Declared risk $\delta$ (credit = estimate − $\mathrm{\Phi}^{-1}(1-\delta)$ × SE)")
    hs = [Line2D([], [], lw=S.LW["line"], label=ELABEL[r], **F7.marker_kw(F7.KEYS[FAM][r])) for r in EST]
    hs += [Line2D([], [], color=S.INK2, lw=S.LW["ref"], ls=(0, (3, 2)), label="Realized = declared"),
           Line2D([], [], color=S.INK2, lw=S.LW["ref"], ls=(0, (1, 1.5)), label="ART TREES (δ = 0.30)")]
    fig.legend(handles=hs, ncol=5, loc="lower left", bbox_to_anchor=(BOT[0][0] / W, 51.5 / H), handlelength=2.0,
               columnspacing=1.6, borderaxespad=0)
    return axs


def main():
    matplotlib.rcdefaults(); S.apply()       # importing make_carbon applied make_figs' rcParams (tight bbox, 9 pt)
    df = MC.load()
    Tac, Tdf = F7.data_ac(df, FAM), data_df(df)
    fig = S.figure(W, H)
    A = F7.panel_ac(fig, Tac, FAM, top=TOP, legend_y=108.0); D = panel_df(fig, Tdf)
    for x, y, l in ((2.0, 106.5, "a"), (66.0, 106.5, "b"), (124.0, 106.5, "c"),
                    (2.0, 50.0, "d"), (66.0, 50.0, "e"), (124.0, 50.0, "f")):
        S.letter_mm(fig, x, y, l)
    S.source_data("si_fig_carbon_protocol_a_c", Tac); S.source_data("si_fig_carbon_protocol_d_f", Tdf)
    S.save(fig, "si_fig_carbon_protocol", axes=A + D, panel_ids=list("abcdef"),
           row_groups=[list("abc"), list("def")], column_groups=[["a", "d"], ["b", "e"], ["c", "f"]])
    t = Tac[np.isclose(Tac.top_share, 0.01)].pivot(index="series", columns="map", values="risk")
    print("ART, top 1% risk:\n" + t.round(3).to_string())
    m = Tdf[Tdf.delta.round(3).isin(MARKS)].pivot_table(index=["map", "series"], columns="delta", values="risk")
    print("d-f at the markers:\n" + m.round(3).to_string())
    gap = [abs(Tac[(Tac.code == mp) & (Tac.key == F7.KEYS[FAM][r]) & np.isclose(Tac.top_share, 0.01)].risk.iloc[0]
               - Tdf[(Tdf.code == mp) & (Tdf.series == ELABEL[r]) & np.isclose(Tdf.delta, 0.30)].risk.iloc[0])
           for mp in MC.MAPS for r in EST]
    print(f"d-f at delta = 0.30 vs the 1% points of a-c: max |difference| {max(gap):.4f}")


if __name__ == "__main__":
    main()
