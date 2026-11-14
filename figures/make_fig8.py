"""Fig. 8 | Is coverage valid in every place? Paying by scale (paper/FIGURE_PLAN.md, result 5).

a  quota rules (a fixed share selected in every state or country): coverage in the worst state / country for
   Normalized CP, GeoCP, JOMI, JOMI calibrated within each state, and SC-GeoCCP (l1 = 0.95); tick = the
   expected worst group if every group had exactly 0.90 (independent binomial, fig_data.noise_limit); right: share of
   infinite intervals among the selected units.
b  where SC-GeoCCP's guarantee can be afforded (FIA, state quota): Theorem 1 lower bound for every basis
   function (bump at three scales, state indicator), each plot coloured by the function it belongs to
   (experiments/run_scg_slack.py).
c  worst group minus the sampling-noise limit at three scales (all selected units, state / country, spatial
   cell) for JOMI and SC-GeoCCP; thin lines pair the two methods on the same setting.
d  FIA state quota, realised coverage shortfall below 0.90 in each state (choropleth): GeoCP vs SC-GeoCCP.
Data: results/scg_groups_*, scg_groups_*_l1fix, bl_groups_*, main_*, scg_main_*_l1fix, scg_slack_<FIA>.csv;
state outlines from Natural Earth 1:110m (figures/basemap.py; downloaded files, hence python -I).
    FIGURE_QA_HELPERS=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python -I figures/make_fig8.py
writes figs/nc_fig8_scales.{pdf,svg,png}, figs/source_data/nc_fig8{a,b,c,d}.csv, figs/qa/
"""
import os, sys
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402
import basemap as BM                                                  # noqa: E402

W, H = 183, 142
A_METH = ["SCP stud", "GeoCP", "JOMI stud", "JOMI|state stud", "SC-GeoCCP"]
INF_METH = ["JOMI stud", "JOMI|state stud", "SC-GeoCCP"]
C_METH = ["JOMI stud", "SC-GeoCCP"]
FIA_STATE = D.fia_states()                                            # group code -> state (fig_data.fia_states)
NOGO = "#d9d8d3"                                    # no guarantee / no data
# layout (mm)
A1, A2, CC = (33.0, 81.0, 63.0, 48.0), (100.0, 81.0, 22.0, 48.0), (141.0, 81.0, 40.0, 48.0)
EXT = (-10900, -5650, 2700, 5500)        # km; margins keep the outline clear of the map titles and labels
MW = 46.0; MH = MW * (EXT[3] - EXT[2]) / (EXT[1] - EXT[0])
MAPX_B, MAPX_D, MAPY = (3.0, 51.0), 116.0, (35.5, 6.5)


# ------------------------------------------------------------------ a
def data_a():
    rows = []
    for nm, *_ in S.DATASETS:
        g = D.groups(nm, "state"); r = D.quota_rule(nm); P = D.pooled(D.runs(nm))
        for mth in A_METH:
            w, wi, ng, _ = D.worst(g, r, mth); nul = D.noise_limit(nm, "state")
            rows.append(dict(dataset=nm, rule=r, method=S.SHORT[mth], key=mth, worst=w, inf_in_worst=wi, groups=ng,
                             noise_limit=nul, inf_share=P.loc[(r, mth)].inf if (r, mth) in P.index else np.nan))
    return pd.DataFrame(rows)


def panel_a(fig, T):
    Y, heads, ymax = D.dataset_rows()
    a1, a2 = S.axes_mm(fig, *A1), S.axes_mm(fig, *A2)
    off = dict(zip(A_METH, np.linspace(-0.3, 0.3, len(A_METH))))
    for ax, xl in ((a1, (0.2, 1.02)), (a2, (-0.03, 1.03))):
        ax.set_ylim(ymax, 0); ax.set_xlim(*xl); S.hgrid(ax, Y.values())
        ax.tick_params(axis="y", length=0); S.clean(ax, left=False)
    S.target(a1)
    for nm, *_ in S.DATASETS:
        x = T[T.dataset == nm].set_index("key")
        a1.scatter([x.loc["JOMI stud", "noise_limit"]], [Y[nm]], marker="|", s=34, color=S.INK2, linewidths=0.8, zorder=4)
        for mth in A_METH:
            S.dot(a1, [x.loc[mth, "worst"]], [Y[nm] + off[mth]], mth, s=10, z=3)
        for mth in INF_METH:
            S.dot(a2, [x.loc[mth, "inf_share"]], [Y[nm] + off[mth]], mth, s=10, z=3)
    a1.set_yticks(list(Y.values())); a1.set_yticklabels([lab for _, lab, _ in S.DATASETS], fontsize=S.FS["label"])
    a1.tick_params(axis="y", pad=6.5)
    for nm, *_ in S.DATASETS:
        # arrow: the direction of the quota rule itself (only DHS takes the lowest), not the policy direction of Fig. 4a
        a1.text(-0.015, Y[nm], "↓" if D.quota_rule(nm) == "quota_bottom" else "↑", transform=a1.get_yaxis_transform(),
                ha="right", va="center", fontsize=S.FS["tick"], color=S.MUTED)
    for yy, lab in heads:
        S.text_mm(fig, 2.0, A1[1] + A1[3] * (1 - yy / ymax), lab, ha="left", va="center", fontsize=S.FS["note"],
                  color=S.INK2, fontstyle="italic")
    a2.set_yticks([])
    a1.set_xticks([0.2, 0.4, 0.6, 0.8, 0.9]); a1.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "0.9"])
    a2.set_xticks([0, 0.5, 1]); a2.set_xticklabels(["0", "0.5", "1"])
    a1.set_xlabel("Coverage in the worst state or country"); a2.set_xlabel("Share of +∞ intervals")
    hs = [S.handle(m) for m in A_METH] + [Line2D([], [], ls="none", marker="|", ms=5.5, mew=0.8, color=S.INK2,
                                                 label="Sampling-noise limit")]
    fig.legend(handles=hs, ncol=3, loc="lower left", bbox_to_anchor=(A1[0] / W, (A1[1] + A1[3] + 1.6) / H),
               handletextpad=0.25, columnspacing=1.1, borderaxespad=0, labelspacing=0.35)
    return [a1, a2]


# ------------------------------------------------------------------ c
def data_c():
    rows = []
    for nm, *_ in S.DATASETS:
        r = D.quota_rule(nm); P = D.pooled(D.runs(nm))
        gs, gc = D.groups(nm, "state"), D.groups(nm, "cell")
        for mth in C_METH:
            rows.append(dict(dataset=nm, method=S.SHORT[mth], key=mth, scale="All selected", level=0,
                             worst=P.loc[(r, mth), "cov"], noise_limit=1 - S.ALPHA))
            for lv, (lab, g) in enumerate((("State or country", gs), ("Spatial cell", gc)), start=1):
                w, _, ng, _ = D.worst(g, r, mth); nul = D.noise_limit(nm, "state" if lv == 1 else "cell")
                rows.append(dict(dataset=nm, method=S.SHORT[mth], key=mth, scale=lab, level=lv, worst=w,
                                 noise_limit=nul, groups=ng))
    T = pd.DataFrame(rows); T["gap"] = T.worst - T.noise_limit
    return T


def panel_c(fig, T):
    ax = S.axes_mm(fig, *CC)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.43, 0.07)
    S.hgrid(ax, [-0.3, -0.2, -0.1]); S.target(ax, v=0.0, orient="h")
    off = {"JOMI stud": -0.17, "SC-GeoCCP": 0.17}
    for nm, *_ in S.DATASETS:
        x = T[T.dataset == nm]
        for lv in range(3):
            v = x[x.level == lv].set_index("key").gap
            ax.plot([lv + off[m] for m in C_METH], [v[m] for m in C_METH], color=S.GRID, lw=0.5, zorder=1)
    for mth in C_METH:
        x = T[T.key == mth]
        S.dot(ax, x.level + off[mth], x.gap, mth, s=6, z=3, alpha=0.85)
        med = x.groupby("level").gap.median()
        ax.plot(med.index + off[mth], med.values, color=S.METHODS[mth][1], lw=S.LW["line"], zorder=4)
        S.dot(ax, med.index + off[mth], med.values, mth, s=24, z=5, edge=S.INK)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["All\nselected", "State or\ncountry", "Spatial\ncell"])
    ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0.0]); ax.set_yticklabels(["−0.4", "−0.3", "−0.2", "−0.1", "0"])
    ax.set_ylabel("Worst group minus sampling-noise limit")
    ax.text(2.45, 0.012, "limit", ha="right", va="bottom", fontsize=S.FS["note"], color=S.INK2)
    for nm, lv, lab in (("health_stroke", 1, "Stroke*"), ("seattle", 2, "Seattle")):
        v = T[(T.dataset == nm) & (T.level == lv) & (T.key == "JOMI stud")].gap.iloc[0]
        ax.annotate(lab, (lv + off["JOMI stud"], v), xytext=(-6, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=S.FS["note"], color=S.INK2)
    S.text_mm(fig, CC[0], CC[1] + CC[3] + 1.2, "Worst group across scales", fontsize=S.FS["title"],
              fontweight="bold", ha="left", va="bottom")
    hs = [Line2D([], [], ls="-", lw=S.LW["line"], color=S.INK2, marker="o", ms=4.2, mfc=S.INK2, mec=S.INK,
                 mew=S.LW["marker"], label="Median of 11"), Line2D([], [], ls="-", lw=0.5, color=S.GRID,
                                                              label="Same setting")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.3, borderaxespad=0.3)
    return ax


# ------------------------------------------------------------------ b, d
@lru_cache(None)
def fia():
    """Plot coordinates (km), state codes, and the lon / lat -> km map that put the outlines on them."""
    z = D.DS.load(D.DS.FIA)
    return z["coords"], z["state"], BM.km_projection(z["coords"], z["Xrf"][:, 0], z["Xrf"][:, 1])


@lru_cache(None)
def states():
    return BM.conus()


def basemap(ax, fills=None):
    proj = fia()[2]
    if fills:
        BM.fill(ax, states(), proj, fills)
    BM.outline(ax, states(), proj, coast=BM.COAST)


def map_axes(fig, x, y):
    ax = S.axes_mm(fig, x, y, MW, MH)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def paint(ax, s, val, cmap, norm, bad=NOGO):
    col = np.array([bad] * len(val), dtype=object)
    ok = np.isfinite(val)
    col = [cmap(norm(v)) if k else bad for v, k in zip(val, ok)]
    ax.scatter(s[:, 0], s[:, 1], s=0.35, c=col, linewidths=0, rasterized=True, zorder=2)


def data_b():
    sl = pd.read_csv(os.path.join(S.RES, f"scg_slack_{D.DS.FIA}.csv"))
    s, st, _ = fia()
    out, maps = [], []
    for lv, lab in ((0, "4 regional bumps"), (1, "12 bumps"), (2, "36 bumps")):
        b = sl[(sl.kind == "bump") & (sl.level == lv)]
        d2 = ((s[:, None, :] - b[["cx", "cy"]].values[None, :, :]) ** 2).sum(-1)
        own = d2.argmin(1); fl = b.floor.values[own]
        maps.append((lab, fl))
        out += [dict(map=lab, centre_x=r.cx, centre_y=r.cy, bandwidth=r.h, mean_f_selected=r.Ef, slack=r.slack,
                     floor=r.floor) for r in b.itertuples()]
    u = sl[sl.kind == "unit"].set_index("unit")
    maps.append(("States (quota units)", u.floor.reindex(st).values))
    out += [dict(map="States (quota units)", state=FIA_STATE[int(k)], mean_f_selected=r.Ef, slack=r.slack, floor=r.floor)
            for k, r in u.iterrows()]
    return maps, pd.DataFrame(out), sl.nbar.iloc[0]


def panel_b(fig, maps):
    cmap, norm = S.ramp("Purples", 0.22), Normalize(0, 1 - S.ALPHA)
    s = fia()[0]; axs = []
    for k, (lab, fl) in enumerate(maps):
        ax = map_axes(fig, MAPX_B[k % 2], MAPY[k // 2]); axs.append(ax)
        v = np.where(np.isfinite(fl) & (fl > 0), fl, np.nan)
        paint(ax, s, v, cmap, norm); basemap(ax)
        ax.text(0, 1.0, lab, transform=ax.transAxes, fontsize=S.FS["note"], ha="left", va="bottom", color=S.INK)
    cax = S.axes_mm(fig, 99.0, 13.0, 1.8, 44.0)
    cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax)
    cb.set_ticks([0, 0.3, 0.6, 0.9]); cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=1.5, width=0.4)
    cb.set_label("Guaranteed coverage (Theorem 1)", fontsize=S.FS["label"], labelpad=2)
    cax.legend(handles=[Patch(facecolor=NOGO, edgecolor="none", label="None")], loc="lower left",
               bbox_to_anchor=(-0.3, 1.015), handlelength=0.8, handleheight=0.8, borderaxespad=0, handletextpad=0.3)
    return axs, cax


def data_d():
    g = D.groups(D.DS.FIA, "state"); g = g[(g.rule == "quota") & (g.n >= D.NMIN)]
    rows = []
    for mth in ("GeoCP", "SC-GeoCCP"):
        for r in g[g.method == mth].itertuples():
            rows.append(dict(method=S.SHORT[mth], key=mth, state=FIA_STATE[int(r.group)], code=int(r.group),
                             coverage=r.c / r.n, shortfall=max(0.0, 1 - S.ALPHA - r.c / r.n), n_selected_pooled=int(r.n)))
    return pd.DataFrame(rows)


# state labels go inside the state when the label box fits there with this margin (label_spot); the others sit
# at a fixed lon / lat in open space, VT with a leader line to the state
FIT = 1.1
OUTSIDE = {"FL": (-78.4, 28.0, False), "ME": (-69.0, 48.55, False), "VT": (-72.9, 46.4, True)}


def state_labels(ax, fills):
    """Postal codes of the FIA states, in whichever of white / ink reads better on the fill."""
    fit = BM.label_features(ax, states(), fia()[2], {k: k for k in fills}, fills, fit=FIT, outside=OUTSIDE)
    print("state labels (margin; inside if >= %.1f): " % FIT + ", ".join(f"{k} {v:.2f}" for k, v in fit.items()))


def panel_d(fig, T):
    cmap, norm = S.ramp("Reds", 0.18), Normalize(0, 0.65)
    axs = []
    for k, mth in enumerate(("GeoCP", "SC-GeoCCP")):
        ax = map_axes(fig, MAPX_D, MAPY[k]); axs.append(ax)
        x = T[T.key == mth].set_index("code")
        fills = {FIA_STATE[c]: S.GREY_L if v <= 0 else cmap(norm(v)) for c, v in x.shortfall.items()}
        basemap(ax, fills)
        worst = x.coverage.idxmin()
        ax.text(0, 1.0, f"{S.SHORT[mth]}: worst state {x.coverage.min():.2f} ({FIA_STATE[worst]})",
                transform=ax.transAxes, fontsize=S.FS["note"], ha="left", va="bottom", color=S.INK)
        if k == 0:
            state_labels(ax, fills)
    cax = S.axes_mm(fig, 164.0, 13.0, 1.8, 44.0)
    cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax)
    cb.set_ticks([0, 0.2, 0.4, 0.6]); cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=1.5, width=0.4)
    cb.set_label("Coverage shortfall below 0.90 in the state", fontsize=S.FS["label"], labelpad=2)
    cax.legend(handles=[Patch(facecolor=S.GREY_L, edgecolor="none", label="≥ 0.90"),
                        Patch(facecolor="white", edgecolor=BM.COAST, linewidth=0.4, label="No data")],
               loc="lower left", bbox_to_anchor=(-0.3, 1.015), handlelength=0.8, handleheight=0.8, borderaxespad=0,
               handletextpad=0.3, labelspacing=0.3)
    return axs, cax


def main():
    S.apply()
    Ta, Tc, Td = data_a(), data_c(), data_d()
    maps, Tb, nbar = data_b()
    fig = S.figure(W, H)
    A = panel_a(fig, Ta); C = panel_c(fig, Tc)
    B, cb_b = panel_b(fig, maps); Dm, cb_d = panel_d(fig, Td)
    for x, y, l in ((2.0, H - 1.5, "a"), (131.0, H - 1.5, "c"), (2.0, 68.5, "b"), (110.5, 68.5, "d")):
        S.letter_mm(fig, x, y, l)
    S.source_data("nc_fig8a", Ta); S.source_data("nc_fig8b", Tb); S.source_data("nc_fig8c", Tc)
    S.source_data("nc_fig8d", Td)
    S.save(fig, "nc_fig8_scales", axes=A + [C] + B + Dm, panel_ids=["a1", "a2", "c", "b1", "b2", "b3", "b4", "d1", "d2"],
           row_groups=[["a1", "a2"], ["b1", "b2"], ["b3", "b4"]], column_groups=[["b1", "b3"], ["b2", "b4"], ["d1", "d2"]],
           exclude_axes=[cb_b, cb_d])
    print(f"n_bar = {nbar:.1f}")
    print(Tc.groupby(["key", "scale"]).gap.median().round(3).to_string())


if __name__ == "__main__":
    main()
