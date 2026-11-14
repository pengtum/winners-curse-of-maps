"""Fig. 6 | Coverage in every region: what RegionCP adds to JOMI (2026-11-10 redesign, so that
RegionCP is compared with JOMI directly). The full earlier version, with normalized CP, GeoCP, four floor maps and the GeoCP state
map, is make_fig8.py -> figs/nc_fig8_scales, now Supplementary Fig. 31.

a  quota rules (5% of the test units allocated to states, countries, climate regions or zones): coverage in the worst
   group for JOMI, JOMI calibrated within each group and RegionCP (l1 = 0.95); light violet band, RegionCP's change from
   JOMI (numbers at right); tick, the sampling-noise limit; right, share of infinite intervals.
b  FIA, state quota: coverage each state is guaranteed before outcomes are measured, JOMI (the floor its overall
   guarantee implies, 1 - alpha / w) against RegionCP (Theorem 1), in classes; the bump floors are in Supplementary Fig. 31b.
c  worst group minus its sampling-noise limit at three scales for JOMI and RegionCP: every setting (small) and the
   median of 11 (large, value written); no lines join settings.
d  FIA, state quota: shortfall of coverage below 0.90 in each state for JOMI and RegionCP.
Data and helpers: make_fig8.py (data_a, data_b, data_c, fia, basemap, paint, state_labels); Natural Earth outlines
(downloaded files, hence python -I).
    FIGURE_QA_HELPERS=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python -I figures/make_fig_regions.py
writes figs/nc_regions.{pdf,svg,png}, figs/source_data/nc_regions_{a,b,c,d}.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import matplotlib.transforms as mtransforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402
import make_fig8 as F8                                                # noqa: E402  (data and map helpers)

W, H = 183, 92
J, JG, R = "JOMI stud", "JOMI|state stud", "SC-GeoCCP"
LAB = {J: "JOMI", JG: "JOMI within each group", R: "RegionCP (ours)"}
# layout (mm): top row a1 | gain | a2 and c; bottom row four maps
A1, A2, CC = (33.0, 44.5, 50.0, 36.0), (100.0, 44.5, 19.0, 36.0), (138.0, 44.5, 43.0, 36.0)
MW = 42.0; MH = MW * (F8.EXT[3] - F8.EXT[2]) / (F8.EXT[1] - F8.EXT[0]); MAPY = 9.5
MAPX = (2.5, 46.5, 95.0, 139.0)                                       # b1, b2 | d1, d2


# ------------------------------------------------------------------ a
def panel_a(fig, T):
    Y, heads, ymax = D.dataset_rows()
    a1, a2 = S.axes_mm(fig, *A1), S.axes_mm(fig, *A2)
    for ax, xl in ((a1, (0.42, 1.01)), (a2, (-0.04, 1.04))):
        ax.set_ylim(ymax, 0); ax.set_xlim(*xl); S.hgrid(ax, Y.values())
        ax.tick_params(axis="y", length=0); S.clean(ax, left=False)
    S.target(a1)
    gx = mtransforms.blended_transform_factory(a1.transAxes, a1.transData)
    for nm, *_ in S.DATASETS:
        x = T[T.dataset == nm].set_index("key")
        j, r, jg = x.loc[J, "worst"], x.loc[R, "worst"], x.loc[JG, "worst"]
        y = Y[nm]
        a1.plot([j, r], [y, y], color=S.VIOLET_BAND if r >= j else S.SHADE, lw=4.2, solid_capstyle="butt", zorder=1)
        a1.scatter([x.loc[J, "noise_limit"]], [y], marker="|", s=30, color=S.INK2, linewidths=0.8, zorder=2)
        S.dot(a1, [jg], [y + 0.28], JG, s=9, z=3)
        S.dot(a1, [j], [y], J, s=11, z=4)
        S.dot(a1, [r], [y], R, s=12, z=6)
        d = r - j
        a1.text(1.035, y, f"{d:+.2f}".replace("-", "−"), transform=gx, ha="left", va="center", fontsize=S.FS["tick"],
                color=S.VIOLET if d >= 0.02 else S.MUTED, fontweight="bold" if d >= 0.02 else "normal")
        S.dot(a2, [x.loc[JG, "inf_share"]], [y + 0.28], JG, s=9, z=3)
        S.dot(a2, [x.loc[R, "inf_share"]], [y], R, s=12, z=6)
    a1.text(1.035, -0.35, "Gain", transform=gx, ha="left", va="bottom", fontsize=S.FS["tick"], color=S.INK2)
    a1.set_yticks(list(Y.values())); a1.set_yticklabels([lab for _, lab, _ in S.DATASETS], fontsize=S.FS["label"])
    a1.tick_params(axis="y", pad=6.5)
    for nm, *_ in S.DATASETS:   # the direction of the quota rule (only DHS takes the lowest)
        a1.text(-0.015, Y[nm], "↓" if D.quota_rule(nm) == "quota_bottom" else "↑", transform=a1.get_yaxis_transform(),
                ha="right", va="center", fontsize=S.FS["tick"], color=S.MUTED)
    for yy, lab in heads:
        S.text_mm(fig, 2.0, A1[1] + A1[3] * (1 - yy / ymax), lab, ha="left", va="center", fontsize=S.FS["note"],
                  color=S.INK2, fontstyle="italic")
    a2.set_yticks([])
    a1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); a1.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    a2.set_xticks([0, 0.5, 1]); a2.set_xticklabels(["0", "0.5", "1"])
    a1.set_xlabel("Coverage in the worst state, country or region"); a2.set_xlabel("Share of +∞ intervals")
    hs = [S.handle(J, LAB[J]), S.handle(JG, LAB[JG]), S.handle(R, LAB[R]),
          Line2D([], [], ls="none", marker="|", ms=5.0, mew=0.8, color=S.INK2, label="Sampling-noise limit"),
          Patch(facecolor=S.VIOLET_BAND, edgecolor="none", label="RegionCP minus JOMI")]   # a band, drawn as a fill
    fig.legend(handles=hs, ncol=3, loc="lower left", bbox_to_anchor=(2.0 / W, (A1[1] + A1[3] + 1.6) / H),   # rows: JOMI,
               handletextpad=0.25, columnspacing=1.0, borderaxespad=0, handlelength=1.4, labelspacing=0.3)    # RegionCP, band
    return [a1, a2]


# ------------------------------------------------------------------ c
def panel_c(fig, T):
    ax = S.axes_mm(fig, *CC)
    ax.set_xlim(-0.55, 2.85); ax.set_ylim(-0.43, 0.08)
    S.hgrid(ax, [-0.3, -0.2, -0.1]); S.target(ax, v=0.0, orient="h")
    off = {J: -0.18, R: 0.18}
    for mth in (J, R):
        x = T[T.key == mth]
        S.dot(ax, x.level + off[mth], x.gap, mth, s=5, z=3, alpha=0.75, emph=False)
        med = x.groupby("level").gap.median()
        S.dot(ax, med.index + off[mth], med.values, mth, s=26, z=6, edge=S.INK)
        for lv, v in med.items():
            if lv == 0:
                continue
            ax.text(lv + off[mth] + (0.13 if mth == R else -0.13), v, f"{v:+.2f}".replace("-", "−"),
                    ha="left" if mth == R else "right", va="center", fontsize=S.FS["tick"],
                    color=S.VIOLET if mth == R else S.BLUE, fontweight="bold" if mth == R else "normal")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["All\nselected", "State or\ncountry", "Spatial\ncell"])
    ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0.0]); ax.set_yticklabels(["−0.4", "−0.3", "−0.2", "−0.1", "0"])
    ax.set_ylabel("Worst group minus noise limit", labelpad=1.5)
    ax.text(2.8, 0.012, "limit", ha="right", va="bottom", fontsize=S.FS["note"], color=S.INK2)
    ax.text(-0.5, -0.425, "Large: median of 11 settings\nSmall: each setting", ha="left", va="bottom", fontsize=S.FS["note"],
            color=S.INK2, linespacing=1.15)
    return ax


# ------------------------------------------------------------------ b, d
def map_axes(fig, x):
    ax = S.axes_mm(fig, x, MAPY, MW, MH)
    ax.set_xlim(F8.EXT[0], F8.EXT[1]); ax.set_ylim(F8.EXT[2], F8.EXT[3]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


# floors in classes (user 2026-11-10: on a continuous 0-0.9 ramp the two maps looked the same, as nearly every floor lies
# between 0.7 and 0.9); one hue, light to dark, plus grey for no guarantee
FLOOR_BINS = [0.3, 0.6, 0.7, 0.8, 0.85, 0.9]
FLOOR_LABELS = ["< 0.6", "0.6–0.7", "0.7–0.8", "0.8–0.85", "≥ 0.85"]
def floor_colours():
    base = S.ramp("Purples", 0.22)
    cols = [base(x) for x in (0.12, 0.32, 0.52, 0.74, 0.97)]
    return ListedColormap(cols), BoundaryNorm(FLOOR_BINS, len(cols), clip=True)


def data_b():
    """Per FIA state under the state quota: the floor JOMI's overall guarantee implies, 1 - alpha / w for a state holding a
    share w of the selected plots (all of the miscoverage could fall there), and RegionCP's Theorem 1 floor
    (results/scg_slack_<FIA>.csv; w is its Ef, the expected selected share, equal to the pooled share of the 100 splits)."""
    sl = pd.read_csv(os.path.join(S.RES, f"scg_slack_{D.DS.FIA}.csv"))
    u = sl[sl.kind == "unit"].copy()
    return pd.DataFrame(dict(state=[F8.FIA_STATE[int(k)] for k in u.unit], code=u.unit.astype(int).values,
                             share_selected=u.Ef.values, jomi_implied_floor=1 - S.ALPHA / u.Ef.values,
                             regioncp_floor=u.floor.values))


def hatch_lines(ax, path, step=70.0, color=None, lw=0.35):
    """Diagonal lines over a state's polygon (km units), clipped to it: real strokes rather than a PDF hatch pattern."""
    bb = path.get_extents(); h = bb.y1 - bb.y0
    segs, c = [], bb.x0 - h
    while c < bb.x1:
        segs.append([(c, bb.y0), (c + h, bb.y1)]); c += step
    clip = PathPatch(path, facecolor="none", edgecolor="none", transform=ax.transData)
    ax.add_patch(clip)
    lc = LineCollection(segs, colors=color or S.MUTED, linewidths=lw, zorder=2)
    lc.set_clip_path(clip); ax.add_collection(lc)


class Hatched:
    """Legend key for 'no guarantee' (drawn by HandlerHatched)."""


class HandlerHatched(HandlerBase):
    def create_artists(self, legend, orig_handle, xd, yd, width, height, fontsize, trans):
        x0, y0, x1, y1 = -xd, -yd, -xd + width, -yd + height
        arts = [Rectangle((x0, y0), width, height, facecolor="white", edgecolor=S.MUTED, linewidth=0.3, transform=trans)]
        n = 4; c = x0 - height
        while c < x1:                                   # y - y0 = x - c, clipped to the key box by hand
            xa, xb = max(x0, c), min(x1, c + height)
            if xb > xa:
                arts.append(Line2D([xa, xb], [y0 + xa - c, y0 + xb - c], color=S.MUTED, lw=0.35, transform=trans))
            c += width / n
        return arts


def panel_b(fig, U):
    cmap, norm = floor_colours()
    axs = []
    for k, (col, lab) in enumerate((("jomi_implied_floor", LAB[J]), ("regioncp_floor", LAB[R]))):
        ax = map_axes(fig, MAPX[k]); axs.append(ax)
        fills = {r.state: cmap(norm(getattr(r, col))) for r in U.itertuples() if getattr(r, col) > 0}
        F8.basemap(ax, fills)
        for r in U.itertuples():              # no guarantee: hatched, so it never reads as panel d's grey (>= 0.90)
            if getattr(r, col) <= 0:
                hatch_lines(ax, F8.BM._path(F8.states()[r.state], F8.fia()[2]))
        v = U[col]
        info = (f"≤ {v.max():.2f}, none in {int((v <= 0).sum())} of {len(v)} states" if col == "jomi_implied_floor"
                else f"floors {v.min():.2f}–{v.max():.2f}")
        ax.text(0, 1.0, f"{lab}: {info}", transform=ax.transAxes, fontsize=S.FS["note"], ha="left", va="bottom", color=S.INK)
    handles = [Patch(facecolor=cmap(i), edgecolor="none", label=l) for i, l in enumerate(FLOOR_LABELS)]
    handles.append(Hatched())
    fig.legend(handles=handles, labels=FLOOR_LABELS + ["None"], handler_map={Hatched: HandlerHatched()}, ncol=6, loc="center left", bbox_to_anchor=(2.0 / W, 4.4 / H), handlelength=0.9,
               handleheight=0.8, borderaxespad=0, handletextpad=0.3, columnspacing=0.8,
               title="Coverage guaranteed in each state before any outcome is measured", title_fontsize=S.FS["note"],
               alignment="left")
    return axs


def data_d():
    g = D.groups(D.DS.FIA, "state"); g = g[(g.rule == "quota") & (g.n >= D.NMIN)]
    rows = []
    for mth in (J, R):
        for r in g[g.method == mth].itertuples():
            rows.append(dict(method=LAB[mth], key=mth, state=F8.FIA_STATE[int(r.group)], code=int(r.group),
                             coverage=r.c / r.n, shortfall=max(0.0, 1 - S.ALPHA - r.c / r.n), n_selected_pooled=int(r.n)))
    return pd.DataFrame(rows)


def panel_d(fig, T):
    cmap, norm = S.ramp("Reds", 0.18), Normalize(0, 0.65)
    axs = []
    for k, mth in enumerate((J, R)):
        ax = map_axes(fig, MAPX[2 + k]); axs.append(ax)
        x = T[T.key == mth].set_index("code")
        fills = {F8.FIA_STATE[c]: S.GREY_L if v <= 0 else cmap(norm(v)) for c, v in x.shortfall.items()}
        F8.basemap(ax, fills)
        worst = x.coverage.idxmin()
        ax.text(0, 1.0, f"{LAB[mth]}: worst state {x.coverage.min():.2f} ({F8.FIA_STATE[worst]})",
                transform=ax.transAxes, fontsize=S.FS["note"], ha="left", va="bottom", color=S.INK)
        if k == 0:   # the maps are 42 mm wide here (46 in make_fig8), so the inside-label margin is relaxed to 1.0 (box just fits)
            fit = F8.BM.label_features(ax, F8.states(), F8.fia()[2], {c: c for c in fills}, fills, fit=1.0,
                                       outside=dict(F8.OUTSIDE, CA=(-124.6, 33.4, True), MO=(-97.6, 38.4, True), VT=(-73.3, 47.4, True)))   # 6.2 pt: CA, MO outside
            print("state labels (margin): " + ", ".join(f"{c} {v:.2f}" for c, v in fit.items()))
    cax = S.axes_mm(fig, 107.0, 4.6, 40.0, 1.6)
    cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.2, 0.4, 0.6]); cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=1.5, width=0.4, pad=1)
    S.text_mm(fig, 149.0, 5.4, "Shortfall below 0.90", fontsize=S.FS["note"], color=S.INK, ha="left", va="center")
    fig.legend(handles=[Patch(facecolor=S.GREY_L, edgecolor="none", label="≥ 0.90"),
                        Patch(facecolor="white", edgecolor=F8.BM.COAST, linewidth=0.4, label="No data")],
               ncol=1, loc="center left", bbox_to_anchor=(92.0 / W, 5.4 / H), handlelength=0.8, handleheight=0.8,
               borderaxespad=0, handletextpad=0.3, labelspacing=0.15)
    return axs, cax


def main():
    S.apply()
    Ta = F8.data_a(); Ta = Ta[Ta.key.isin([J, JG, R])].reset_index(drop=True)
    Tc = F8.data_c(); Td = data_d()
    maps, _, nbar = F8.data_b(); Ub = data_b()
    fig = S.figure(W, H)
    A = panel_a(fig, Ta); C = panel_c(fig, Tc)
    B = panel_b(fig, Ub); Dm, cb_d = panel_d(fig, Td)
    for x, y, l in ((2.0, H - 1.0, "a"), (127.0, H - 1.0, "c"), (2.0, 37.5, "b"), (91.0, 37.5, "d")):
        S.letter_mm(fig, x, y, l)
    S.source_data("nc_regions_a", Ta); S.source_data("nc_regions_b", Ub); S.source_data("nc_regions_c", Tc)
    S.source_data("nc_regions_d", Td)
    S.save(fig, "nc_regions", axes=A + [C] + B + Dm, panel_ids=["a1", "a2", "c", "b1", "b2", "d1", "d2"],
           row_groups=[["a1", "a2"], ["b1", "b2"], ["d1", "d2"]], column_groups=[], exclude_axes=[cb_d])
    print(f"n_bar = {nbar:.1f}")
    print(Ta.pivot_table(index="dataset", columns="key", values="worst").round(3).to_string())
    print(Td.pivot_table(index="state", columns="key", values="coverage").round(3).to_string())


if __name__ == "__main__":
    main()
