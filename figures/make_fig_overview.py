"""Overview figure: the research problem and the solution, at a glance.

a  maps rank places and decisions follow; b  the failure there, on real data: the FIA benchmark biomass map, its highest 5%
of test plots in one split and whether split CP's 90% interval covers each, with the coverage pooled over 100 splits;
c  what a decision needs to know; d  the approach (2026-11-10: titled 'Our approach', not the package name): audit and rule, reference set by swapping,
   calibration on it, and what comes out, with the mapselect function behind each output.
a and d are schematics on a seeded synthetic map, not data; their selection, reference set and thresholds are computed with
mapselect, so the parts show what the method does on that map. b (since 2026-11-09; before, simulated errors) reads
results/units_<FIA>_rep0.npz and results/fig1_fia_split_cp.csv (experiments/fia_fig1_numbers.py); state outlines from
Natural Earth 1:110m (figures/basemap.py; downloaded files, hence python -I).
    FIGURE_QA_HELPERS=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python -I figures/make_fig_overview.py
writes figs/nc_overview.{pdf,svg,png}, figs/source_data/nc_overview_b_{map,coverage}.csv
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "package", "src")]; sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import mapselect as ms                                                # noqa: E402
import fig_data as D                                                  # noqa: E402
import basemap as BM                                                  # noqa: E402

W, H = 183, 127
PINK = "#fbe6e4"; TINT = "#e8f0fb"
GREEN = "#5f8f6b"; BROWN = "#8a6d4f"
RELIEF = matplotlib.colors.LinearSegmentedColormap.from_list("relief", ["#f7f6f1", "#c9c4b4"])


def field(xy, w, h):
    x, y = xy[:, 0] / w, xy[:, 1] / h
    return np.exp(-((x - .68) ** 2 + (y - .58) ** 2) / (2 * .17 ** 2)) + .55 * np.exp(-((x - .22) ** 2 + (y - .30) ** 2) / (2 * .12 ** 2))


def world(seed=11, n=300, K=12):
    """One synthetic population on the unit map; errors grow with the value, so the top of the map is the hard part."""
    r = np.random.default_rng(seed)
    xy = r.random((n, 2))
    mu = field(xy, 1, 1) + 1e-6 * r.random(n)
    y = mu + (0.05 + 0.32 * mu ** 2) * r.standard_normal(n)
    p = r.permutation(n); C, T = p[:n // 2], p[n // 2:]
    sel = ms.top_k(mu[C], mu[T], K)
    return dict(xy=xy, mu=mu, S=np.abs(y - mu), C=C, T=T, J=T[sel.selected], R=C[sel.sets[0]], sel=sel)


def place(xy, x0, y0, w, h, pad=0.7):
    return np.c_[x0 + pad + xy[:, 0] * (w - 2 * pad), y0 + pad + xy[:, 1] * (h - 2 * pad)]


def tmap(ax, x0, y0, w, h):
    gx, gy = np.meshgrid(np.linspace(0, w, 161), np.linspace(0, h, 97))
    Z = field(np.c_[gx.ravel(), gy.ravel()], w, h).reshape(gx.shape)
    ax.contourf(gx + x0, gy + y0, Z, levels=np.linspace(Z.min(), Z.max(), 8), cmap=RELIEF, zorder=1)
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=S.GRID, lw=.4, zorder=6))


def text(ax, x, y, s, **kw):
    return ax.text(x, y, s, **dict(dict(fontsize=S.FS["note"], color=S.INK, ha="left", va="baseline"), **kw))


def title(ax, x, y, letter, s):
    ax.text(x, y, letter, fontsize=S.FS["panel"], fontweight="bold", va="baseline")
    ax.text(x + 4.2, y, s, fontsize=S.FS["title"], fontweight="bold", va="baseline")


# ---- icons, all in millimetres around (x, y)
def tree(ax, x, y, s=1.0, c=GREEN):
    ax.add_patch(Rectangle((x - .28 * s, y - 2.2 * s), .56 * s, 1.0 * s, fc=BROWN, ec="none", zorder=5))
    for dy, hw in ((-1.4, 1.35), (-.5, 1.1), (.35, .8)):
        ax.add_patch(Polygon([(x - hw * s, y + dy * s), (x + hw * s, y + dy * s), (x, y + (dy + 1.45) * s)], fc=c, ec="none", zorder=5))


def house(ax, x, y, s=1.0, c=S.INK2):
    ax.add_patch(Rectangle((x - 1.15 * s, y - 1.5 * s), 2.3 * s, 1.75 * s, fc=c, ec="none", zorder=5))
    ax.add_patch(Polygon([(x - 1.65 * s, y + .2 * s), (x + 1.65 * s, y + .2 * s), (x, y + 1.6 * s)], fc=c, ec="none", zorder=5))
    ax.add_patch(Rectangle((x - .3 * s, y - 1.5 * s), .6 * s, .95 * s, fc="white", ec="none", zorder=6))


def drop(ax, x, y, s=1.0, c=S.BLUE):
    ax.add_patch(Circle((x, y - .5 * s), 1.1 * s, fc=c, ec="none", zorder=5))
    ax.add_patch(Polygon([(x - 1.0 * s, y - .1 * s), (x + 1.0 * s, y - .1 * s), (x, y + 1.7 * s)], fc=c, ec="none", zorder=5))


def satellite(ax, x, y):
    ax.add_patch(Rectangle((x - 1.1, y - 1.1), 2.2, 2.2, fc=S.INK2, ec="none", zorder=5))
    for sx in (-1, 1):
        a = x + sx * 1.1; b = x + sx * 5.2
        ax.add_patch(Rectangle((min(a, b), y - .85), 4.1, 1.7, fc=S.BLUE, ec="none", alpha=.75, zorder=5))
        ax.plot([a, b], [y, y], color="white", lw=.35, zorder=6)


def bubble(ax, x, y, w, h, head, body):
    S.card(ax, x, y, w, h, fc="white", ec=S.GREY_L, lw=.6, r=1.3)
    ax.add_patch(Polygon([(x + w - .1, y + h / 2 + 1.0), (x + w - .1, y + h / 2 - 1.0), (x + w + 1.7, y + h / 2)], fc="white", ec=S.GREY_L, lw=.6, zorder=1))
    ax.plot([x + w - .1, x + w - .1], [y + h / 2 - .92, y + h / 2 + .92], color="white", lw=1.0, zorder=2)
    text(ax, x + 1.5, y + h - 3.8, head, fontweight="bold")
    text(ax, x + 1.5, y + h - 6.9, body)


# ---- panels
def panel_a(ax, w):
    title(ax, 1.5, 122.5, "a", "Machine-learning maps rank places, and decisions follow")
    xs = (3.5, 31.5, 60.5, 90.0); head = ("Data", "Map", "Rule selects", "Decision")
    for x, s in zip(xs, head):
        text(ax, x, 115.8, s, fontweight="bold", fontsize=S.FS["label"])
    for a, b in ((25.8, 29.8), (56.4, 58.9), (85.6, 88.4)):
        S.arrow(ax, (a, 98), (b, 98), lw=.8)
    # data
    satellite(ax, 14.2, 108.0)
    ax.add_patch(Polygon([(12.6, 106.6), (15.8, 106.6), (22.5, 91.2), (5.8, 91.2)], fc=S.BLUE, ec="none", alpha=.08, zorder=1))
    for k, (x, s) in enumerate(((7.5, 1.1), (11.2, 1.3), (15.0, 1.0), (18.6, 1.25), (22.0, 1.05))):
        tree(ax, x, 90.2 + (k % 2) * .9, s)
    ax.plot([3.5, 25.2], [87.6, 87.6], color=S.GREY_L, lw=.6)
    text(ax, 3.5, 83.9, "Imagery and field plots")
    # map: prediction and its uncertainty layer
    tmap(ax, 31.5, 96.2, 23.5, 17.2); text(ax, 31.5, 92.9, "Prediction")
    S.card(ax, 31.5, 84.6, 23.5, 6.6, fc=S.SHADE, ec=S.GRID, lw=.4, r=.3)
    ax.text(43.2, 87.9, "± an error layer", ha="center", va="center", fontsize=S.FS["note"], color=S.INK2)
    text(ax, 31.5, 81.4, "Valid on average")
    # rule
    z = world(); w0, h0 = 24.0, 28.0; x0, y0 = 60.5, 85.4
    tmap(ax, x0, y0, w0, h0)
    S.units(ax, place(z["xy"][z["T"]], x0, y0, w0, h0), "test", s=1.8)
    S.units(ax, place(z["xy"][z["J"]], x0, y0, w0, h0), "selected", s=6.5, z=5)
    text(ax, x0, 81.4, "Top K, quota or budget")
    # decisions
    for fn, y, s in ((tree, 108.2, "Carbon credits"), (house, 98.0, "Aid"), (drop, 88.4, "Flood relief")):
        fn(ax, 92.8, y, 1.25 if fn is tree else 1.15); text(ax, 96.6, y - 1.0, s)
    # question
    S.card(ax, 3.5, 72.6, 109.6, 6.0, fc=PINK, ec="none", lw=0, r=1.0)
    ax.text(58.3, 75.5, "Is the map's uncertainty still right at the places it selected?", ha="center", va="center", fontsize=S.FS["label"], fontweight="bold", color=S.MISS)


US_EXT = (-10900, -5650, 2700, 5500)        # km; the FIA map window of Fig. 4 (make_fig8.py)
WEST_EXT = (-10790, -9690, 3520, 5480)      # km; Washington to California, where the map's highest plots lie
BIG = 10                                    # pt; the two coverage numbers of b


def fia_split():
    """b's data: split 0 of the FIA benchmark setting (test plots, the map's highest 5% and whether split CP's 90% interval
    covers each; results/units_<FIA>_rep0.npz) and split CP coverage pooled over the 100 splits (results/fig1_fia_split_cp.csv)."""
    z = np.load(os.path.join(S.RES, f"units_{D.DS.FIA}_rep0.npz"), allow_pickle=False)
    d = D.DS.load(D.DS.FIA)
    assert np.array_equal(d["coords"], z["coords"])
    J, cov = z["top5|idx"], z["top5|SCP |res||cov"].astype(bool)
    R, C = np.abs(z["y"] - z["mu"]), z["C"]
    q = np.sort(R[C])[int(np.ceil(.9 * (len(C) + 1))) - 1]
    assert np.array_equal(cov, R[J] <= q)                              # the stored coverage is split CP on |y - mu|
    p = pd.read_csv(os.path.join(S.RES, "fig1_fia_split_cp.csv")); s = p.sum()
    per = {"all": p.n_cov_all / p.m, "top": p.n_cov_top5 / p.K}                # coverage in each split
    spread = {k: (1.96 * v.std(ddof=1) / np.sqrt(len(v)), v.min(), v.max()) for k, v in per.items()}
    return dict(xy=z["coords"], T=z["T"], J=J, cov=cov, mu=z["mu"], lonlat=d["Xrf"][:, :2], state=z["state"],
                proj=BM.km_projection(d["coords"], d["Xrf"][:, 0], d["Xrf"][:, 1]), n_splits=len(p),
                all=(int(s.n_cov_all), int(s.m)), top=(int(s.n_cov_top5), int(s.K)), spread=spread)


def map_axes(fig, x, y, w, ext):
    h = w * (ext[3] - ext[2]) / (ext[1] - ext[0])                     # equal scale in x and y
    a = S.axes_mm(fig, x, y, w, h)
    a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    return a, h


def fia_map(a, f, ext, s_test, s_sel, ring):
    feats = BM.conus()
    BM.fill(a, feats, f["proj"], {st: S.SHADE for st in D.fia_states().values()}, box=ext)
    BM.outline(a, feats, f["proj"], coast=BM.COAST, box=ext)
    T = f["T"]
    a.scatter(f["xy"][T, 0], f["xy"][T, 1], s=s_test, color=S.GREY_L, lw=0, zorder=4)
    o = np.random.default_rng(0).permutation(len(f["J"]))             # neither colour sits on top as a block
    J, cov = f["J"][o], f["cov"][o]
    a.scatter(f["xy"][J, 0], f["xy"][J, 1], s=s_sel, c=[S.INK if k else S.MISS for k in cov], edgecolors="white",
              linewidths=ring, zorder=5)


def panel_b(fig, ax):
    title(ax, 117.5, 122.5, "b", "Often not: the winner's curse of maps")
    f = fia_split()
    text(ax, 119.0, 118.0, "Forest biomass map, US FIA plots (one split)", color=S.INK2)
    loc, hl = map_axes(fig, 119.0, 96.8, 36.0, US_EXT)
    fia_map(loc, f, US_EXT, 0.05, 0.8, 0.0)
    loc.add_patch(Rectangle((WEST_EXT[0], WEST_EXT[2]), WEST_EXT[1] - WEST_EXT[0], WEST_EXT[3] - WEST_EXT[2],
                            fc="none", ec=S.INK2, lw=0.5, zorder=6))
    zw = 23.4; zh = zw * (WEST_EXT[3] - WEST_EXT[2]) / (WEST_EXT[1] - WEST_EXT[0])
    zoom, _ = map_axes(fig, 182.4 - zw, 96.8 + hl - zh, zw, WEST_EXT)
    for sp in zoom.spines.values():                                    # framed like its box on the locator
        sp.set_visible(True); sp.set_color(S.INK2); sp.set_linewidth(0.5)
    fia_map(zoom, f, WEST_EXT, 0.3, 3.2, 0.25)
    for st, lon, lat in (("WA", -118.9, 47.3), ("OR", -120.6, 44.0), ("CA", -117.0, 34.6)):
        p = f["proj"](np.array([[lon, lat]], float))[0]
        zoom.text(p[0], p[1], st, fontsize=S.FS["note"], color=S.INK2, ha="center", va="center", zorder=7)
    mm = lambda dx, dy: (WEST_EXT[0] + dx / zw * (WEST_EXT[1] - WEST_EXT[0]), WEST_EXT[2] + dy / zh * (WEST_EXT[3] - WEST_EXT[2]))   # noqa: E731
    # In the Pacific off southern California. The collision audit flags these two labels against California's bounding
    # box (text-fill-edge WARN); checked at final size on 2026-11-09: they sit west of the coast and touch no outline.
    for dy, col, lab in ((4.0, S.MISS, "Outside"), (1.6, S.INK, "Inside")):
        zoom.scatter(*mm(0.9, dy), s=4.0, color=col, edgecolors="white", linewidths=.25, zorder=7)
        zoom.text(*mm(1.9, dy), lab, fontsize=S.FS["note"], color=S.INK, ha="left", va="center", zorder=7)
    (ca, ma), (ct, kt) = f["all"], f["top"]
    text(ax, 119.0, 91.2, "Inside the 90% interval", color=S.INK2)
    text(ax, 119.0, 88.6, f"(split CP, {f['n_splits']} splits)", color=S.INK2)
    ax.text(119.0, 82.4, f"{ca / ma:.2f}", fontsize=BIG, fontweight="bold", color=S.INK, va="baseline")
    text(ax, 128.6, 82.6, "all test plots")
    ax.text(119.0, 76.0, f"{ct / kt:.2f}", fontsize=BIG, fontweight="bold", color=S.MISS, va="baseline")
    text(ax, 128.6, 76.2, "the map's highest 5%")
    # source data: the split drawn, and the pooled numbers
    T = f["T"]; rank = np.full(len(f["mu"]), np.nan); rank[T] = (-f["mu"][T]).argsort().argsort() + 1
    sel = np.isin(T, f["J"]); inside = np.full(len(T), np.nan); inside[sel] = [float(f["cov"][list(f["J"]).index(j)]) for j in T[sel]]
    st = D.fia_states()
    S.source_data("nc_overview_b_map", pd.DataFrame(dict(plot=T, lon=f["lonlat"][T, 0], lat=f["lonlat"][T, 1],
                  state=[st[int(g)] for g in f["state"][T]], map_rank=rank[T].astype(int), selected=sel.astype(int),
                  inside_split_cp_interval=inside)))
    sp = f["spread"]
    S.source_data("nc_overview_b_coverage", pd.DataFrame(dict(
        plots=["all test plots", "the map's highest 5% of test plots"], splits=[f["n_splits"]] * 2,
        plot_splits=[ma, kt], inside_split_cp_interval=[ca, ct], share_inside=[ca / ma, ct / kt],
        ci95_halfwidth_over_splits=[sp["all"][0], sp["top"][0]], lowest_split=[sp["all"][1], sp["top"][1]],
        highest_split=[sp["all"][2], sp["top"][2]])))
    return loc, zoom


def panel_c(ax):
    title(ax, 1.5, 60.8, "c", "What a decision must know")
    rows = (("Interval", "How wide at each selected plot?"), ("Project total", "How much carbon can I credit?"),
            ("Funding", "Which villages can I safely fund?"), ("Region", "Does it hold in my country?"), ("Audit size", "How many plots must I visit?"))
    y = 47.9
    for a, b in rows:
        bubble(ax, 1.5, y, 43.0, 10.1, a, b); y -= 11.3


def panel_d(ax):
    x0 = 50.5
    S.card(ax, x0, 2.4, W - x0 - .4, 63.0, fc="white", ec=S.BLUE, lw=.8, r=1.5)
    title(ax, x0 + 2.0, 60.8, "d", "Our approach: calibrate each place only on audited places the same rule would select")
    cw, ch, y0 = (W - x0 - .4 - 4.0 - 8.0) / 3, 33.4, 24.5; xs = [x0 + 2.0, x0 + 2.0 + cw + 4.0, x0 + 2.0 + 2 * (cw + 4.0)]
    z = world()
    names = ("Audit at random, fix the rule", "Swap: would it be selected too?", "Calibrate on that reference set")
    for k, (x, s) in enumerate(zip(xs, names)):
        S.card(ax, x, y0, cw, ch, fc="white", ec=S.GRID, lw=.6, r=1.0)
        S.badge(ax, x + 2.1, y0 + ch - 2.2, str(k + 1)); text(ax, x + 4.0, y0 + ch - 3.0, s, fontweight="bold")
        if k < 2:
            S.arrow(ax, (x + cw + .5, y0 + ch / 2), (x + cw + 3.5, y0 + ch / 2), lw=.8)
    mw, mh, my = cw - 4.0, 21.3, y0 + 7.0
    # 1: audited and candidates
    x = xs[0] + 2.0; tmap(ax, x, my, mw, mh)
    S.units(ax, place(z["xy"][z["T"]], x, my, mw, mh), "test", s=2.6)
    S.units(ax, place(z["xy"][z["C"]], x, my, mw, mh), "calibration", s=3.2)
    S.units(ax, place(z["xy"][z["J"]], x, my, mw, mh), "selected", s=7.5, z=5)
    S.units(ax, [[x + 1.0, y0 + 3.5]], "calibration", s=3.2); text(ax, x + 2.0, y0 + 2.7, "Audited")
    S.units(ax, [[x + 13.0, y0 + 3.5]], "test", s=2.6); text(ax, x + 14.0, y0 + 2.7, "Candidate")
    S.units(ax, [[x + 26.6, y0 + 3.5]], "selected", s=7.5); text(ax, x + 27.6, y0 + 2.7, "Selected")
    # 2: reference set and one swap
    x = xs[1] + 2.0; tmap(ax, x, my, mw, mh)
    out = np.setdiff1d(z["C"], z["R"])
    S.units(ax, place(z["xy"][out], x, my, mw, mh), "calibration", s=2.6)
    S.units(ax, place(z["xy"][z["J"]], x, my, mw, mh), "selected", s=7.5, z=5)
    S.units(ax, place(z["xy"][z["R"]], x, my, mw, mh), "reference", s=8.5, z=5)
    pr = place(z["xy"][z["R"]], x, my, mw, mh); pj = place(z["xy"][z["J"]], x, my, mw, mh)
    every = place(z["xy"][np.r_[z["C"], z["J"]]], x, my, mw, mh); best = (-1, None)
    for a1, p1 in enumerate(pr):                                        # the pair whose labels sit clearest of the other units
        for b1, p2 in enumerate(pj):
            g = np.linalg.norm(p1 - p2)
            if not 8 <= g <= 15:
                continue
            u = (p1 - p2) / g; l1, l2 = p1 + 2.6 * u, p2 - 2.6 * u
            clear = min(np.linalg.norm(every - l1, axis=1).min(), np.linalg.norm(every - l2, axis=1).min())
            if clear > best[0]:
                best = (clear, (a1, b1, l1, l2))
    a_, b_, l1, l2 = best[1]
    for q_, lab, at in ((pr[a_], "i", l1), (pj[b_], "j", l2)):
        ax.add_patch(Circle(q_, 1.05, fc="none", ec=S.INK, lw=.6, zorder=7))
        ax.text(at[0], at[1], lab, fontsize=S.FS["label"], fontstyle="italic", fontweight="bold", ha="center", va="center", zorder=8)
    ax.add_patch(FancyArrowPatch(tuple(pr[a_]), tuple(pj[b_]), arrowstyle="<|-|>,head_length=2.6,head_width=1.0", mutation_scale=1, color=S.INK, lw=.7, shrinkA=3.2, shrinkB=3.2, connectionstyle="arc3,rad=-.35", zorder=7))
    S.units(ax, [[x + 1.0, y0 + 3.5]], "reference", s=8.5); text(ax, x + 2.2, y0 + 2.7, "Reference set: selected if swapped")
    # 3: scores, all audited against the reference set
    x = xs[2] + 2.0; xa, xb = x + 1.0, x + mw - 1.0
    Sc, Sr = z["S"][z["C"]], z["S"][z["R"]]; top = max(Sc.max(), Sr.max()) * 1.04
    sx = lambda v: xa + np.asarray(v) / top * (xb - xa)
    r = np.random.default_rng(2)
    yall, yref = my + 15.0, my + 4.8
    ax.scatter(sx(Sc), yall + r.uniform(-1.3, 1.3, len(Sc)), s=2.4, color=S.GREY_M, lw=0, zorder=4)
    ax.scatter(sx(Sr), yref + r.uniform(-1.3, 1.3, len(Sr)), s=7, color=S.BLUE, lw=0, zorder=4)
    q0 = np.sort(Sc)[int(np.ceil(.9 * (len(Sc) + 1))) - 1]; q1 = float(ms.jomi(z["S"][z["C"]], z["sel"], .1)[0])
    assert np.isfinite(q1) and q1 > q0
    ax.plot([sx(q0)] * 2, [yall - 2.4, yall + 2.4], color=S.INK, lw=1.0, zorder=5)
    ax.plot([sx(q1)] * 2, [yref - 2.4, yref + 2.4], color=S.INK, lw=1.0, zorder=5)
    text(ax, xa, yall + 3.1, "All audited places: too small a margin")
    text(ax, xa, yref + 3.1, "Reference set: the margin needed", color=S.BLUE)
    ax.plot([xa, xb], [my, my], color=S.INK2, lw=.5)
    text(ax, xa, my - 2.6, "Error size; tick marks the 90% cut-off", color=S.INK2)
    # outputs
    S.arrow(ax, (x0 + 2.0 + 1.5 * cw + 4.0 + cw / 2 - cw / 2, y0 - .2), (x0 + 2.0 + 1.5 * cw + 4.0, y0 - 3.1), lw=.8)
    rows = (("Interval", "coverage ≥ 1 − α", "jomi"), ("Project total", "Pr(total ≥ L) ≥ 1 − δ", "exact_lower_total"), ("Funding", "false discovery rate ≤ q", "cfbh"),
            ("Region", "a floor from its data", "regioncp"), ("Audit size", "known before fieldwork", "plan"))
    gw = (W - x0 - .4 - 4.0 - 4 * 1.2) / 5; y1 = 9.6; hh = 11.4
    for k, (a, b, c) in enumerate(rows):
        gx = x0 + 2.0 + k * (gw + 1.2)
        S.card(ax, gx, y1, gw, hh, fc=TINT, ec="none", lw=0, r=.9)
        text(ax, gx + 1.1, y1 + hh - 3.0, a, fontweight="bold")
        text(ax, gx + 1.1, y1 + hh - 5.8, b)
        text(ax, gx + 1.1, y1 + hh - 8.9, c, family="monospace", color=S.BLUE)
    ax.text(x0 + 2.0, 5.8, "Needs a random audit, a separately fitted map and a rule that uses covariates alone. Valid under any spatial autocorrelation,", fontsize=S.FS["note"], color=S.INK2, va="baseline")
    ax.text(x0 + 2.0, 3.4, "on average over audits; small audits give infinite intervals and weak floors.", fontsize=S.FS["note"], color=S.INK2, va="baseline")


def main():
    S.apply()
    fig = S.figure(W, H); ax = S.canvas(fig, 0, 0, W, H)
    panel_a(ax, W); loc, zoom = panel_b(fig, ax); panel_c(ax); panel_d(ax)
    S.arrow(ax, (46.4, 33.2), (50.2, 33.2), lw=1.0, color=S.BLUE)
    # Recorded exclusion (2026-11-09): b's zoom is the locator's enlargement, not a comparable panel (two map scales by
    # design; tops aligned at the same y by construction). The auditor reports NOT AUDITABLE for a pair whose only group is
    # exempted, so the zoom is excluded and the locator is audited as the figure's one data panel, as the single panel b
    # was before.
    S.save(fig, "nc_overview", axes=[loc], panel_ids=["b"], exclude_axes=[zoom])


if __name__ == "__main__":
    main()
