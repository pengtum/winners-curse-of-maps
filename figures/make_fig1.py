"""Fig. 2 (Fig. 1 before 2026-11-05) | The winner's curse of maps: the concept (paper/FIGURE_PLAN.md).

a  the smooth map, three decisions: carbon credits for the highest K units, aid for the lowest K units in every
   country (a quota), and flood relief within a budget that pays travel costs from a depot; open triangles, the units
   that only the rough map picks for carbon
b  valid on average only: split CP calibrated on all units covers 0.90 of all test units, but it is too narrow at the
   carbon picks, missed mostly above the interval on the smooth map (which flattens the peak) and mostly below it on a
   rough map of the same world (which copies nearby training units and so invents peaks where samples are few; ranking
   picks them, the classical winner's curse), and it covers every aid pick with intervals far wider than needed
c  why the selected places differ, on the profile A-B of a (bent at the peak, down its south-east flank): errors grow
   with the value (spatial heteroscedasticity), maps flatten peaks or invent them (Gelman & Price 1999), and the
   extremes sit where samples are few
The spread-scaled check (nc_fig1b.csv, coverage_true_spread) repeats b with the score divided by the true error spread:
what selection does beyond the error spread.
d  this paper's answer, a preview of Fig. 1d and Supplementary Fig. 25: calibrate on the reference set, borrow strength across scales only as far
   as the data allow, and bound the decision itself (totals, funding)

Everything is computed on one synthetic world (seeded; a schematic, not data): a fixed population of map units with a
lowland at a floor, two peaks and a sparsely sampled north-east; a kernel smoother as the model (the smooth map, whose
bandwidth widens where samples are few, drives a, c and d; b adds the carbon rule on a rough map, the same smoother with a
fixed narrow bandwidth); split CP with the absolute-residual score; the selections by excp_select.refsets (the paper's
closed forms). Coverage in b pools
300 random splits; the maps, profiles and dots show the first split (b: the first 30).
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_fig1.py
writes figs/nc_fig1_concept.{pdf,svg,png}, figs/source_data/nc_fig1{a,b,c}.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
from excp_select import basis, refsets                                # noqa: E402

# ---- layout (mm)
W, GAP = 183.0, 3.0
CW, H1 = (W - 3 * GAP) / 4, 63.0                # four cards a-d (60 until 2026-11-10: text now >= 6.2 pt)
H, Y1 = H1 + 1.0, 0.5
TX = 1.4                                        # text inset in a card
MW, MH = 40.0, 28.0                             # the map; the synthetic world's coordinates are these millimetres
MX = (CW - MW) / 2
STRIPS = []                                     # (axes, inner box, texts) of every strip, for the overflow guard

# ---- the synthetic world: a lowland at a floor, uplands with two peaks, and a north-east sampled sparsely
SEED, R, SHOW, K, QUOTA, BUDGET = 11, 300, 30, 6, 2, 4.0
ROUGH_H = 0.6                                   # bandwidth (mm) of the rough map
FLOOR = 0.12
P1, P2, DEPOT = (31.0, 21.0), (10.0, 22.5), (4.5, 16.5)
BORDERS = (8.5, 16.5, 24.5, 32.5)              # five countries as north-south strips: mean x of the four borders
A, B = np.array([4.0, 3.0]), np.array([39.0, 12.5])                          # the profile A-B of panel c,
PATH = np.array([A, [30.0, 21.0], B])                                         # bent over the peak, down its south-east flank
NEAR = 1.4                                      # units within this distance (mm) of A-B appear on the profile
MARK = {"carbon": ("^", 11.0), "aid": ("v", 11.0), "flood": ("s", 7.0)}
BORDER_LS, COST_LS = (0, (3.2, 1.2, 0.8, 1.2)), (0, (2.2, 1.6))
GROUPS = [("all", "All test units"), ("carbon", "Carbon, top K, smooth map"), ("carbon_rough", "Carbon, top K, rough map"),
          ("aid", "Aid, lowest K per country"), ("flood", "Flood relief, budget")]


def bump(s, c, sd):
    return np.exp(-((s[:, 0] - c[0]) ** 2 + (s[:, 1] - c[1]) ** 2) / (2 * sd ** 2))


def sig(z):
    return 1 / (1 + np.exp(-z))


def truth(s):
    s = np.atleast_2d(s)
    return FLOOR + 0.30 * sig((s[:, 1] - 12.5) / 2.2) + 0.42 * bump(s, P1, 5.2) + 0.22 * bump(s, P2, 3.8)


def noise_sd(f):
    """Error size grows with the value: 0.02 on the lowland floor, about 0.07 on the highest peak."""
    return 0.02 + 0.09 * np.maximum(f - FLOOR, 0) ** 1.5


def thinning(s):
    """Share of the lattice left unsampled: 0 almost everywhere, 0.72 in the north-east (the data desert)."""
    return 0.72 * sig((s[:, 0] - 25) / 1.4) * sig((s[:, 1] - 13.5) / 1.4)


def border_x(k, yy):
    """The k-th country border: a gently winding north-south line."""
    return BORDERS[k] + 1.1 * np.sin(np.asarray(yy) / 3.2 + 1.7 * k)


def country_of(s):
    return sum((s[:, 0] > border_x(k, s[:, 1])).astype(int) for k in range(len(BORDERS)))


def country_cells(n=57):
    """The five countries as polygons (for the layer of country indicators in d)."""
    yy = np.linspace(0, MH, n)
    edges = ([np.c_[np.zeros(n), yy]] + [np.c_[border_x(k, yy), yy] for k in range(len(BORDERS))]
             + [np.c_[np.full(n, MW), yy]])
    return [np.r_[edges[k], edges[k + 1][::-1]] for k in range(len(edges) - 1)]


def population():
    rng = np.random.default_rng(SEED)
    gx, gy = np.meshgrid(np.arange(0.9, MW - 0.4, 1.45), np.arange(0.9, MH - 0.4, 1.45))
    s = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-0.42, 0.42, (gx.size, 2))
    s = s[rng.uniform(size=len(s)) > thinning(s)]
    f = truth(s)
    y = np.maximum(f + noise_sd(f) * rng.standard_normal(len(s)), 0.0)
    return s, f, y, country_of(s)


def smoother(s_tr, y_tr, s, k=8, hf=0.55):
    """The model: a Gaussian kernel smoother whose bandwidth widens where training units are sparse (0.55 times the
    distance to the 8th nearest)."""
    d, _ = cKDTree(s_tr).query(s, k=k)
    h = np.maximum(hf * d[:, -1], 0.8)
    w = np.exp(-0.5 * ((s[:, None, :] - s_tr[None]) ** 2).sum(-1) / h[:, None] ** 2)
    return (w * y_tr).sum(1) / w.sum(1)


def rough(s_tr, y_tr, s):
    """The rough map: the same smoother with a fixed narrow bandwidth, so it copies the noise of the nearest training
    units where samples are few."""
    w = np.exp(-0.5 * ((s[:, None, :] - s_tr[None]) ** 2).sum(-1) / ROUGH_H ** 2)
    return (w * y_tr).sum(1) / w.sum(1)


def picks(s, mu, country, cal, te, cost, budget):
    """Test units the three rules select (population indices), by the paper's closed-form selections."""
    quota = {u: QUOTA for u in range(len(BORDERS) + 1)}
    v = mu - mu.min() + 0.02                                      # predicted damage, made positive
    return {"carbon": te[refsets.topk(mu[cal], mu[te], K).selected],
            "aid": te[refsets.quota(mu[cal], mu[te], country[cal], country[te], quota, largest=False).selected],
            "flood": te[refsets.knapsack_greedy(v[cal], cost[cal], v[te], cost[te], budget).selected]}


def simulate(s, y, country, sd=None):
    """R random splits (training fold 40 / calibration 30 / test 30): refit, calibrate split CP, select. Returns the
    per-split counts, every unit's residual over its split-CP half-width by group, and the first split."""
    rng = np.random.default_rng(0)
    cost = 1 + np.hypot(*(s - DEPOT).T) / 6                       # travel cost from the depot
    budget = BUDGET * np.median(cost)
    rows, z, first = [], {g: [] for g, _ in GROUPS}, None
    sd = noise_sd(truth(s)) if sd is None else sd                 # the true error spread, for the scaled check
    for r in range(R):
        p = rng.permutation(len(s)); n_tr, n_c = int(0.4 * len(s)), int(0.3 * len(s))
        tr, cal, te = p[:n_tr], p[n_tr:n_tr + n_c], p[n_tr + n_c:]
        mu, mu_r = smoother(s[tr], y[tr], s), rough(s[tr], y[tr], s)
        res, res_r = y - mu, y - mu_r
        k = int(np.ceil((1 - S.ALPHA) * (n_c + 1)))
        q, q_r = np.sort(np.abs(res[cal]))[k - 1], np.sort(np.abs(res_r[cal]))[k - 1]
        sel = dict(all=te, **picks(s, mu, country, cal, te, cost, budget))
        sel["carbon_rough"] = te[refsets.topk(mu_r[cal], mu_r[te], K).selected]
        qs = np.sort(np.abs(res[cal]) / sd[cal])[k - 1]; qs_r = np.sort(np.abs(res_r[cal]) / sd[cal])[k - 1]
        for g, idx in sel.items():
            rr, qq, qz = (res_r, q_r, qs_r) if g == "carbon_rough" else (res, q, qs)
            rows.append(dict(split=r, group=g, selected=len(idx), covered=int((np.abs(rr[idx]) <= qq).sum()),
                             covered_scaled=int((np.abs(rr[idx]) / sd[idx] <= qz).sum())))
            z[g].append(rr[idx] / qq)
        if r == 0:
            first = dict(tr=tr, cal=cal, te=te, mu=mu, mu_r=mu_r, q=q, q_r=q_r, sel=sel)
    return pd.DataFrame(rows), z, first


def summary(counts, z):
    out = []
    for g, lab in GROUPS:
        c = counts[counts.group == g]; zz = np.abs(np.concatenate(z[g]))
        zs = np.concatenate(z[g])
        out.append(dict(group=g, label=lab, splits=len(c), selected_per_split=c.selected.mean(),
                        coverage=c.covered.sum() / c.selected.sum(), needed_halfwidth=np.quantile(zz, 1 - S.ALPHA),
                        coverage_true_spread=c.covered_scaled.sum() / c.selected.sum(),
                        missed_above=np.mean(zs > 1), missed_below=np.mean(zs < -1), mean_error_over_q=zs.mean()))
    return pd.DataFrame(out).set_index("group")


def profile(s, y, first, n=241):
    """Truth, the 90% band of outcomes and the first split's map along A-B, and the units near the line."""
    seg = np.diff(PATH, axis=0); L = np.r_[0, np.cumsum(np.hypot(*seg.T))]
    t = np.linspace(0, 1, n)
    line = np.c_[np.interp(t * L[-1], L, PATH[:, 0]), np.interp(t * L[-1], L, PATH[:, 1])]
    f = truth(line); sd = noise_sd(f)
    mu = smoother(s[first["tr"]], y[first["tr"]], line)
    mu_r = rough(s[first["tr"]], y[first["tr"]], line)
    d, ts = np.full(len(s), np.inf), np.zeros(len(s))             # distance to the path and position along it
    for k, (p0, v) in enumerate(zip(PATH[:-1], seg)):
        u = np.clip((s - p0) @ v / (v @ v), 0, 1)
        dk = np.hypot(*(s - p0 - u[:, None] * v).T)
        better = dk < d
        d[better], ts[better] = dk[better], (L[k] + u[better] * np.hypot(*v)) / L[-1]
    near = np.flatnonzero(d < NEAR)
    return (pd.DataFrame(dict(t=t, x=line[:, 0], y=line[:, 1], truth=f, band_lo=f - 1.645 * sd, band_hi=f + 1.645 * sd,
                              prediction=mu, prediction_rough=mu_r)), near, ts)


# ---- drawing helpers
def strip(ax, rows, **kw):
    h, texts = S.strip(ax, 0.9, 0.9, CW - 1.8, rows, **kw)
    STRIPS.append((ax, (0.9 + 1.0, 0.9, CW - 0.9 - 1.0, 0.9 + h), texts))
    return h


def marker(ax, x, y, mk, s, face=S.INK, edge="white", z=5):
    ax.scatter([x], [y], s=s, marker=mk, facecolor=face, edgecolor=edge, linewidths=0.35, zorder=z)


def dodge(x, gap=0.75, dy=0.55):
    """Vertical offsets that move a dot off its left neighbour when the two would overlap (alternately up, down)."""
    o = np.argsort(x); off = np.zeros(len(x)); last, k = -np.inf, 0
    for i in o:
        if x[i] - last < gap:
            k += 1; off[i] = dy if k % 2 else -dy
        else:
            k = 0
        last = x[i]
    return off


# ---- panels
def panel_a(fig, x0, s, y, first):
    ax = S.card_panel(fig, x0, Y1, CW, H1, "a", "Smooth map, three decisions", tx=TX)
    y0 = H1 - 5.0 - MH
    tr = first["tr"]
    S.shaded_map(ax, MX, y0, MW, MH, field=lambda p: smoother(s[tr], y[tr], p), lines="#c4bfb2", n=(161, 113))
    yy = np.linspace(0, MH, 113)
    for k in range(len(BORDERS)):                                  # country borders: dash-dot
        ax.plot(border_x(k, yy) + MX, yy + y0, color=S.INK2, lw=0.5, ls=BORDER_LS, zorder=3)
    th = np.linspace(0, 2 * np.pi, 181)
    for r in (6.0, 12.0):                                          # equal travel cost from the depot
        for piece in S.clip_line(np.asarray(DEPOT) + r * np.c_[np.cos(th), np.sin(th)], (0, MW, 0, MH)):
            ax.plot(piece[:, 0] + MX, piece[:, 1] + y0, color=S.GREY_M, lw=0.45, ls=COST_LS, zorder=3)
    ax.scatter(s[:, 0] + MX, s[:, 1] + y0, s=1.3, color=S.GREY_L, linewidths=0, zorder=4)
    for g, (mk, sz) in MARK.items():
        p = s[first["sel"][g]]
        ax.scatter(p[:, 0] + MX, p[:, 1] + y0, s=sz, marker=mk, facecolor=S.INK, edgecolor="white", linewidths=0.35,
                   zorder=5)
    marker(ax, DEPOT[0] + MX, DEPOT[1] + y0, "p", 18, face="white", edge=S.INK, z=6)
    ax.plot(PATH[:, 0] + MX, PATH[:, 1] + y0, color=S.INK, lw=0.5, zorder=6)
    p = s[np.setdiff1d(first["sel"]["carbon_rough"], first["sel"]["carbon"])]   # picked only on the rough map: open
    ax.scatter(p[:, 0] + MX, p[:, 1] + y0, s=MARK["carbon"][1], marker="^", facecolor="white", edgecolor=S.INK,
               linewidths=0.45, zorder=5)
    for p, lab, d in ((A, "A", (-1.0, -0.6)), (B, "B", (-0.25, -1.25))):   # B: a spot clear of the flank's contours
        ax.text(p[0] + MX + d[0], p[1] + y0 + d[1], lab, fontsize=S.FS["note"], fontweight="bold", ha="center",
                va="center", zorder=7)
    yk = y0 - 3.0
    for k, (g, lab) in enumerate((("carbon", "Carbon: the highest K;"),
                                  ("aid", "Aid: the lowest K in each country"),
                                  ("flood", "Flood relief: a budget paying travel"))):
        mk, sz = MARK[g]
        marker(ax, TX + 0.7, yk - S.LINE * k + 0.6, mk, sz)
        x = S.run(ax, TX + 1.8, yk - S.LINE * k, [(lab, {})], gap=1.0)
        if g == "carbon":                                          # and the rough map's picks, open
            marker(ax, x + 0.7, yk + 0.6, mk, sz, face="white", edge=S.INK)
            S.run(ax, x + 1.8, yk, [("rough map", {})])
    x = S.run(ax, TX + 1.8, yk - S.LINE * 3, [("costs from the depot", {})], gap=1.0)
    marker(ax, x + 0.6, yk - S.LINE * 3 + 0.6, "p", 18, face="white", edge=S.INK)
    for k, (lab, kw) in enumerate((("Profile A–B, drawn in c", dict(color=S.INK, lw=0.5)),
                                   ("Equal travel cost", dict(color=S.GREY_M, lw=0.45, ls=COST_LS)))):
        yy = yk - S.LINE * (4 + k)
        ax.plot([TX, TX + 1.4], [yy + 0.55] * 2, solid_capstyle="butt", **kw)
        x = S.run(ax, TX + 1.8, yy, [(lab, {})], gap=2.0)
        if k == 1:                                                 # and the borders, on the same row
            ax.plot([x, x + 2.6], [yy + 0.55] * 2, color=S.INK2, lw=0.5, ls=BORDER_LS)
            S.run(ax, x + 3.0, yy, [("Border", {})])
    strip(ax, ["quotas follow", "countries, budgets follow travel", "distance, and the rules favour the",
               "map's extremes."], head="Decisions are spatial:", n=4)
    return ax


def panel_b(fig, x0, summ, z):
    ax = S.card_panel(fig, x0, Y1, CW, H1, "b", "Valid on average only", tx=TX)
    h, XR = H1, 3.0
    xa, xb = TX + 0.4, CW - TX - 0.4
    X = lambda v: xa + (xb - xa) * (np.clip(v, -XR, XR) + XR) / (2 * XR)      # noqa: E731
    # key: the band is the split-CP interval (one width per map), ticks the width 90% coverage needs there
    yk = h - 7.2
    ax.add_patch(Rectangle((xa, yk - 0.2), 3.0, 1.6, facecolor=S.GRID, edgecolor="none"))
    S.run(ax, xa + 3.6, yk, [("Split-CP interval, one width per map", {})])
    yk -= S.LINE
    ax.plot([xa + 1.5] * 2, [yk - 0.3, yk + 1.5], color=S.INK, lw=0.7)
    x = S.run(ax, xa + 3.6, yk, [("Needed for 90% here", {})], gap=2.0)
    ax.scatter([x + 0.5], [yk + 0.6], s=3.0, color=S.MISS, linewidths=0)
    S.run(ax, x + 1.4, yk, [("Missed", {})])
    rng = np.random.default_rng(1)
    pitch, band = 6.3, 2.8
    y = yk - 3.6                                                  # baseline of the first row label
    for g, lab in GROUPS:
        mk = MARK.get(g.replace("_rough", ""))
        if mk:
            marker(ax, TX + 0.7, y + 0.6, *mk, **(dict(face="white", edge=S.INK) if g == "carbon_rough" else {}))
        ax.text(TX + (1.8 if mk else 0.0), y, lab, fontsize=S.FS["note"], ha="left", va="baseline")
        ax.text(xb, y, f"covers {summ.loc[g, 'coverage']:.2f}", fontsize=S.FS["note"], ha="right", va="baseline")
        yb = y - 1.1 - band
        ax.add_patch(Rectangle((X(-1), yb), X(1) - X(-1), band, facecolor=S.GRID, edgecolor="none", zorder=1))
        need = summ.loc[g, "needed_halfwidth"]
        for v in (-need, need):
            ax.plot([X(v)] * 2, [yb - 0.25, yb + band + 0.25], color=S.INK, lw=0.7, zorder=4)
        zz = np.concatenate(z[g][:SHOW]); zz = rng.choice(zz, min(len(zz), 110), replace=False)
        yy = yb + 0.45 + (band - 0.9) * rng.uniform(size=len(zz))
        miss = np.abs(zz) > 1
        ax.scatter(X(zz[~miss]), yy[~miss], s=2.0, color=S.GREY_M, linewidths=0, zorder=3)
        ax.scatter(X(zz[miss]), yy[miss], s=2.0, color=S.MISS, linewidths=0, zorder=3)
        y -= pitch
    ya = y + pitch - 1.1 - band - 0.9                             # axis under the last row
    ax.plot([X(-XR), X(XR)], [ya, ya], color=S.INK2, lw=S.LW["axis"], solid_capstyle="butt")
    for v, lab in ((-2, "−2q"), (-1, "−q"), (0, "0"), (1, "q"), (2, "2q")):
        ax.plot([X(v)] * 2, [ya, ya - 0.7], color=S.INK2, lw=S.LW["axis"])
        ax.text(X(v), ya - 2.6, lab, fontsize=S.FS["tick"], ha="center", va="baseline")
    ax.text((xa + xb) / 2, ya - 5.0, "Outcome minus prediction (q: half-width)", fontsize=S.FS["note"],
            color=S.INK2, ha="center", va="baseline")
    strip(ax, ["valid on average,", "wrong at selected units: too narrow", "where errors are large or share a",
               "sign, too wide where small."], head="Winner's curse:", n=4)
    return ax


def panel_c(fig, x0, prof, near, ts, s, y, first):
    ax = S.card_panel(fig, x0, Y1, CW, H1, "c", "Why selected places differ", tx=TX)
    h = H1
    xa, xb = TX + 1.5, CW - TX - 1.5                              # A and B sit just outside, on the baseline
    T = lambda t: xa + (xb - xa) * np.asarray(t)                                 # noqa: E731
    vmin, vmax = 0.0, prof.band_hi.max() + 0.02
    ph = 10.0
    titles = ["Errors grow with the value", "Maps flatten peaks or invent them", "Extremes sit where samples are few"]
    yt = h - 7.2
    for k, title in enumerate(titles):
        ax.text(TX, yt, f"{k + 1}", fontsize=S.FS["note"], fontweight="bold", ha="left", va="baseline")
        ax.text(TX + 1.7, yt, title, fontsize=S.FS["note"], ha="left", va="baseline")
        yb = yt - 1.4 - ph
        V = lambda v, yb=yb: yb + ph * (np.asarray(v) - vmin) / (vmax - vmin)    # noqa: E731
        if k == 0:
            ax.fill_between(T(prof.t), V(prof.band_lo), V(prof.band_hi), color=S.GRID, linewidth=0, zorder=1)
            ax.plot(T(prof.t), V(prof.truth), color=S.INK, lw=0.8, zorder=3)
            ax.scatter(T(ts[near]), V(y[near]), s=1.6, color=S.GREY_M, linewidths=0, zorder=4)
        elif k == 1:
            gap = prof.truth > prof.prediction
            ax.fill_between(T(prof.t), V(prof.prediction), V(prof.truth), where=gap, color=S.MISS, alpha=0.18, linewidth=0,
                            zorder=1)
            ax.plot(T(prof.t), V(prof.prediction_rough), color=S.GREY_M, lw=0.5, zorder=2)
            pk = np.intersect1d(near, np.setdiff1d(first["sel"]["carbon_rough"], first["sel"]["carbon"]))
            for j in pk:                                                 # its overshoot there: map minus truth
                ax.plot([T(ts[j])] * 2, [V(truth(s[j : j + 1])[0]), V(first["mu_r"][j])], color=S.MISS, lw=0.6,
                        solid_capstyle="butt", zorder=4)
            ax.scatter(T(ts[pk]), V(first["mu_r"][pk]), s=MARK["carbon"][1] * 0.8, marker="^", facecolor="white",
                       edgecolor=S.INK, linewidths=0.45, zorder=5)
            ax.plot(T(prof.t), V(prof.truth), color=S.INK, lw=0.8, zorder=3)
            ax.plot(T(prof.t), V(prof.prediction), color=S.INK2, lw=0.8, ls=(0, (2.4, 1.4)), zorder=3)
            for j, (lab, col, lw, ls) in enumerate((("Truth", S.INK, 0.8, "-"), ("Smooth map", S.INK2, 0.8, (0, (2.4, 1.4))),
                                                    ("Rough map", S.GREY_M, 0.5, "-"))):
                yy = yb + ph - 1.4 - 2.4 * j
                ax.plot([xa, xa + 3.0], [yy + 0.55] * 2, color=col, lw=lw, ls=ls)
                ax.text(xa + 3.6, yy, lab, fontsize=S.FS["note"], ha="left", va="baseline")
        else:
            sparse = np.flatnonzero(thinning(prof[["x", "y"]].values) > 0.36)          # where sampling halves
            t0, t1 = float(prof.t[sparse[0]]), float(prof.t[sparse[-1]])
            ax.add_patch(Rectangle((T(t0), yb), T(t1) - T(t0), ph, facecolor=S.SHADE, edgecolor="none", zorder=0))
            ax.plot(T(prof.t), V(prof.truth), color=S.GREY_L, lw=0.8, zorder=2)
            for t in ts[near]:
                ax.plot([T(t)] * 2, [yb, yb + 1.3], color=S.INK2, lw=0.45, solid_capstyle="butt", zorder=3)
            ax.text((T(t0) + T(t1)) / 2, yb + 2.2, "few samples", fontsize=S.FS["note"], color=S.INK2, ha="center",
                    va="baseline", zorder=4)
        ax.plot([xa, xb], [yb, yb], color=S.GREY_L, lw=S.LW["axis"], zorder=2)
        for t, lab, ha, dx in ((0, "A", "right", -0.35), (1, "B", "left", 0.35)):
            ax.text(T(t) + dx, yb, lab, fontsize=S.FS["note"], fontweight="bold", ha=ha, va="baseline", zorder=5)
        yt = yb - 2.7
    strip(ax, ["the selected places", "are the least typical in error size,",
               "map distortion or sampling."], head="Three causes:", n=3)
    return ax


def panel_d(fig, x0, s, y, country, first):
    ax = S.card_panel(fig, x0, Y1, CW, H1, "d", "This paper's answer", tx=TX)
    h = H1
    mu, cal, te, tr = first["mu"], first["cal"], first["te"], first["tr"]
    # 1  the reference set of a top-K rule: calibration units above the same threshold T as the winners
    yk = h - 7.2
    S.units(ax, [[TX + 0.65, yk + 0.55]], "reference", z=5)
    S.lines(ax, TX + 1.8, yk, ["Calibrate only on places the rule",
                               "would also select (reference set)"])
    sel = refsets.topk(mu[cal], mu[te], K)
    T_ = np.sort(mu[te])[::-1][K]                                  # the (K+1)-th largest test prediction
    lo, hi = np.quantile(mu[np.r_[cal, te]], 0.88), mu[np.r_[cal, te]].max()
    xa, xb = TX + 7.0, CW - TX - 0.8
    X = lambda v: xa + (xb - xa) * (np.asarray(v) - lo) / (hi - lo)              # noqa: E731
    rows = (("Test", te, yk - 6.9), ("Calib.", cal, yk - 9.4))
    for lab, idx, yy in rows:
        ax.text(TX, yy - 0.55, lab, fontsize=S.FS["note"], color=S.INK2, ha="left", va="baseline")
        ax.plot([xa - 0.6, xb + 0.6], [yy] * 2, color=S.GRID, lw=0.4, zorder=1)
        m = idx[mu[idx] >= lo]
        xy = np.c_[X(mu[m]), yy + dodge(X(mu[m]))]
        on = np.isin(m, te[sel.selected]) if lab == "Test" else mu[m] > T_
        S.units(ax, xy[~on], "test" if lab == "Test" else "calibration", z=3)
        S.units(ax, xy[on], "selected" if lab == "Test" else "reference", z=4)
    ax.plot([X(T_)] * 2, [yk - 10.4, yk - 5.6], color=S.INK, lw=0.5, ls=(0, (2, 1.2)), zorder=5)
    ax.text(X(T_), yk - 5.1, "$T$", fontsize=S.FS["note"], ha="center", va="baseline")
    ax.text(xb + 0.6, yk - 5.1, r"predicted value $\hat\mu$ →", fontsize=S.FS["note"], color=S.INK2, ha="right",
            va="baseline")
    # 2  borrow strength across scales, as far as the data allow (the basis of SC-GeoCCP)
    y2 = yk - 13.2
    S.dot(ax, TX + 0.65, y2 + 0.6, "SC-GeoCCP", s=8, z=5)
    S.lines(ax, TX + 1.8, y2, ["Borrow strength across scales only",
                               "as far as the data allow (RegionCP)"])
    Bs = basis.Multiscale(s[tr], levels=(4, 12), min_per_centre=1)
    box = np.array([[0, 0], [MW, 0], [MW, MH], [0, MH]], float)
    aid_country = pd.Series(country[first["sel"]["carbon"]]).mode()[0]
    whole = lambda T, z: ax.add_patch(Polygon(T(box), closed=True, facecolor=S.VIOLET, alpha=0.14,   # noqa: E731
                                              edgecolor="none", zorder=z + 0.4))
    bumps = lambda lv: lambda T, z: S.bump_layer(ax, T, Bs.centres[Bs.level == lv], Bs.h[Bs.level == lv],  # noqa: E731
                                                 (MW, MH), z)
    bottom = S.layer_stack(ax, TX + 0.4, y2 - S.LINE - 1.4, (MW, MH), [
        ("Whole map", whole), ("Regions", bumps(0)), ("Localities", bumps(1)),
        ("Countries", lambda T, z: S.cell_layer(ax, T, country_cells(), z, fill=aid_country))],
        pitch=3.9, view=dict(sx=0.42, kx=0.20, sy=0.12))
    # 3  bound the decision itself
    y3 = bottom - 3.2
    ax.text(TX, y3, "Bound the decision: totals, funding", fontsize=S.FS["note"], ha="left",
            va="baseline")
    x = TX
    for lab in (r"Pr(total $\geq L$) $\geq 1-\delta$", r"FDR $\leq q$"):
        t = ax.text(x + 1.0, y3 - 3.5, lab, fontsize=S.FS["note"], ha="left", va="baseline", zorder=3)
        bb = t.get_window_extent(fig.canvas.get_renderer()); inv = ax.transData.inverted()
        x1 = inv.transform((bb.x1, 0))[0]
        S.card(ax, x, y3 - 4.4, x1 - x + 1.0, 3.0, fc=S.SHADE, ec="none", r=0.6, z=1)
        x = x1 + 2.2
    strip(ax, ["valid where", "the decision lands, region by region", "where the data allow, for totals",
               "and funding too."], head="This paper (Fig. 1):", n=4)
    return ax


def main():
    S.apply()
    s, f, y, country = population()
    counts, z, first = simulate(s, y, country)
    summ = summary(counts, z)
    prof, near, ts = profile(s, y, first)
    fig = S.figure(W, H)
    A_ = panel_a(fig, 0.0, s, y, first)
    B_ = panel_b(fig, CW + GAP, summ, z)
    C_ = panel_c(fig, 2 * (CW + GAP), prof, near, ts, s, y, first)
    D_ = panel_d(fig, 3 * (CW + GAP), s, y, country, first)
    bad = {ax.texts[0].get_text(): S.overflow(ax, (0.9, 0.3, CW - 0.9, H1 - 0.3)) for ax in (A_, B_, C_, D_)}
    for ax, box, texts in STRIPS:
        bad.setdefault("strips", []).extend(S.overflow(ax, box, texts))
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise SystemExit(f"text outside its card or strip: {bad}")
    role = np.full(len(s), "train", object); role[first["cal"]] = "calibration"; role[first["te"]] = "test"
    units = pd.DataFrame(dict(x=s[:, 0], y=s[:, 1], truth=f, noise_sd=noise_sd(f), outcome=y, country=country,
                              role_split1=role, map_split1=first["mu"], map_rough_split1=first["mu_r"]))
    for g in list(MARK) + ["carbon_rough"]:
        units[f"picked_{g}_split1"] = np.isin(np.arange(len(s)), first["sel"][g])
    S.source_data("nc_fig1a", units)
    S.source_data("nc_fig1b", summ.reset_index())
    S.source_data("nc_fig1c", prof)
    S.save(fig, "nc_fig1_concept", axes=[A_, B_, C_, D_], panel_ids=list("abcd"), row_groups=[["a", "b", "c", "d"]],
           column_groups=[])
    print(f"units {len(s)}, split-CP half-width (split 1) {first['q']:.3f}")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
