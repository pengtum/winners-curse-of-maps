"""Supplementary Fig. 25 (main Fig. 2 before 2026-11-05) | From selection to certified decisions: the method flow (paper/FIGURE_PLAN.md).

Left to right, every card ends in what it guarantees (or assumes):
a  input: one finite map whose labelled units are split at random into a training fold, calibration and test units;
   any model (prediction mu, scale sigma); a decision rule
b  step 1, selection: the rule picks S-hat from the test units (drawn: the top K by mu)
c  step 2, the reference set R_j: the calibration units the rule would also select had they swapped places with j,
   with the closed forms for top-K and the three rules of Proposition 1 (excp_select/refsets.py), and, last, a rule
   that selects a whole fixed bin b of mu: its reference set is the bin, so Mondrian CP (calibration within fixed bins
   of the prediction) is reference-set calibration for that rule (Jin & Ren 2025, App. A.2)
d  step 3, calibration on R_j: JOMI (one quantile of the scores in R_j) and SC-GeoCCP (a threshold over a multiscale
   spatial basis with an L1 penalty; Theorem 1 and Proposition 2, theory/SC_GeoCCP.md)
e-g step 4, the decision layer: an interval per selected unit, a certified total over S-hat (pool lemma, exact rank
   envelope; theory/POOL_LEMMA.md T1) and certified funding (cfBH on the reference set; T2)
footer: where the validity comes from (the random split; design-based) and what may not enter the calibration.

The mini-maps show a synthetic map (seeded; a schematic, not data). Its reference set, basis, thresholds, total bound
and funding decisions are computed with the paper's own code (excp_select: refsets.topk, basis.Multiscale, the JOMI
and SC-GeoCCP presets, totals.envelope_exact / lower_total; cfBH as in experiments/run_cfbh.py with the sign flipped
for a top-K rule), so the parts show what the method does on that map. The seed and the eligibility line were chosen
so that one map shows all three decisions (split CP missing some winners, a total bound, part of S-hat funded); the
rates on real data are in Figs 2-5 (renumbered 2026-11-09).
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_fig2.py
writes figs/nc_fig2_method.{pdf,svg,png}, figs/source_data/nc_fig2{,_summary}.csv, figs/qa/
"""
import os, sys
from math import ceil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
from excp_select import basis, methods as Mt, refsets, totals         # noqa: E402

# ---- layout (mm)
W, H = 183.0, 119.04
GAP = 4.0                                       # between cards (the flow arrows sit here)
CW1, H1 = (W - 3 * GAP) / 4, 67.54              # row 1: a-d (2026-10-30: +2.54 mm, one text row, for the bin row in c)
CW2, H2 = (W - 2 * GAP) / 3, 31.0               # row 2, the decision layer: e-g
Y1 = H - 0.5 - H1                               # bottom of row 1
YH = Y1 - 5.2                                   # baseline of the decision-layer header
Y2 = YH - 2.6 - H2                              # bottom of row 2
FOOT = S.strip_height(3)
YF = Y2 - 2.2 - FOOT                            # bottom of the footer
MW, MH = 40.0, 24.0                             # mini-map; the synthetic map's coordinates are these millimetres
MX, MTOP = (CW1 - MW) / 2, H1 - 5.0             # mini-map inset and top, in a card's coordinates
TX = 1.4                                        # text inset in a card

# ---- the synthetic map: a true field with three hills; noise that grows fast on the hills, which the scale model
# misses; and a map that is too high on part of the main hill. So the top-K winners are over-predicted and noisier
# than their scale says (the winner's curse), and split CP calibrated on all units under-covers them.
SEED, K, L1, Q = 6, 20, 0.95, 0.10
DELTA = 1 / 3                                   # total bound at the risk VCS declares (as Fig. 4's exact bound)
HILLS = [(29.0, 14.5, 1.0, 5.0), (9.0, 7.0, 0.55, 4.5), (16.0, 19.5, 0.35, 3.0)]      # x, y, height, sd (mm)
ERROR = [(31.0, 11.5, 0.30, 3.2), (7.0, 17.0, -0.12, 4.0)]
STATES = np.array([[7, 18], [8, 6], [20, 5], [21, 17], [32, 20], [33, 7]], float)   # admin-unit seeds (Voronoi)
C_ELIG = 0.6                                    # eligibility line of the funding card (eligible: outcome > c)
STRIPS = []                                     # (axes, inner box, texts) of every strip, for the overflow guard


def bumps(s, spec):
    s = np.atleast_2d(s)
    return sum(a * np.exp(-((s[:, 0] - x) ** 2 + (s[:, 1] - y) ** 2) / (2 * b ** 2)) for x, y, a, b in spec)


def mu_of(s):
    return 0.1 + bumps(s, HILLS) + bumps(s, ERROR)


def synthetic_map():
    rng = np.random.default_rng(SEED)
    gx, gy = np.meshgrid(np.arange(1.0, MW - 0.5, 1.6), np.arange(1.0, MH - 0.5, 1.6))
    s = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-0.45, 0.45, (gx.size, 2))
    f = 0.1 + bumps(s, HILLS)
    out = np.maximum(f + (0.05 + 0.22 * f ** 2) * rng.standard_normal(len(s)), 0.0)  # outcome >= 0: a known floor
    mu = mu_of(s)
    role = np.full(len(s), "train", object)
    p = rng.permutation(len(s)); n_tr, n_c = int(0.4 * len(s)), int(0.3 * len(s))
    role[p[n_tr:n_tr + n_c]] = "calibration"; role[p[n_tr + n_c:]] = "test"
    unit = np.argmin(((s[:, None] - STATES[None]) ** 2).sum(-1), 1)
    return pd.DataFrame(dict(x=s[:, 0], y=s[:, 1], role=role, mu=mu, sigma=0.06 + 0.07 * np.maximum(mu, 0.1),
                             outcome=out, unit=unit))


def run_method(D):
    """Selection, reference set, thresholds, the certified total and the funding decisions on the synthetic map."""
    xy = D[["x", "y"]].values
    tr, cal, test = (np.flatnonzero(D.role.values == r) for r in ("train", "calibration", "test"))
    sel = refsets.topk(D.mu.values[cal], D.mu.values[test], K)                    # top-K: one R for every j
    Sel, R = test[sel.selected], cal[sel.member[0]]
    score = (np.abs(D.outcome - D.mu) / D.sigma).values
    B = basis.Multiscale(xy[tr], levels=(4, 12, 36), min_per_centre=1).with_units(D.unit.values[tr])
    Phi = B(xy, D.unit.values)
    ctx = dict(S_cal=score[cal], w_cal=np.ones(len(cal)), w_test=np.ones(len(test)), Phi_cal=Phi[cal],
               Phi_test=Phi[test])
    q_jomi = Mt.jomi(sel).thresholds(ctx, S.ALPHA)
    q_scg = Mt.sc_geoccp(sel, L1, B.penalize).thresholds(ctx, S.ALPHA)
    k = ceil((1 - S.ALPHA) * (len(cal) + 1)); q_split = np.sort(score[cal])[k - 1]
    # certified total (pool lemma + exact rank envelope; support floor 0)
    C = totals.envelope_exact(K + len(R), K, DELTA)
    v = (D.outcome - D.mu).values
    L = D.mu.values[Sel].sum() + totals.lower_total(v[R], K, 0.0 - D.mu.values[Sel].max(), C)
    # certified funding: H_j "not eligible" (outcome <= c); p-values from R; BH at level q
    g = ((C_ELIG - D.mu) / D.sigma).values
    V_R = 1e6 * (D.outcome.values[R] > C_ELIG) + g[R]
    pv = (1 + (V_R[None, :] < g[Sel][:, None]).sum(1)) / (len(R) + 1)
    o = np.argsort(pv); ok = np.flatnonzero(pv[o] <= Q * np.arange(1, K + 1) / K)
    fund = np.zeros(K, bool)
    if len(ok):
        fund[o[:ok.max() + 1]] = True
    D = D.assign(selected=False, reference=False, funded=False, q_split=q_split)
    D.loc[Sel, "selected"] = True; D.loc[R, "reference"] = True; D.loc[Sel[fund], "funded"] = True
    D.loc[Sel, "q_jomi"] = q_jomi; D.loc[Sel, "q_scgeoccp"] = q_scg; D.loc[Sel, "p_cfbh"] = pv
    y, m = D.outcome.values[Sel], D.mu.values[Sel]; sd = D.sigma.values[Sel]
    summ = dict(units=len(D), train=len(tr), calibration=len(cal), test=len(test), K=K, R=len(R),
                q_split=q_split, q_jomi=float(q_jomi[0]), q_scgeoccp_min=float(q_scg.min()),
                q_scgeoccp_max=float(q_scg.max()), miss_split=int((np.abs(y - m) > sd * q_split).sum()),
                miss_jomi=int((np.abs(y - m) > sd * q_jomi).sum()), miss_scgeoccp=int((np.abs(y - m) > sd * q_scg).sum()),
                map_total=float(m.sum()), true_total=float(y.sum()), certified_total=float(L), delta=DELTA,
                c_eligible=C_ELIG, eligible=int((y > C_ELIG).sum()), funded=int(fund.sum()),
                funded_not_eligible=int((y[fund] <= C_ELIG).sum()), q=Q)
    return D, B, summ


# ---- drawing helpers
def card_axes(fig, x0, y0, w, h, letter, title, step=None, tag=None):
    return S.card_panel(fig, x0, y0, w, h, letter, title, step=step, tag=tag, tx=TX)


def strip(ax, w, rows, **kw):
    h, texts = S.strip(ax, 0.9, 0.9, w - 1.8, rows, **kw)
    STRIPS.append((ax, (0.9 + 1.0, 0.9, w - 0.9 - 1.0, 0.9 + h), texts))
    return h


def mini_map(ax, x0, y0, shade=True):
    S.shaded_map(ax, x0, y0, MW, MH, field=mu_of if shade else None)


def pts(D, m, x0, y0):
    return np.c_[D.x.values[m] + x0, D.y.values[m] + y0]


# ---- panels
def panel_a(fig, x0, D):
    ax = card_axes(fig, x0, Y1, CW1, H1, "a", "Input: one finite map")
    y0 = MTOP - MH
    mini_map(ax, MX, y0, shade=False)
    for role in ("train", "calibration", "test"):
        S.units(ax, pts(D, D.role.values == role, MX, y0), role, z=3)
    # the split bar doubles as the key: fill = role, as on the map
    yb = y0 - 5.6
    ax.text(MX, yb + 3.1, "Random split of the labelled units", fontsize=S.FS["note"], color=S.INK2, ha="left",
            va="baseline")
    xb = MX
    for role, frac, lab, ink in (("train", 0.4, "Training fold", S.INK), ("calibration", 0.3, "Calibration", S.INK),
                                 ("test", 0.3, "Test", "white")):
        fc, ec, _, _ = S.ROLE[role]
        ax.add_patch(Rectangle((xb, yb), MW * frac, 2.4, facecolor=fc, edgecolor=S.INK2 if fc == "white" else fc,
                               lw=0.45, zorder=3))
        ax.text(xb + MW * frac / 2, yb + 1.2, lab, fontsize=S.FS["note"], color=ink, ha="center", va="center_baseline",
                zorder=4)
        xb += MW * frac
    S.lines(ax, TX, yb - 3.3, ["Any model, fitted on the training fold:",
                               r"prediction $\hat\mu$ and scale $\hat\sigma$ (e.g. RF, GeoFM)",
                               r"A rule: top or bottom $K$, a quota per group,",
                               "a travel-cost budget or a publication mask"])
    strip(ax, CW1, ["Calibration and test units", "are a random split of one fixed map: the only",
                    "randomness the guarantees use."], head="Assumption", n=5)
    return ax


def panel_b(fig, x0, D):
    ax = card_axes(fig, x0, Y1, CW1, H1, "b", "Selection", step="1")
    y0 = MTOP - MH
    mini_map(ax, MX, y0)
    S.units(ax, pts(D, (D.role.values == "test") & ~D.selected.values, MX, y0), "test", z=3)
    S.units(ax, pts(D, D.selected.values, MX, y0), "selected", z=4)
    yk = y0 - 3.0
    x = S.keyrow(ax, MX, yk, [("test", "Test unit"), ("selected", r"Selected, $\hat S$")])
    for k, c in enumerate(S.MAPRAMP(np.linspace(0, 1, 6))):           # shading key: predicted value
        ax.add_patch(Rectangle((x + 0.9 * k, yk - 0.1), 0.9, 1.6, facecolor=c, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x, yk - 0.1), 5.4, 1.6, facecolor="none", edgecolor=S.GRID, lw=0.4, zorder=4))
    ax.text(x + 6.0, yk, r"$\hat\mu$", fontsize=S.FS["note"], ha="left", va="baseline")
    S.lines(ax, TX, yk - 3.6, [r"The rule picks $\hat S$ from the test units, here",
                              rf"the top $K$ = {K} by predicted value $\hat\mu$.",
                              r"Split-CP intervals cover $1-\alpha$ of all units,",
                              "not of the selected ones: the winner's curse."])
    strip(ax, CW1, ["The rule uses covariates and", "predictions, never labels, and treats all",
                    "calibration units alike."], head="Requirement", n=5)
    return ax


def panel_c(fig, x0, D):
    ax = card_axes(fig, x0, Y1, CW1, H1, "c", r"Reference set $R_j$", step="2")
    y0 = MTOP - MH
    mini_map(ax, MX, y0)
    cal = D.role.values == "calibration"
    S.units(ax, pts(D, cal & ~D.reference.values, MX, y0), "calibration", z=3)
    S.units(ax, pts(D, D.reference.values, MX, y0), "reference", z=4)
    S.units(ax, pts(D, D.selected.values, MX, y0), "selected", z=4)
    # the swap: a selected unit j and a reference unit i whose connecting arc runs clear of the other units
    j, i, rad, lj, li = swap_pair(D, MX, y0)
    pj, pi = pts(D, [j], MX, y0)[0], pts(D, [i], MX, y0)[0]
    ax.scatter([pj[0], pi[0]], [pj[1], pi[1]], s=24, facecolor="none", edgecolor=S.INK, linewidths=0.55, zorder=7)
    ax.add_patch(FancyArrowPatch(pj, pi, arrowstyle="<|-|>,head_length=2.2,head_width=0.9", mutation_scale=1,
                                 connectionstyle=f"arc3,rad={rad}", color=S.INK, lw=0.55, shrinkA=3.4, shrinkB=3.4,
                                 zorder=7))
    for spot, lab in ((lj, "j"), (li, "i")):
        ax.text(*spot, f"${lab}$", fontsize=S.FS["label"], ha="center", va="center", zorder=8)
    yk = y0 - 3.0
    S.keyrow(ax, MX, yk, [("calibration", "Calibration unit"), ("reference", r"In $R_j$")])
    y = S.lines(ax, TX, yk - 3.6, [r"$R_j$: the calibration units $i$ the rule would also",
                                   r"select had $i$ and $j$ swapped places:"])
    for rule, form in (("Top $K$", r"$\hat\mu_i>\theta$, $\theta$ the $(K{+}1)$-th test $\hat\mu$"),
                       ("Quota $K_g$", r"$\hat\mu_i>\theta_{g(i)}$, one per group"),
                       ("Budget $B$", r"$Q_{-j}(r_i)+t_i\leq B$, $r$ = value/cost"),
                       ("Mask", r"$W_i+K_h(s_i,s_j)\geq(1-\alpha)/\alpha$"),
                       # last, a rule that selects a whole fixed bin b of mu-hat: its reference set is the bin, so
                       # Mondrian CP is reference-set calibration for that rule (Jin & Ren 2025, App. A.2)
                       ("Whole bin $b$", r"$\hat\mu_i\in b$: Mondrian CP")):
        if rule.startswith("Whole bin"):    # 0.25 mm more lead: the accent on mu-hat would reach the Mask row's
            y -= 0.25                       # subscripts (the audited text boxes then clear by 0.19 pt, as Top K/Quota)
        ax.text(TX + 0.4, y, rule, fontsize=S.FS["note"], color=S.INK2, ha="left", va="baseline")
        ax.text(TX + 14.6, y, form, fontsize=S.FS["note"], color=S.INK, ha="left", va="baseline")
        y -= S.LINE
    strip(ax, CW1, [r"Given that $j$ is selected,", r"$j$ is exchangeable with the units of $R_j$",
                    r"(JOMI): their scores calibrate $j$. The closed",
                    "forms agree with literal swaps (300 random", "instances)."], n=5)
    return ax


def arc(p0, p2, rad, t=np.linspace(0.12, 0.88, 20)):
    """Points of matplotlib's arc3 connection from p0 to p2 (a quadratic Bezier; the canvas is isotropic)."""
    d = p2 - p0; c1 = (p0 + p2) / 2 + rad * np.array([d[1], -d[0]]); t = t[:, None]
    return (1 - t) ** 2 * p0 + 2 * t * (1 - t) * c1 + t ** 2 * p2


LAB = (0.35, 1.1)                               # half-width, half-height of a one-letter 6 pt label (mm)


def label_spot(p, avoid, others, box):
    """Where to put the one-letter label of the unit at p: beside it (the letters are narrow), clear of the ring
    around p, pointing away from every direction in `avoid` (the swap partner, the side the arc bows to) and inside
    the map box. Returns the spot and its clearance (mm) from the other units' marker edges."""
    best, out = -np.inf, p
    for off in ((1.65, 0.0), (-1.65, 0.0), (1.65, 0.9), (-1.65, 0.9), (1.65, -0.9), (-1.65, -0.9)):
        off = np.asarray(off); c = p + off
        if any(np.dot(off, v) / (np.hypot(*off) * np.hypot(*v)) > 0.2 for v in avoid):
            continue
        if not (box[0] + LAB[0] < c[0] < box[2] - LAB[0] and box[1] + LAB[1] < c[1] < box[3] - LAB[1]):
            continue
        gap = np.hypot(np.maximum(np.abs(others[:, 0] - c[0]) - LAB[0], 0),
                       np.maximum(np.abs(others[:, 1] - c[1]) - LAB[1], 0)).min() - 0.45
        if gap > best:
            best, out = gap, c
    return out, best


def swap_pair(D, x0, y0, near=(4.0, 6.5)):
    """The swap drawn in c: of the (selected j, reference i) pairs 4-6.5 mm apart and both bow directions, the one
    whose arc and two labels keep farthest from the other units drawn on the map (and stay inside it). Returns
    j, i, the arc's rad and the two label spots."""
    P = pts(D, np.ones(len(D), bool), x0, y0)
    shown = np.flatnonzero((D.role.values == "calibration") | D.selected.values)
    box = (x0, y0, x0 + MW, y0 + MH)
    best, out = -np.inf, None
    for j in np.flatnonzero(D.selected.values):
        for i in np.flatnonzero(D.reference.values):
            d = P[i] - P[j]
            if not near[0] <= np.hypot(*d) <= near[1]:
                continue
            others = P[np.setdiff1d(shown, [i, j])]
            for rad in (-0.45, 0.45):
                c = arc(P[j], P[i], rad)
                if not ((c[:, 0] > box[0] + 0.8) & (c[:, 0] < box[2] - 0.8) & (c[:, 1] > box[1] + 0.8)
                        & (c[:, 1] < box[3] - 0.8)).all():
                    continue
                bulge = np.sign(rad) * np.array([d[1], -d[0]])         # the side the arc bows to
                lj, gj = label_spot(P[j], [d, bulge], others, box)
                li, gi = label_spot(P[i], [-d, bulge], others, box)
                score = min(np.hypot(*(c[:, None, :] - others[None]).transpose(2, 0, 1)).min() - 0.45, gj, gi)
                if score > best:
                    best, out = score, (j, i, rad, lj, li)
    return out


def panel_d(fig, x0, D, B, summ):
    ax = card_axes(fig, x0, Y1, CW1, H1, "d", r"Calibration on $R_j$", step="3")
    h = H1
    # (i) JOMI: one quantile of the reference-set scores (dot plot), against split CP's quantile of all scores
    yk = h - 7.4
    S.dot(ax, TX + 0.75, yk + 0.6, "JOMI stud", s=7, z=5)
    ax.text(TX + 1.9, yk, r"JOMI: one quantile of the scores in $R_j$", fontsize=S.FS["note"], ha="left",
            va="baseline")
    sc = np.sort(((D.outcome - D.mu).abs() / D.sigma).values[D.reference.values])
    qj, qs = summ["q_jomi"], summ["q_split"]
    xa, xb, smax = TX + 0.6, CW1 - TX - 0.6, max(sc.max(), qj) * 1.06
    X = lambda v: xa + (xb - xa) * np.asarray(v) / smax                         # noqa: E731
    ya = h - 13.6
    ax.plot([xa, xb], [ya, ya], color=S.INK2, lw=S.LW["axis"], solid_capstyle="butt", zorder=2)
    col = np.floor((X(sc) - xa) / 0.95).astype(int)              # dot plot: stack dots that share a 0.95 mm bin
    for c in np.unique(col):
        m = np.flatnonzero(col == c)
        S.units(ax, np.c_[np.full(len(m), xa + (c + 0.5) * 0.95), ya + 0.62 + 0.92 * np.arange(len(m))],
                "reference", z=3)
    stack = pd.Series(col).value_counts()                        # dots per bin
    for q, c, lab, width in ((qs, S.GREY_M, "split CP", 8.2), (qj, S.INK, r"$\hat q_j$", 2.4)):
        bins = np.arange(int((X(q) + 0.45 - xa) // 0.95), int((X(q) + 0.45 + width - xa) // 0.95) + 1)
        top = max([stack.get(b, 0) for b in bins] + [0])
        yl = max(ya + 2.6, ya + 0.62 + 0.92 * top + 0.5)
        ax.plot([X(q)] * 2, [ya - 0.5, yl + 1.3], color=c, lw=0.6, zorder=4)
        ax.text(X(q) + 0.45, yl, lab, fontsize=S.FS["note"], color=S.INK2 if c == S.GREY_M else S.INK, ha="left",
                va="baseline", zorder=5)
    ax.text(xa, ya - 2.4, r"$C_j=\hat\mu_j\pm\hat\sigma_j\hat q_j$; scores $|Y-\hat\mu|/\hat\sigma$ in $R_j$",
            fontsize=S.FS["note"], color=S.INK2, ha="left", va="baseline")
    # (ii) SC-GeoCCP: a threshold over a multiscale basis anchored on the training fold
    yk = ya - 6.0
    S.dot(ax, TX + 0.75, yk + 0.6, "SC-GeoCCP", s=9, z=5)
    S.lines(ax, TX + 1.9, yk, ["RegionCP (ours): a threshold that varies",
                               r"over a basis $\phi_k$: $\hat q(x)=\Sigma_k\hat\beta_k\phi_k(x)$"])
    top = yk - S.LINE - 1.7
    unit = pd.Series(D.unit.values[D.selected.values]).mode()[0]          # the admin unit holding most winners
    bumps = lambda lv: lambda T, z: S.bump_layer(ax, T, B.centres[B.level == lv], B.h[B.level == lv], (MW, MH), z)  # noqa: E731
    bottom = S.layer_stack(ax, TX + 0.2, top, (MW, MH), [
        ("4 bumps", bumps(0)), ("12 bumps", bumps(1)), ("36 bumps", bumps(2)),
        ("Admin units", lambda T, z: S.cell_layer(ax, T, S.voronoi(STATES, (0, 0, MW, MH)), z, fill=unit))])
    y = bottom - 2.9
    S.lines(ax, TX, y, ["L1-penalized quantile regression on $R_j$: a finer",
                        "scale counts only as far as its data allow"])
    strip(ax, CW1, [r"JOMI: coverage $\geq 1-\alpha$ given", "selection. RegionCP: the same; in each",
                    r"unit (on average, each bump) $\geq 1-\alpha-$slack,",
                    r"slack $\propto$ 1/(data there) (Thm 1); +∞ only", "where JOMI is +∞ (Prop. 2)."], n=5)
    return ax


def count(n):
    return "none" if n == 0 else str(n)


def panel_e(fig, x0, D, summ):
    ax = card_axes(fig, x0, Y2, CW2, H2, "e", "Interval for each selected unit", tag="Figs 2 and 6")
    h = H2
    sel = D[D.selected].sort_values("mu", ascending=False)
    lo_s, hi_s = sel.mu - sel.sigma * sel.q_split, sel.mu + sel.sigma * sel.q_split
    lo_g, hi_g = sel.mu - sel.sigma * sel.q_scgeoccp, sel.mu + sel.sigma * sel.q_scgeoccp
    vmin, vmax = min(lo_s.min(), lo_g.min(), sel.outcome.min()), max(hi_s.max(), hi_g.max(), sel.outcome.max())
    yb, yt = h - 15.6, h - 5.4
    Y = lambda v: yb + (yt - yb) * (np.asarray(v) - vmin) / (vmax - vmin)       # noqa: E731
    xs = TX + 0.9 + 1.95 * np.arange(K)
    for x, a, b, c, d, t in zip(xs, lo_s, hi_s, lo_g, hi_g, sel.outcome):
        ax.plot([x - 0.38] * 2, Y([a, b]), color=S.GREY_M, lw=0.85, solid_capstyle="butt", zorder=3)
        ax.plot([x + 0.38] * 2, Y([c, d]), color=S.VIOLET, lw=0.85, solid_capstyle="butt", zorder=3)
        ax.plot([x - 0.62, x + 0.62], [Y(t)] * 2, color=S.INK, lw=0.7, solid_capstyle="butt", zorder=4)
    xl = xs[-1] + 2.6
    for k, (c, lw, lab) in enumerate(((S.GREY_M, 0.85, "Normalized CP"), (S.VIOLET, 0.85, "RegionCP"),
                                      (S.INK, 0.7, "True value"))):
        yy = yt - 1.3 - 2.75 * k
        if lab == "True value":
            ax.plot([xl - 0.62, xl + 0.62], [yy + 0.55] * 2, color=c, lw=lw, solid_capstyle="butt")
        else:
            ax.plot([xl, xl], [yy - 0.2, yy + 1.5], color=c, lw=lw, solid_capstyle="butt")
        ax.text(xl + 1.4, yy, lab, fontsize=S.FS["note"], ha="left", va="baseline")
    S.lines(ax, TX, h - 18.9, [rf"The {K} selected units, ranked by $\hat\mu$. Normalized split CP,",
                               f"calibrated on all units, misses {count(summ['miss_split'])} true values; RegionCP",
                               f"(step 3) misses {count(summ['miss_scgeoccp'])}. +∞ would mean abstain: survey the unit."])
    strip(ax, CW2, [r"Coverage $\geq 1-\alpha$ given selection (d)."], n=1)
    return ax


def panel_f(fig, x0, D, summ):
    ax = card_axes(fig, x0, Y2, CW2, H2, "f", r"Certified total over $\hat S$", tag="carbon credits, Fig. 5")
    h = H2
    vmax = max(summ["map_total"], summ["true_total"]) * 1.02
    xa, xb = TX + 15.0, CW2 - TX - 1.0
    X = lambda v: xa + (xb - xa) * v / vmax                                     # noqa: E731
    for k, (lab, v, c) in enumerate(((r"Map total $\sum\hat\mu$", summ["map_total"], S.GREY_M),
                                     (r"Certified $L$", summ["certified_total"], S.BLUE))):
        yy = h - 8.6 - 3.6 * k
        ax.add_patch(Rectangle((xa, yy - 0.15), X(v) - xa, 2.0, facecolor=c, edgecolor="none", zorder=3))
        ax.text(TX, yy + 0.3, lab, fontsize=S.FS["note"], ha="left", va="baseline")
    xt = X(summ["true_total"])
    ax.plot([xt, xt], [h - 13.3, h - 6.1], color=S.INK, lw=0.7, zorder=4)
    ax.text(xt, h - 5.6, "True total", fontsize=S.FS["note"], ha="center", va="baseline")
    ax.plot([xa, xa], [h - 13.3, h - 6.1], color=S.INK2, lw=S.LW["axis"], zorder=4)
    S.lines(ax, TX, h - 18.9, [r"Pool lemma: under top or bottom $K$, $\hat S$ is a uniform random",
                               r"subset of the pool $\hat S\cup R$, so the residuals in $R$ bound the total",
                               "(exact rank envelope; needs a known floor, e.g. biomass ≥ 0)."])
    strip(ax, CW2, [r"Pr(total over $\hat S\geq L$) $\geq 1-\delta$ (here $\delta$ = 1/3)."], n=1)
    return ax


def panel_g(fig, x0, D, summ):
    ax = card_axes(fig, x0, Y2, CW2, H2, "g", "Certified funding", tag="aid targeting, Fig. 4")
    h = H2
    sel = D[D.selected].sort_values("mu", ascending=False)
    xs = TX + 0.9 + 1.75 * np.arange(K); yy = h - 9.2
    for x, f in zip(xs, sel.funded):
        ax.scatter([x], [yy], s=7, marker="s", facecolor=S.BLUE if f else "white", edgecolor=S.BLUE if f else S.INK2,
                   linewidths=0.45, zorder=3)
    ax.plot([xs[0] - 0.7, xs[-1] + 0.7], [yy + 1.6] * 2, color=S.INK2, lw=0.4)
    ax.text(xs[0] - 0.7, yy + 2.2, rf"The {K} selected units by $\hat\mu$", fontsize=S.FS["note"], color=S.INK2,
            ha="left", va="baseline")
    xk = xs[-1] + 3.0
    for k, (f, lab) in enumerate(((True, "Fund"), (False, "Survey first"))):
        ky = h - 7.3 - 2.75 * k
        ax.scatter([xk], [ky + 0.55], s=7, marker="s", facecolor=S.BLUE if f else "white",
                   edgecolor=S.BLUE if f else S.INK2, linewidths=0.45, zorder=3)
        ax.text(xk + 1.2, ky, lab, fontsize=S.FS["note"], ha="left", va="baseline")
    S.lines(ax, TX, h - 18.9, [r"Conformal p-values from $R$ for the null '$j$ is not eligible', then",
                               rf"Benjamini–Hochberg at level $q$: fund the units it rejects ({summ['funded']} here)",
                               "and survey the rest (cfBH on the reference set)."])
    strip(ax, CW2, [r"False discovery rate $\leq q$ among the funded."], n=1)
    return ax


def flow(bg):
    """Arrows between the row-1 cards, the bracket from c and d into the decision layer, its header and the footer."""
    ym = Y1 + MTOP - MH / 2
    for k in range(3):
        x1 = (CW1 + GAP) * k + CW1
        S.arrow(bg, (x1 + 0.55, ym), (x1 + GAP - 0.55, ym))
    xc, xd = (CW1 + GAP) * 2 + CW1 / 2, (CW1 + GAP) * 3 + CW1 / 2
    yt, yb = Y1 - 0.35, Y1 - 1.7
    bg.plot([xc, xc, xd, xd], [yt, yb, yb, yt], color=S.INK2, lw=0.6, solid_joinstyle="miter", zorder=4)
    yr = YH - 1.4
    S.arrow(bg, ((xc + xd) / 2, yb), ((xc + xd) / 2, yr + 0.05))
    bg.plot([0.0, W], [yr, yr], color=S.GRID, lw=0.6, zorder=3)
    S.badge(bg, TX + 1.25, YH + 0.85, "4")
    S.run(bg, TX + 3.3, YH, [("Decision layer", dict(fontsize=S.FS["title"], fontweight="bold")),
                             ("on the selected units, built on the reference set (c) and the calibrated thresholds (d)",
                              dict(color=S.INK2))], gap=1.3)
    h, texts = S.strip(bg, 0.0, YF, W, [
        "every guarantee rests on the random split of one fixed map (a design-based argument): nothing about spatial "
        "dependence is modelled, so the guarantees hold",
        "under any spatial autocorrelation. Locality may enter through the score, a basis anchored on the training fold "
        "or random localization, but not through kernels centred",
        "at the test unit (GeoCP-style weights), which change with the unit being tested. Not covered: transfer to "
        "unmapped regions, preferential labelling, deterministic block splits."], head="Source of validity:")
    STRIPS.append((bg, (1.0, YF, W - 1.0, YF + h), texts))


def main():
    S.apply()
    D, B, summ = run_method(synthetic_map())
    fig = S.figure(W, H)
    bg = S.canvas(fig, 0, 0, W, H)
    A = panel_a(fig, 0.0, D)
    Bx = panel_b(fig, CW1 + GAP, D)
    C = panel_c(fig, 2 * (CW1 + GAP), D)
    Dx = panel_d(fig, 3 * (CW1 + GAP), D, B, summ)
    E = panel_e(fig, 0.0, D, summ)
    F = panel_f(fig, CW2 + GAP, D, summ)
    G = panel_g(fig, 2 * (CW2 + GAP), D, summ)
    flow(bg)
    bad = {ax.texts[0].get_text(): S.overflow(ax, (0.9, 0.3, ax.get_xlim()[1] - 0.9, ax.get_ylim()[1] - 0.3))
           for ax in (A, Bx, C, Dx, E, F, G)}
    bad["figure"] = S.overflow(bg, (0.0, 0.0, W, H))
    for ax, box, texts in STRIPS:
        bad.setdefault("strips", []).extend(S.overflow(ax, box, texts))
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise SystemExit(f"text outside its card or strip: {bad}")
    S.source_data("nc_fig2", D.drop(columns=["q_split"]))
    S.source_data("nc_fig2_summary", pd.DataFrame([summ]))
    S.save(fig, "nc_fig2_method", axes=[A, Bx, C, Dx, E, F, G], panel_ids=list("abcdefg"),
           row_groups=[["a", "b", "c", "d"], ["e", "f", "g"]], column_groups=[])
    print(pd.Series(summ).to_string())


if __name__ == "__main__":
    main()
