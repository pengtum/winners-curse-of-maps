"""Selection-conditional presets with the excp.family Method interface.

ctx is the usual excp ctx over the FULL test set (S_cal, w_cal, w_test, s_cal, s_test,
s_train, w_train, g_cal, g_test, rng); every method returns thresholds for the selected units
only, in the order of sel.selected (interval = {y : S(x_j, y) <= q_j}).

  jomi(sel)               JOMI: constant engine on the reference set (Thm 1 of Jin & Ren)
  jomi(sel, by_group=1)   + Mondrian on the test unit's group: coverage given (selected, state)
  jomi_excp(sel, kappa)   JOMI x RLCP+ (design-weighted kNN bandwidth); theory/SC_RLCP.md
  sc_geoccp(sel, l1, pen)  JOMI x conditional calibration on a multiscale spatial basis
                          (ctx['Phi_cal'], ctx['Phi_test']); theory/SC_GeoCCP.md
  by(sel, m)              Benjamini-Yekutieli-adjusted split CP at level alpha |S| / m
  on_selected(method, sel) any excp.family method evaluated on the selected units (baselines)
"""
import numpy as np
from excp import engines as E, localizers as Lz
from excp.family import Method

TEST_KEYS = ("w_test", "s_test", "x_test", "g_test", "gamma_test", "Phi_test")


def subset_ctx(ctx, idx):
    return {k: (np.asarray(v)[idx] if k in TEST_KEYS and v is not None else v) for k, v in ctx.items()}


def _loc(sel, c, by_group):
    loc = sel.localizer()
    if by_group:
        loc = Lz.compose(loc, Lz.mondrian(c["g_cal"], c["g_test"]))
    return loc


def jomi(sel, by_group=False):
    def fn(ctx, a):
        c = subset_ctx(ctx, sel.selected)
        return E.quantile(c["S_cal"], c["w_cal"], c["w_test"], a, _loc(sel, c, by_group))
    return Method("JOMI" + ("|group" if by_group else ""), fn,
                  "coverage given selection" + (" and group" if by_group else ""), "Jin & Ren Thm 1")


def jomi_excp(sel, kappa, by_group=False):
    def fn(ctx, a):
        c = subset_ctx(ctx, sel.selected)
        loc = _loc(sel, c, by_group)
        if kappa is not None:
            loc = Lz.compose(loc, Lz.rlcp_plus(c["s_cal"], c["s_test"], c["s_train"], c["w_cal"],
                                               c["w_test"], c["w_train"], kappa, c["rng"]))
        return E.quantile(c["S_cal"], c["w_cal"], c["w_test"], a, loc)
    return Method("JOMI-ExCP (RLCP+)", fn, "coverage given selection, RLCP+-tilted",
                  "SC-RLCP lemma (theory/SC_RLCP.md)")


def sc_geoccp(sel, l1, penalize):
    """SC-GeoCCP: per selected unit j, Gibbs-Cherian-Candes augmented quantile regression over its
    reference set R_j plus the test point (score imputed at +inf), basis ctx['Phi_cal'] /
    ctx['Phi_test'] (e.g. basis.Multiscale), L1 penalty l1 on the `penalize` directions.
    Guarantee (theory/SC_GeoCCP.md Thm 1, dual-certificate proof): for every f = Phi c >= 0,
    E[f(X_j)(1{Y_j in C_j} - (1-a)) | j selected] >= -l1 ||c_pen||_1 E[1/|R_j u {j}|], for ANY
    optimal vertex the LP solver returns. With l1 >= 1-a the threshold is +inf exactly when JOMI's
    is (Prop. 2)."""
    def fn(ctx, a):
        c = subset_ctx(ctx, sel.selected)
        S, Pc, Pt = np.asarray(c["S_cal"]), np.asarray(c["Phi_cal"]), np.asarray(c["Phi_test"])
        q = np.empty(len(sel))
        for r in range(len(sel)):
            m = sel.member[r]
            # Numerical safeguard (2026-11-08): when every reference score is below 1e-6 (outcome and prediction both 0,
            # so the scores are only the 1e-9 tie-breaking jitter), the LP solver's tolerance (~1e-7) exceeds the scale of
            # the data. The programme is positively homogeneous in S (scores / c -> minimizer / c), so it is solved on
            # S / max(S) and the threshold multiplied back: the same minimizer up to that factor. Other reference sets
            # take the original path unchanged. Only FIA, bottom 1%, met this case (paper/archive_2026-11-08/).
            Sm = S[m]; sc = float(Sm.max()) if Sm.size else 0.0
            sc = sc if (0.0 < sc < 1e-6) else 1.0
            q[r] = sc * E.fclass(Sm / sc, np.ones(m.sum()), Pc[m], np.ones(1), Pt[r:r + 1], a,
                                 l1=l1, penalize=penalize)[0]
        return q
    return Method(f"SC-GeoCCP(l1={l1:g})", fn, "selection-conditional function-class coverage (L1 slack)",
                  "SC-GeoCCP Thm 1 (theory/SC_GeoCCP.md)")


def by(sel, m):
    def fn(ctx, a):
        c = subset_ctx(ctx, sel.selected)
        a_adj = a * max(len(sel), 1) / m
        return E.quantile(c["S_cal"], np.ones(len(c["S_cal"])), np.ones(len(sel)), a_adj)
    return Method("BY split CP", fn, "FCR heuristic (no selection-conditional guarantee)", "BY 2005")


def on_selected(method, sel):
    def fn(ctx, a):
        return method.thresholds(subset_ctx(ctx, sel.selected), a)
    return Method(method.name, fn, method.guarantee + " (marginal; none given selection)", method.theorem)


def upper_bound(sel, alpha):
    """Theorem 1 upper bound 1 - alpha + E[1 / (1 + |R|)] (unweighted JOMI, no ties):
    returns the per-unit term so callers can average over units and splits."""
    return 1 - alpha + 1.0 / (1.0 + sel.sizes)
