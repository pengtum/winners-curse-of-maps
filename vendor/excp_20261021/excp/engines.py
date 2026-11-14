"""Engines E: turn (calibration scores, exchange weights, localizer) into per-test thresholds.

quantile     constant function class: Pi-weighted step quantile with the test atom (Thm I);
             `level` overrides 1-alpha (GeoCP v1's legacy ceil((1-a)(n+1))/n without atom)
fclass       function-class engine = design-weighted GeoCCP (III(d)); optional L1 shrinkage
local_qr     GWQR / local-linear quantile regression at the test point (no guarantee)
crc          conformal risk control with Pi weights (III(e))
bayes_layer  WBCP-style posterior layer: threshold = max(point threshold, beta-credible one)
union_max    GROUND-style aggregation: max over several thresholds (union of sets)
"""
import numpy as np
from scipy.optimize import linprog


def quantile(S_cal, w_cal, w_test, alpha, loc=None, level=None, chunk=3000):
    """Threshold_j = inf{s: F_j(s) >= level}, F_j the distribution putting mass
    w_i * fac_ij on S_i and w_j * atom_j on +inf (normalised). +inf if unreachable."""
    S_cal = np.asarray(S_cal, float); w_cal = np.asarray(w_cal, float); w_test = np.asarray(w_test, float)
    lev = 1 - alpha if level is None else level
    o = np.argsort(S_cal); Ss = S_cal[o]; wc = w_cal[o]
    m = len(w_test); q = np.empty(m)
    if loc is None:
        cs = np.cumsum(wc)
        tot = cs[-1] + w_test
        idx = np.searchsorted(cs, lev * tot - 1e-9 * tot, side="left")
        q = np.full(m, np.inf); ok = idx < len(Ss)
        q[ok] = Ss[idx[ok]]
        return q
    for a in range(0, m, chunk):
        sl = slice(a, min(m, a + chunk))
        Wm = loc.fac(sl)[:, o] * wc[None, :]
        cs = np.cumsum(Wm, axis=1)
        tot = cs[:, -1] + w_test[sl] * loc.atom(sl)
        idx = (cs < (lev * tot)[:, None] - 1e-9 * tot[:, None]).sum(1)
        qq = np.full(len(idx), np.inf)
        ok = (idx < len(Ss)) & (tot > 0)
        qq[ok] = Ss[idx[ok]]
        q[sl] = qq
    return q


def fclass(S_cal, w_cal, Phi_cal, w_test, Phi_test, alpha, l1=0.0, penalize=None):
    """Design-weighted augmented quantile regression with the test score imputed at +inf:
        min_beta sum_i w_i rho_tau(S_i - Phi_i beta) - tau w_j Phi_j beta + l1 * sum_{k pen} |beta_k|
    threshold_j = Phi_j beta_j (+inf if unbounded). l1 = 0 gives the exact L2 guarantee of
    III(d); l1 > 0 keeps it for the unpenalized directions and adds a slack <= l1 / (sum w)
    for each penalized direction (GeoCP_framework.md III(d'))."""
    S_cal = np.asarray(S_cal, float); w_cal = np.asarray(w_cal, float)
    tau = 1 - alpha
    n, p = Phi_cal.shape
    pen = np.zeros(p, bool) if penalize is None else np.asarray(penalize, bool)
    k = int(pen.sum()) if l1 > 0 else 0
    # variables: beta (p, free) | t (k >= 0, |beta_pen| <= t) | u (n) | v (n)
    A_eq = np.hstack([Phi_cal, np.zeros((n, k)), np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (k + 2 * n)
    A_ub = b_ub = None
    if k:
        P = np.zeros((k, p)); P[np.arange(k), np.flatnonzero(pen)] = 1
        Z = np.zeros((k, 2 * n))
        A_ub = np.vstack([np.hstack([P, -np.eye(k), Z]), np.hstack([-P, -np.eye(k), Z])])
        b_ub = np.zeros(2 * k)
    c0 = np.r_[np.zeros(p), np.full(k, l1), tau * w_cal, (1 - tau) * w_cal]
    out = np.empty(len(w_test))
    for j in range(len(w_test)):
        c = c0.copy(); c[:p] = -tau * w_test[j] * Phi_test[j]
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=S_cal, bounds=bounds, method="highs")
        out[j] = Phi_test[j] @ r.x[:p] if r.status == 0 else np.inf
    return out


def local_qr(S_cal, s_cal, X_cal, s_test, X_test, alpha, k=50):
    """GWQR-CP-type engine: for each test point, a Gaussian-kernel-weighted linear quantile
    regression of the scores on the features over its k nearest calibration neighbours
    (bandwidth = k-th neighbour distance); no test atom, no calibration step -> no guarantee
    (family_map: GWQR-CP). Leakage-free version (the original 'optimized' mode used test labels)."""
    from scipy.spatial import cKDTree
    tau = 1 - alpha
    tree = cKDTree(np.asarray(s_cal, float))
    dist, idx = tree.query(np.asarray(s_test, float), k=k)
    out = np.empty(len(s_test))
    for j in range(len(s_test)):
        ii = idx[j]; h = max(dist[j, -1], 1e-12)
        w = np.exp(-0.5 * (dist[j] / h) ** 2)
        Z = np.c_[np.ones(k), (X_cal[ii] - X_cal[ii].mean(0)) / (X_cal[ii].std(0) + 1e-9)]
        z0 = np.r_[1.0, (X_test[j] - X_cal[ii].mean(0)) / (X_cal[ii].std(0) + 1e-9)]
        p = Z.shape[1]
        A_eq = np.hstack([Z, np.eye(k), -np.eye(k)])
        c = np.r_[np.zeros(p), tau * w, (1 - tau) * w]
        r = linprog(c, A_eq=A_eq, b_eq=S_cal[ii], bounds=[(None, None)] * p + [(0, None)] * 2 * k,
                    method="highs")
        out[j] = max(z0 @ r.x[:p], 0.0) if r.status == 0 else np.inf
    return out


def crc(L_cal, w_cal, w_test, alpha, lam_grid, B=1.0):
    """Pi-weighted conformal risk control (III(e)). L_cal (n, G) nonincreasing in lambda (grid
    ascending). lambda_j = min{lam: (sum_i w_i L_i(lam) + w_j B) / (W + w_j) <= alpha}."""
    w_cal = np.asarray(w_cal, float); W = w_cal.sum()
    risk = (w_cal[:, None] * np.asarray(L_cal, float)).sum(0)
    out = np.empty(len(w_test))
    for j, wj in enumerate(np.asarray(w_test, float)):
        ok = np.flatnonzero((risk + wj * B) / (W + wj) <= alpha)
        out[j] = lam_grid[ok[0]] if len(ok) else np.inf
    return out


def bayes_layer(S_cal, w_cal, w_test, alpha, beta=0.9, loc=None, draws=100, rng=None, chunk=500):
    """WBCP-style data-conditional layer (D axis): scaled-Dirichlet posterior over the
    calibration distribution, tilde V_i propto w_i E_i (E ~ Exp(1)) plus the trailing test atom;
    returns max(point threshold, beta-quantile of the posterior thresholds). The max keeps the
    marginal guarantee of the point threshold (it can only enlarge the set)."""
    rng = rng or np.random.default_rng(0)
    q0 = quantile(S_cal, w_cal, w_test, alpha, loc)
    S_cal = np.asarray(S_cal, float); o = np.argsort(S_cal); Ss = S_cal[o]
    wc = np.asarray(w_cal, float)[o]; m = len(w_test); qb = np.empty(m)
    for a in range(0, m, chunk):
        sl = slice(a, min(m, a + chunk)); mm = sl.stop - sl.start
        F = (loc.fac(sl)[:, o] if loc is not None else np.ones((mm, len(Ss)))) * wc[None, :]
        at = np.asarray(w_test, float)[sl] * (loc.atom(sl) if loc is not None else 1.0)
        E = rng.exponential(size=(draws, mm, len(Ss))); Ea = rng.exponential(size=(draws, mm))
        cs = np.cumsum(F[None] * E, axis=2)
        tot = cs[:, :, -1] + at[None] * Ea
        idx = (cs < ((1 - alpha) * tot)[:, :, None]).sum(2)
        Q = np.where(idx < len(Ss), Ss[np.minimum(idx, len(Ss) - 1)], np.inf)
        qb[sl] = np.quantile(Q, beta, axis=0)
    return np.maximum(q0, qb)


def union_max(*thresholds):
    """GROUND / MDCP-style aggregation: the union of valid sets is valid (conservative)."""
    return np.max(np.vstack(thresholds), axis=0)
