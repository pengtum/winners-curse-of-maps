"""III(g): Pi not identifiable -> worst case over a box of exchange weights.

If the true weights satisfy w_i in [w_hat_i / Gamma, w_hat_i * Gamma] for all units, the
worst-case weighted quantile (Jin, Ren & Candes 2023 closed form: lower weights on small
scores, upper weights on large scores and on the test atom) is valid (Theorem I with the
worst Pi in the box). Gamma = 1 recovers the plug-in estimated-weight method.
"""
import numpy as np


def gamma_box_quantile(S_cal, w_hat_cal, w_hat_test, Gamma, alpha):
    """Threshold_j = smallest S_(k) with  sum_{i<=k} l_i / (sum_{i<=k} l_i + sum_{i>k} u_i + u_j) >= 1-alpha,
    l = w_hat / Gamma, u = w_hat * Gamma (scores sorted ascending). +inf if unreachable."""
    S = np.asarray(S_cal, float); o = np.argsort(S); Ss = S[o]
    wc = np.asarray(w_hat_cal, float)[o]
    l, u = wc / Gamma, wc * Gamma
    Lk = np.cumsum(l)                                   # sum_{i<=k} l_i
    Uk = np.r_[np.cumsum(u[::-1])[::-1][1:], 0.0]       # sum_{i>k} u_i
    lhs = alpha * Lk - (1 - alpha) * Uk                 # increasing in k
    rhs = (1 - alpha) * np.asarray(w_hat_test, float) * Gamma
    idx = np.searchsorted(lhs, rhs - 1e-12 * np.abs(rhs), side="left")
    q = np.full(len(rhs), np.inf); ok = idx < len(Ss)
    q[ok] = Ss[idx[ok]]
    return q


def gamma_value(S_cal, w_hat_cal, w_hat_test, alpha, claim, grid=(1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16)):
    """Gamma-value map: for each test unit, the largest Gamma on the grid for which `claim(q)`
    (a boolean function of the robust thresholds, e.g. 'interval excludes c') still holds;
    0 if it fails already at Gamma = 1."""
    out = np.zeros(len(w_hat_test))
    alive = np.ones(len(w_hat_test), bool)            # claim held for every smaller Gamma
    for G in grid:
        alive &= np.asarray(claim(gamma_box_quantile(S_cal, w_hat_cal, w_hat_test, G, alpha)), bool)
        out[alive] = G
    return out
