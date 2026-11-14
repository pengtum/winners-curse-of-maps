"""Sampling designs on a finite population, their exchange weights, design inheritance under
re-splitting (M1-T2), and the label-budget law (theory/M1_budget_law.md)."""
import numpy as np


def srswor(N, n, rng):
    return rng.choice(N, n, replace=False)


def poisson(pi, rng):
    return np.flatnonzero(rng.random(len(pi)) < pi)


def cps(ptil, n, rng):
    """Conditional Poisson (rejective) with working probabilities ptil and size n."""
    while True:
        Z = rng.random(len(ptil)) < ptil
        if Z.sum() == n:
            return np.flatnonzero(Z)


def cps_inclusion(ptil, n):
    """Exact first-order inclusion probabilities of CPS(ptil, n) by prefix/suffix elementary
    symmetric polynomials (rescaled)."""
    r = ptil / (1 - ptil); N = len(r); r = r * n / (np.e * r.sum())
    pre = np.zeros((N + 1, n + 1)); pre[0, 0] = 1
    for i in range(N):
        pre[i + 1] = pre[i]; pre[i + 1, 1:] += r[i] * pre[i, :-1]
    suf = np.zeros((N + 2, n + 1)); suf[N + 1, 0] = suf[N, 0] = 1
    for i in range(N - 1, -1, -1):
        suf[i] = suf[i + 1]; suf[i, 1:] += r[i] * suf[i + 1, :-1]
    en = pre[N, n]
    return np.array([r[k] * (pre[k, :n] @ suf[k + 1, :n][::-1]) / en for k in range(N)])


def stratified(strata, alloc, rng):
    strata = np.asarray(strata)
    return np.concatenate([rng.choice(np.flatnonzero(strata == h), alloc[h], replace=False)
                           for h in np.unique(strata)])


def split(L, rng, frac=0.5, strata=None, clusters=None):
    """Training / calibration split that INHERITS the design (M1-T2): SRS of fixed size inside
    L, inside each stratum, or by whole clusters. Returns (D_train, C_cal)."""
    L = np.asarray(L)
    if clusters is not None:
        ub = np.unique(clusters[L]); tb = rng.choice(ub, int(round(frac * len(ub))), replace=False)
        tr = np.isin(clusters[L], tb); return L[tr], L[~tr]
    if strata is not None:
        tr = np.zeros(len(L), bool)
        for h in np.unique(strata[L]):
            idx = np.flatnonzero(strata[L] == h)
            tr[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
        return L[tr], L[~tr]
    perm = rng.permutation(len(L)); k = int(round(frac * len(L)))
    return L[perm[:k]], L[perm[k:]]


# ------------------------------------------------------------------ label budget law
def coverage_tail(ns, m, alpha, eps):
    """Exact P(realized coverage over m unsampled units < 1 - alpha - eps) under SRSWOR
    calibration of size n (pool n + m); m=None gives the Beta limit."""
    from scipy.stats import hypergeom, beta
    ns = np.atleast_1d(ns); t = 1 - alpha - eps
    k = np.ceil((ns + 1) * (1 - alpha)).astype(int); ok = k <= ns
    out = np.zeros(len(ns))
    if m is None:
        out[ok] = beta.cdf(t, k[ok], ns[ok] + 1 - k[ok]); return out
    cmax = int(np.ceil(m * t - 1e-9)) - 1
    if cmax < 0:
        return out
    draws = np.minimum(k + cmax, ns + m)
    out[ok] = 1 - hypergeom.cdf(k[ok] - 1, ns[ok] + m, ns[ok], draws[ok])
    return out


def min_labels(alpha, eps, delta, m, nmax=20000):
    """Smallest n with P(coverage < 1 - alpha - eps) <= delta for every n' >= n (n' <= nmax)."""
    ns = np.arange(1, nmax + 1)
    bad = np.flatnonzero(coverage_tail(ns, m, alpha, eps) > delta)
    if len(bad) == 0:
        return 1
    return np.nan if bad[-1] + 1 >= nmax else int(ns[bad[-1]] + 1)
