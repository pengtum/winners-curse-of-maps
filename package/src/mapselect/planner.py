"""Audit planner: what a rule's reference sets will hold, computed before any audited outcome is measured.

For a top- or bottom-K rule with distinct keys fixed given the map, under a random split of a finite population into
n audited units and m candidates, |R| = #{audited units ranked above the (K+1)-th candidate} is negative
hypergeometric (M = n + m, successes n, r = K + 1), so with tau = 1 - alpha:

  finite interval    Pr(finite) = Pr(|R| >= ceil(tau / alpha))
  moment             E[1 / (|R| + 1)] = m (1 - eps) / {K (n + 1)},  eps = C(n + m - K, m - K - 1) / C(n + m, m - 1)
  floor (Theorem 1)  floor(u) = tau - l1 m (1 - eps) / {(n + 1) M_u},  M_u = E[sum over selected of f(X_j)]
                     (M_u = K_u, the expected selected units of region u, for a group indicator)
  exact total bound  Pr(N_0 >= 1) = K / l, so C_0 = 0 (no outcome bound enters) needs K / (n_R + K) <= delta

A targeted audit draws the n audited units at random from a stratum fixed given the map (for example the highest-
predicted 5% of the population). Every statement above then holds with m = the stratum's candidates, because
the stratum is itself a finite population under a random split, and a top-K rule selects inside it when it holds at
least K candidates. Formulas are checked against full enumeration in the project's tests.
"""
from math import ceil, exp, lgamma
import numpy as np
from scipy.stats import nhypergeom


def _lchoose(a, b):
    return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)


def _thr(alpha):
    return int(ceil((1 - alpha) / alpha - 1e-12))                       # JOMI is finite iff |R| >= tau / alpha


def p_finite(n, m, K, alpha=0.1):
    """Pr(a top-K rule's selected units get finite (1 - alpha) intervals). Requires K < m."""
    if n < _thr(alpha):
        return 0.0
    return float(nhypergeom(n + m, n, K + 1).sf(_thr(alpha) - 1))


def mean_refset(n, m, K):
    """E|R| = n (K + 1) / (m + 1)."""
    return n * (K + 1) / (m + 1)


def inv_moment(n, m, K):
    """E[1 / (|R| + 1)], exact."""
    if m - K - 1 < 0:
        raise ValueError("needs K < m")
    eps = exp(_lchoose(n + m - K, m - K - 1) - _lchoose(n + m, m - 1))   # log-gamma, so that pools of 10^6 stay fast
    return m * (1 - eps) / (K * (n + 1))


def floor(n, m, K, M_u, alpha=0.1, l1=0.95):
    """Theorem 1's coverage floor for a basis function f >= 0 with expected selected mass M_u; <= 0 means no guarantee."""
    return 1 - alpha - l1 * inv_moment(n, m, K) * K / M_u


def n_for_finite(m, K, prob=0.9, alpha=0.1, nmax=10 ** 7):
    """Smallest audit of a fixed candidate pool m with Pr(finite) >= prob (bisection; p_finite increases in n)."""
    lo, hi = _thr(alpha) - 1, _thr(alpha)
    while p_finite(hi, m, K, alpha) < prob:
        lo, hi = hi, hi * 2
        if hi > nmax:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if p_finite(mid, m, K, alpha) >= prob else (mid, hi)
    return hi


def n_for_floor(m, K, M_u, target, alpha=0.1, l1=0.95, nmax=10 ** 7):
    """Smallest audit with floor(u) >= target for a fixed candidate pool m; None if unreachable below nmax."""
    if target >= 1 - alpha:
        return None
    lo, hi = 0, 1
    while floor(hi, m, K, M_u, alpha, l1) < target:
        lo, hi = hi, hi * 2
        if hi > nmax:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if floor(mid, m, K, M_u, alpha, l1) >= target else (mid, hi)
    return hi


def n_budget_for_finite(population, K, prob=0.9, alpha=0.1):
    """Smallest audit when audited units leave the pool (m = population - n), with K kept fixed.
    p_finite rises with n here too, because n rises and m falls. None if no audit leaves K + 1 candidates and reaches prob."""
    hi_max = population - K - 1
    if hi_max < _thr(alpha) or p_finite(hi_max, population - hi_max, K, alpha) < prob:
        return None
    lo, hi = _thr(alpha) - 1, hi_max
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if p_finite(mid, population - mid, K, alpha) >= prob else (mid, hi)
    return hi


def planned_se_ratio(n, m, K):
    """Planned selection-aware SE of a project total over the SE of K independent errors, (1 + K / E|R|)^(1/2)."""
    return float(np.sqrt(1 + K / max(mean_refset(n, m, K), 1e-12)))
