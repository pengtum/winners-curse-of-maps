"""Certified totals over a top/bottom-K selected set (the manuscript's Lemma 1 and Theorem 2(a)).

Given the pool P = selected u reference set (Lemma P: the selected set is a uniform K-subset of
P), the bucket counts N_b = #{selected values below the (b+1)-th smallest reference value} have a
value-free law. `envelope_exact` returns a nondecreasing integer envelope C with
P(exists b: N_b > C_b | P) <= delta computed EXACTLY by lattice-path DP (no Monte Carlo error) and certified in
integer arithmetic before it is returned;
`lower_total` turns it into a lower bound for the selected total of v (residuals or any fixed
transform), given an explicit support floor for bucket-0 values.
"""
from fractions import Fraction
from itertools import accumulate
from math import comb, lgamma
import numpy as np


def good_paths(ell, K, C):
    """Integer count of the K-subsets of ell slots with N_b <= C_b for all b; len(C) = ell - K + 1. One row at a time."""
    nR = ell - K
    row = [1] * (K + 1)                                                 # row 0: all K marks can come before the first unmarked slot
    for i in range(nR):
        c = int(C[i])
        row = list(accumulate(v if s <= c else 0 for s, v in enumerate(row)))
    return row[K]


def valid_prob(ell, K, C):
    """P(N_b <= C_b for all b) for S a uniform K-subset of ell slots; len(C) = ell - K + 1."""
    return good_paths(ell, K, C) / comb(ell, K)


def violation_exact(ell, K, C):
    """P(exists b: N_b > C_b) as an exact rational."""
    tot = comb(ell, K)
    return Fraction(tot - good_paths(ell, K, C), tot)


def valid_prob_fast(ell, K, C):
    """valid_prob in floating point: row i+1 of the lattice-path counts is the cumulative sum of row i
    masked to s <= C_i; rows are rescaled to avoid overflow. Agrees with valid_prob to ~1e-12."""
    nR = ell - K
    s = np.arange(K + 1); f = np.ones(K + 1); logc = 0.0
    for i in range(nR):
        f = np.cumsum(np.where(s <= C[i], f, 0.0))
        if f[-1] <= 0:
            return 0.0
        logc += np.log(f[-1]); f /= f[-1]
    return float(np.exp(logc + np.log(f[K]) - (lgamma(ell + 1) - lgamma(K + 1) - lgamma(nR + 1))))


def _moments(ell, K):
    """Exact mean and sd of N_b (number of K marked slots before the (b+1)-th unmarked one)."""
    nR = ell - K; b = np.arange(nR)
    # N_b ~ negative-hypergeometric: E = K (b+1) / (nR+1), Var = K (b+1)(nR-b)(ell+1) / ((nR+1)^2 (nR+2))
    mean = K * (b + 1) / (nR + 1)
    var = K * (b + 1) * (nR - b) * (ell + 1) / ((nR + 1) ** 2 * (nR + 2))
    return mean, np.sqrt(np.maximum(var, 1e-12))


def envelope_exact(ell, K, delta, iters=40, fast=False):
    """Shape floor(mean + lam sd) (value-free), lam by bisection on the EXACT violation probability.
    fewer iterations only make it slightly more conservative; fast=True uses valid_prob_fast (floating point, same
    law). The returned envelope is certified in integer arithmetic, violation <= Fraction(repr(delta)) (the shortest
    decimal of the float: exactly 1/20 and 3/10, and 0.3333333333333333 <= 1/3); if the certificate fails, the trivial
    envelope C = K (violation 0) is returned."""
    vp = valid_prob_fast if fast else valid_prob
    nR = ell - K
    if nR == 0:
        return np.array([K])
    mu, sd = _moments(ell, K)

    def C_of(lam):
        C = np.maximum.accumulate(np.minimum(K, np.floor(mu + lam * sd))).astype(int)
        return list(C) + [K]
    lo, hi = 0.0, 12.0
    for _ in range(iters):
        lam = (lo + hi) / 2
        if 1 - vp(ell, K, C_of(lam)) > delta:
            lo = lam
        else:
            hi = lam
    C = C_of(hi)
    if violation_exact(ell, K, C) > Fraction(repr(float(delta))):
        C = [K] * (nR + 1)
    return np.array(C)


def lower_total(r_obs, K, floor, C, weights=None):
    """sum_S v >= sum_b (C_b - C_{b-1}) r_(b) on the envelope event (r_(0) = min(floor, r_(1)));
    with weights (sigma_j > 0) the rearrangement bound for sum_S sigma_j v_j."""
    r_obs = np.sort(np.asarray(r_obs, float))
    if len(r_obs):
        floor = min(floor, r_obs[0])
    r = np.r_[floor, r_obs]
    c = np.diff(np.r_[0, C]).astype(int)
    vals = np.repeat(r, c)[:K]
    if weights is None:
        return float(vals.sum())
    return float((np.sort(vals) * np.sort(np.asarray(weights, float))[::-1]).sum())
