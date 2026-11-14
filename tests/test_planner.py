"""Checks for excp_select/planner.py.

    ~/miniforge3/envs/favela/bin/python tests/test_planner.py

1. E|R|, Pr(finite) and E[1 / (|R| + 1)] against full enumeration of all n-subsets of a small population, as exact rationals.
2. The same three on a targeted audit: enumerate audits inside a stratum and apply the rule to all candidates,
   checking that the units outside the stratum never enter the top K and that the stratum's law applies.
3. The n_for_* searches return the smallest audit that meets the target.
"""
import os, sys
from fractions import Fraction
from itertools import combinations
from math import comb, ceil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True
from excp_select import planner as P                                    # noqa: E402


def enumerate_R(keys, n, K, pool=None):
    """Law of |R| over all audits of size n drawn from `pool` (positions into keys); the rest are candidates."""
    pool = list(range(len(keys))) if pool is None else pool
    law = {}
    for aud in combinations(pool, n):
        a = set(aud)
        cand = sorted((keys[i] for i in range(len(keys)) if i not in a), reverse=True)
        thr = cand[K]
        r = sum(keys[i] > thr for i in aud)
        top = cand[:K]
        assert not pool or min(top) >= min(keys[i] for i in pool) or len(pool) == len(keys)
        law[r] = law.get(r, 0) + 1
    tot = sum(law.values())
    return {r: Fraction(c, tot) for r, c in law.items()}


def check(law, n, m, K, alpha=Fraction(1, 2)):
    er = sum(r * p for r, p in law.items())
    assert er == Fraction(n * (K + 1), m + 1), (er, n, m, K)
    assert abs(float(er) - P.mean_refset(n, m, K)) < 1e-12
    inv = sum(p / (r + 1) for r, p in law.items())
    eps = Fraction(comb(n + m - K, m - K - 1), comb(n + m, m - 1))
    assert inv == Fraction(m, K * (n + 1)) * (1 - eps), (inv, n, m, K)
    assert abs(float(inv) - P.inv_moment(n, m, K)) < 1e-12
    thr = ceil((1 - alpha) / alpha)
    pf = sum(p for r, p in law.items() if r >= thr)
    assert abs(float(pf) - P.p_finite(n, m, K, float(alpha))) < 1e-10, (pf, n, m, K)


def main():
    c = 0
    for N, n, K in [(8, 3, 1), (10, 4, 2), (12, 5, 3), (12, 6, 1), (13, 4, 5), (14, 7, 2)]:
        keys = list(range(N))
        for alpha in (Fraction(1, 2), Fraction(1, 3), Fraction(1, 5)):
            check(enumerate_R(keys, n, K), n, N - n, K, alpha); c += 1
    print(f"1. whole-population enumeration: {c} cases exact")
    c = 0
    for N, Ms, n, K in [(20, 8, 3, 1), (20, 10, 4, 2), (30, 12, 5, 3), (25, 13, 4, 5)]:
        keys = list(range(N)); stratum = keys[-Ms:]                     # the Ms highest keys
        for alpha in (Fraction(1, 2), Fraction(1, 3)):
            check(enumerate_R(keys, n, K, stratum), n, Ms - n, K, alpha); c += 1
    print(f"2. targeted audit, law with m = stratum candidates: {c} cases exact")
    for m, K, prob in [(3000, 30, 0.9), (3000, 150, 0.5), (500, 5, 0.9)]:
        a = P.n_for_finite(m, K, prob)
        assert P.p_finite(a, m, K) >= prob > P.p_finite(a - 1, m, K)
    for m, K, Mu, t in [(3000, 150, 2, 0.8), (3000, 148, 3, 0.7)]:
        a = P.n_for_floor(m, K, Mu, t)
        assert P.floor(a, m, K, Mu) >= t > P.floor(a - 1, m, K, Mu)
    for pop, K in [(6000, 30), (300, 30)]:
        a = P.n_budget_for_finite(pop, K)
        assert P.p_finite(a, pop - a, K) >= 0.9 > P.p_finite(a - 1, pop - a + 1, K)
    print("3. searches return the smallest audit")


if __name__ == "__main__":
    main()
