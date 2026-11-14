"""Checks for theory/POOL_LEMMA.md.

    ~/miniforge3/envs/favela/bin/python tests/test_pool_lemma.py [reps]

1. Lemma P: given the pool P (prefix above the (K+1)-th test unit), the selected set is uniform
   over the K-subsets of P (chi-square over subsets for the most frequent pools, small population).
2. The exact envelope (excp_select.totals.envelope_exact, on its default integer path and on the fast path behind
   experiments/pilot_geocert.py::envelope since 2026-10-28): its violation probability, counted here with this file's
   own lattice-path DP (not the module's certificate) as an exact rational, must be <= delta with no tolerance, and
   the envelope must not be the trivial fall-back C = K. The
   retired Monte Carlo envelope (shape and lambda fitted on the same 10,000 draws) is printed for comparison only;
   its true violation can exceed delta (0.0508 at ell = 120, K = 30).
3. T1(b): the exact-envelope total lower bound covers on the fixed autocorrelated population.
4. T2: cfBH on the reference set controls FDR on the same population (bottom-K aid-type test).
5. excp_select.totals.valid_prob_fast (floating point) vs the integer DP, and the fast envelope's exact
   violation probability (used by experiments/run_carbon.py).
"""
import os, sys
from fractions import Fraction
from math import comb
import numpy as np
from scipy.stats import chisquare
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.dont_write_bytecode = True
from excp_select import refsets                                    # noqa: E402
from test_coverage_sim import population                          # noqa: E402
from pilot_geocert import envelope, lower_total                    # noqa: E402
from excp_select import totals as TOT                              # noqa: E402


def exact_violation(ell, K, C):
    """P(exists b: N_b > C_b) as an exact rational, from this file's own DP (independent of the module certificate)."""
    nR = ell - K
    f = [[0] * (K + 1) for _ in range(nR + 1)]
    f[0][0] = 1
    for i in range(nR + 1):
        for s in range(K + 1):
            v = f[i][s]
            if v and s < K:
                f[i][s + 1] += v
            if v and i < nR and s <= C[i]:
                f[i + 1][s] += v
    return Fraction(comb(ell, K) - f[nR][K], comb(ell, K))


def nontrivial(C, K):
    return any(int(c) < K for c in list(C)[:-1])


def exact_valid_prob(ell, K, C):
    """P(N_b <= C_b for all b) for S a uniform K-subset of ell slots; C has length nR + 1."""
    nR = ell - K
    f = [[0] * (K + 1) for _ in range(nR + 1)]
    f[0][0] = 1
    for i in range(nR + 1):
        for s in range(K + 1):
            v = f[i][s]
            if not v:
                continue
            if s < K:
                f[i][s + 1] += v                      # next slot is S
            if i < nR and s <= C[i]:
                f[i + 1][s] += v                      # (i+1)-th R slot; N_i = s must be <= C_i
    return f[nR][K] / comb(ell, K)


def check_lemma(R=60000):
    rng = np.random.default_rng(1)
    n, m, K = 18, 12, 3; N = n + m
    keys = rng.normal(size=N); order = np.argsort(-keys)              # u_(1..N)
    from collections import Counter, defaultdict
    by_pool = defaultdict(Counter)
    for _ in range(R):
        T = rng.choice(N, m, replace=False); inT = np.zeros(N, bool); inT[T] = True
        flags = inT[order]; pos = np.flatnonzero(flags)
        ell = pos[K] if m > K else N                                  # slots before the (K+1)-th test unit
        S = tuple(sorted(order[pos[:K]]))
        by_pool[ell][S] += 1
    out = []
    for ell, cnt in sorted(by_pool.items(), key=lambda kv: -sum(kv[1].values()))[:3]:
        nsub = comb(ell, K); obs = np.array(list(cnt.values()) + [0] * (nsub - len(cnt)))
        p = chisquare(obs).pvalue
        out.append((ell, int(obs.sum()), nsub, p))
    return out


def mc_envelope(L, K, delta, rng, n_mc=10000):
    """The retired Monte Carlo envelope (pilot_geocert.py before 2026-10-28), for comparison only."""
    nR = L - K
    lab = np.argsort(rng.random((n_mc, L)), axis=1) < K
    cumS = np.cumsum(lab, axis=1)
    Rpos = np.argsort(lab, axis=1, kind="stable")[:, :nR]
    N = np.take_along_axis(cumS, Rpos, axis=1)
    mu_b, sd_b = N.mean(0), N.std(0) + 1e-9
    lo, hi = 0.0, 10.0
    for _ in range(40):
        lam = (lo + hi) / 2
        viol = (N > np.minimum(K, np.floor(mu_b + lam * sd_b))[None, :]).any(1).mean()
        lo, hi = (lam, hi) if viol > delta else (lo, lam)
    return np.r_[np.maximum.accumulate(np.minimum(K, np.floor(mu_b + hi * sd_b))), K]


def check_envelope(delta=0.05):
    rng = np.random.default_rng(3); res = []
    for ell, K in [(40, 10), (120, 30), (300, 150), (60, 30)]:
        Cmc = mc_envelope(ell, K, delta, rng)
        for path, C in (("fast", envelope(ell, K, delta)), ("integer", TOT.envelope_exact(ell, K, delta))):
            C = [int(c) for c in C]
            res.append((ell, K, path, 1 - exact_valid_prob(ell, K, [int(c) for c in Cmc]), exact_violation(ell, K, C),
                        nontrivial(C, K)))
    return res


def check_t1_t2(R=4000, q=0.10, delta=0.05):
    s, y, g = population()
    N = len(y); rng0 = np.random.default_rng(7)
    perm = rng0.permutation(N); tr, rest = perm[:1000], perm[1000:]
    tree = cKDTree(s[tr])
    mu = y[tr][tree.query(s, k=15)[1]].mean(1) + 1e-9 * rng0.standard_normal(N)
    sig = np.maximum(np.abs(y[tr] - mu[tr])[tree.query(s, k=40)[1]].mean(1), 0.05)
    c = np.quantile(y[tr], 0.25)                                       # eligibility line, fixed given train
    gsc = (mu - c) / sig; ymin = y[tr].min() - 1.0                     # support floor (explicit assumption)
    n_cal = 1000; K = 50
    cov_tot, fdp = [], []
    for r in range(R):
        rng = np.random.default_rng(9000 + r)
        p = rng.permutation(rest); C, T = p[:n_cal], p[n_cal:]
        # T1(b): total over the TOP-K by mu, with the exact (integer-certified) envelope
        sel = refsets.topk(mu[C], mu[T], K)
        J = T[sel.selected]; Rr = C[sel.member[0]]; ell = len(Rr) + K
        if r < 400:                                                    # DP-based check is slower
            Cb = envelope(ell, K, delta, rng)
            lb = mu[J].sum() + lower_total((y - mu)[Rr], K, ymin - mu[J].max(), Cb)
            cov_tot.append(lb <= y[J].sum())
        # T2: cfBH on the reference set, bottom-K
        selb = refsets.topk(mu[C], mu[T], K, largest=False)
        Jb = T[selb.selected]; Rb = C[selb.member[0]]
        null_i = y[Rb] >= c
        pv = np.array([(1 + np.sum(null_i & (gsc[Rb] < gsc[j]))) / (len(Rb) + 1) for j in Jb])
        o = np.argsort(pv); ok = np.flatnonzero(pv[o] <= q * np.arange(1, len(pv) + 1) / len(pv))
        rej = np.zeros(len(pv), bool)
        if len(ok):
            rej[o[: ok[-1] + 1]] = True
        fdp.append(np.sum(rej & (y[Jb] >= c)) / max(rej.sum(), 1))
    return np.mean(cov_tot), len(cov_tot), np.mean(fdp), np.std(fdp) / np.sqrt(len(fdp)), q, delta


def check_fast():
    rng = np.random.default_rng(5); worst = 0.0
    for _ in range(500):
        ell = int(rng.integers(2, 90)); K = int(rng.integers(1, ell)); nR = ell - K
        C = np.maximum.accumulate(np.minimum(K, rng.integers(0, K + 1, nR))).tolist() + [K]
        worst = max(worst, abs(exact_valid_prob(ell, K, C) - TOT.valid_prob_fast(ell, K, C)))
    viol = []
    for ell, K, d in [(40, 12, 1 / 3), (120, 60, 0.30), (200, 40, 1 / 3), (150, 120, 0.30)]:
        C = [int(c) for c in TOT.envelope_exact(ell, K, d, iters=16, fast=True)]
        viol.append((ell, K, d, exact_violation(ell, K, C), nontrivial(C, K)))
    return worst, viol


def main(R=4000):
    bad = 0
    print("1. Lemma P: chi-square uniformity of S given the pool (3 most frequent pools)")
    for ell, nobs, nsub, p in check_lemma():
        ok = p > 0.001; bad += not ok
        print(f"   pool size {ell:3d}: {nobs:6d} draws over {nsub:5d} subsets, chi2 p = {p:.3f}  {'ok' if ok else 'FAIL'}")
    print("2. Exact envelope: violation probability as an exact rational (target delta = 0.05; no tolerance)")
    for ell, K, path, v_mc, v, nt in check_envelope():
        ok = v <= Fraction(1, 20) and nt; bad += not ok
        print(f"   ell={ell:3d}, K={K:3d}, {path:7s}: exact envelope P(violation) = {float(v):.5f}"
              f"{'' if nt else ' (trivial fall-back)'}  {'ok' if ok else 'FAIL'}"
              f"   [retired MC envelope: {v_mc:.4f}{', above delta' if v_mc > 0.05 else ''}]")
    cov, n_cov, fdr, se, q, delta = check_t1_t2(R)
    ok1 = cov >= 1 - delta - 3 * np.sqrt(delta * (1 - delta) / n_cov); bad += not ok1
    ok2 = fdr <= q + 3 * se; bad += not ok2
    print(f"3. T1(b) total lower bound: P(total >= bound) = {cov:.4f} over {n_cov} splits (>= {1-delta})  {'ok' if ok1 else 'FAIL'}")
    print(f"4. T2 cfBH on reference set: FDR = {fdr:.4f} (se {se:.4f}) at q = {q}  {'ok' if ok2 else 'FAIL'}")
    worst, viol = check_fast(); ok = worst < 1e-9; bad += not ok
    print(f"5. fast DP: max |integer - float| = {worst:.1e} over 500 random envelopes  {'ok' if ok else 'FAIL'}")
    for ell, K, d, v, nt in viol:
        ok = v <= Fraction(repr(d)) and nt; bad += not ok
        print(f"   fast envelope ell={ell:3d}, K={K:3d}, delta={d:.3f}: exact P(violation) = {float(v):.5f}  {'ok' if ok else 'FAIL'}")
    print("ALL POOL-LEMMA CHECKS PASS" if bad == 0 else f"{bad} FAILURES")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000) else 1)
