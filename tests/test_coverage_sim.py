"""Monte Carlo check of selection-conditional coverage on a FIXED spatially autocorrelated
finite population (design-based: only the calibration/test split is random).

    ~/miniforge3/envs/favela/bin/python tests/test_coverage_sim.py [reps]

Population: N units on [0,10]^2, y = GP field (range ~2) + heteroscedastic noise whose scale is
itself a spatial field, so residuals are strongly autocorrelated and extremes are noisier.
A fixed training fold fits mu (kNN smoother) and sigma; the remaining units are split at random
into calibration / test in every repetition. For each rule we pool coverage over selected test
units: JOMI must be in [1 - alpha, upper bound]; JOMI-ExCP (JOMI x RLCP+, SC-RLCP lemma) and
JOMI|group (quota, selected AND in state) must be >= 1 - alpha; vanilla split CP is not.
Also checks rules.kernel_mask == finiteness of excp.family.geocp_v2.
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True
from excp_select import refsets, methods as Mt  # noqa: E402
from excp import family as F  # noqa: E402

ALPHA = 0.1


def population(N=3000, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 10, (N, 2))
    D = np.sqrt(((s[:, None] - s[None]) ** 2).sum(-1))
    L = np.linalg.cholesky(np.exp(-D / 2.0) + 1e-8 * np.eye(N))
    f = L @ rng.standard_normal(N)
    logsd = 0.8 * (L @ rng.standard_normal(N)) / 1.5
    y = 2 * f + np.exp(logsd) * (1 + 1.5 * np.maximum(f, 0)) * rng.standard_normal(N)
    g = (s[:, 0] // 2.5).astype(int) * 4 + (s[:, 1] // 2.5).astype(int)     # 16 'states'
    return s, y, g


def main(R=3000):
    s, y, g = population()
    N = len(y); rng0 = np.random.default_rng(7)
    perm = rng0.permutation(N); tr, rest = perm[:1000], perm[1000:]
    tree = cKDTree(s[tr])
    _, nn = tree.query(s, k=15)
    mu = y[tr][nn].mean(1) + 1e-9 * rng0.standard_normal(N)
    _, nn2 = tree.query(s, k=40)
    res_tr = np.abs(y[tr] - mu[tr])
    sig = np.maximum(res_tr[nn2].mean(1), 0.05)
    S_abs, S_st = np.abs(y - mu), np.abs(y - mu) / sig
    n_cal = 1000; m = len(rest) - n_cal
    K = 50; Kg = {u: 4 for u in range(16)}
    cost = 1 + np.linalg.norm(s - 5, axis=1) / 3; val = mu - mu[tr].min() + 0.5
    B = 0.05 * m * cost[tr].mean()
    h_mask = 0.35
    acc = {}
    mask_agree = []

    def add(key, cov, ub=None):
        a = acc.setdefault(key, [0, 0, 0.0]); a[0] += int(cov.sum()); a[1] += len(cov)
        if ub is not None:
            a[2] += float(ub.sum())

    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        p = rng.permutation(rest); C, T = p[:n_cal], p[n_cal:]
        ctx = dict(S_cal=S_st[C], w_cal=np.ones(n_cal), w_test=np.ones(m), s_cal=s[C], s_test=s[T],
                   s_train=s[tr], w_train=np.ones(len(tr)), g_cal=g[C], g_test=g[T], rng=rng)
        sels = {
            "top-K": refsets.topk(mu[C], mu[T], K),
            "bottom-K": refsets.topk(mu[C], mu[T], K, largest=False),
            "quota": refsets.quota(mu[C], mu[T], g[C], g[T], Kg),
            "knapsack": refsets.knapsack_greedy(val[C], cost[C], val[T], cost[T], B),
            "geocp-mask": refsets.kernel_mask(s[C], s[T], h_mask, ALPHA),
        }
        if r < 200:
            q2 = F.geocp_v2(h_mask).thresholds(dict(ctx, S_cal=S_abs[C]), ALPHA)
            fin = np.zeros(m, bool); fin[sels["geocp-mask"].selected] = True
            mask_agree.append(np.mean(fin == np.isfinite(q2)))
        for nm, sel in sels.items():
            if len(sel) == 0:
                continue
            j = T[sel.selected]
            q = F.split_cp().thresholds(dict(ctx, S_cal=S_abs[C]), ALPHA)[0]
            add((nm, "vanilla |res|"), S_abs[j] <= q)
            q = F.split_cp().thresholds(ctx, ALPHA)[0]
            add((nm, "vanilla stud"), S_st[j] <= q)
            qa = Mt.jomi(sel).thresholds(dict(ctx, S_cal=S_abs[C]), ALPHA)
            add((nm, "JOMI |res|"), S_abs[j] <= qa, Mt.upper_bound(sel, ALPHA))
            qj = Mt.jomi(sel).thresholds(ctx, ALPHA)
            add((nm, "JOMI stud"), S_st[j] <= qj, Mt.upper_bound(sel, ALPHA))
            qe = Mt.jomi_excp(sel, 60).thresholds(ctx, ALPHA)
            add((nm, "JOMI-ExCP"), S_st[j] <= qe)
            if nm == "quota":
                qg = Mt.jomi(sel, by_group=True).thresholds(ctx, ALPHA)
                add((nm, "JOMI|group"), S_st[j] <= qg)
                # per-group check: pooled within the most extreme state
                gj = g[T][sel.selected]
                for u in (np.argmax([mu[g == v].mean() for v in range(16)]),):
                    add((nm, f"JOMI|group state{u}"), (S_st[j] <= qg)[gj == u])
                    add((nm, f"vanilla stud state{u}"), (S_st[j] <= q)[gj == u])
    se = lambda c, n: np.sqrt(0.09 / max(n, 1))
    print(f"population N={N}, n_cal={n_cal}, m={m}, reps={R}, alpha={ALPHA}")
    print(f"kernel_mask == isfinite(GeoCP v2): mean agreement {np.mean(mask_agree):.6f}")
    print(f"{'rule':11s} {'method':24s} {'cov|sel':>8s} {'MC se':>6s} {'bound':>7s} {'#sel':>8s}  check")
    bad = 0
    for (nm, meth), (c, n, ub) in acc.items():
        cov = c / n; e = se(c, n)
        bound = ub / n if ub else np.nan
        chk = ""
        if meth.startswith("JOMI"):
            ok = cov >= 1 - ALPHA - 3 * e and (np.isnan(bound) or cov <= bound + 3 * e)
            chk = "ok" if ok else "FAIL"; bad += not ok
        print(f"{nm:11s} {meth:24s} {cov:8.4f} {e:6.4f} {bound:7.4f} {n:8d}  {chk}")
    print("ALL SELECTION-CONDITIONAL CHECKS PASS" if bad == 0 else f"{bad} FAILURES")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000) else 1)
