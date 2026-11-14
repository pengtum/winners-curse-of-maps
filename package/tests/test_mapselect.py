"""Run with: python -m pytest tests  (or: python tests/test_mapselect.py)"""
import numpy as np
import mapselect as ms
from mapselect import bruteforce as bf, rules as ru, calibrate as cal
from mapselect.calibrate import _primal_beta, _dual_beta, _active_beta, _kth_two_sorted, objective


def _same(sel, M, sel_brute):
    assert np.array_equal(sel.selected, sel_brute)
    assert np.array_equal(sel.dense(), M)
    assert np.array_equal(sel.sizes, M.sum(1))


def test_top_k_matches_swaps():
    for s in range(40):
        r = np.random.default_rng(s); n, m = r.integers(5, 25), r.integers(6, 25); K = int(r.integers(1, m - 1))
        a, t = dict(mu=r.normal(size=n)), dict(mu=r.normal(size=m))
        for largest in (True, False):
            _same(ms.top_k(a["mu"], t["mu"], K, largest), *bf.reference_matrix(ru.topk(K, largest=largest), a, t)[::-1])


def test_quota_matches_swaps():
    for s in range(40):
        r = np.random.default_rng(100 + s); n, m = r.integers(8, 30), r.integers(8, 30)
        a = dict(mu=r.normal(size=n), g=r.integers(0, 4, n)); t = dict(mu=r.normal(size=m), g=r.integers(0, 4, m))
        Kg = {int(g): int(r.integers(0, 4)) for g in range(4)}
        for largest in (True, False):
            sel = ms.quota(a["mu"], t["mu"], a["g"], t["g"], Kg, largest)
            _same(sel, *bf.reference_matrix(ru.quota(Kg, largest=largest), a, t)[::-1])


def test_budget_matches_swaps():
    for s in range(40):
        r = np.random.default_rng(200 + s); n, m = r.integers(8, 30), r.integers(8, 30)
        a = dict(v=r.random(n) + .1, c=r.random(n) + .5); t = dict(v=r.random(m) + .1, c=r.random(m) + .5)
        B = float(t["c"].sum() * r.uniform(.1, .6))
        sel = ms.budget(a["v"], a["c"], t["v"], t["c"], B)
        _same(sel, *bf.reference_matrix(ru.knapsack_greedy(B), a, t)[::-1])


def test_mask_matches_swaps():
    for s in range(20):
        r = np.random.default_rng(300 + s); n, m = r.integers(10, 30), r.integers(8, 20)
        a = dict(s=r.random((n, 2))); t = dict(s=r.random((m, 2)))
        alpha = float(r.choice([.2, .3, .5])); h = float(r.uniform(.1, .5))
        sel = ms.publication_mask(a["s"], t["s"], h, alpha)
        _same(sel, *bf.reference_matrix(ru.kernel_mask(h, alpha), a, t)[::-1])


def test_kth_two_sorted():
    r = np.random.default_rng(1)
    for _ in range(300):
        a, b = np.sort(r.random(r.integers(0, 9))), np.sort(r.random(r.integers(0, 9)))
        if len(a) + len(b) == 0:
            continue
        k = int(r.integers(1, len(a) + len(b) + 1))
        assert _kth_two_sorted(a, b, k) == np.sort(np.r_[a, b])[k - 1]


def test_jomi_equals_direct_quantile():
    r = np.random.default_rng(2); n, m = 60, 40
    S = r.random(n)
    for sel in (ms.top_k(r.random(n), r.random(m), 5),
                ms.quota(r.random(n), r.random(m), r.integers(0, 3, n), r.integers(0, 3, m), {0: 2, 1: 3, 2: 1}),
                ms.budget(r.random(n) + .1, r.random(n) + .5, r.random(m) + .1, r.random(m) + .5, 6.0)):
        q = ms.jomi(S, sel, .1)
        for a in range(len(sel)):
            v = np.sort(S[sel.reference(a)]); k = int(np.ceil(.9 * (len(v) + 1) - 1e-9 * (len(v) + 1)))
            assert q[a] == (v[k - 1] if k <= len(v) else np.inf)


def test_jomi_coverage():
    """A fixed population with spatially structured values; a random audit each time; a rule picks the top of a noisy map."""
    r = np.random.default_rng(3); N = 600; mu = r.normal(size=N); y = mu + r.normal(size=N) * (1 + (mu > 1))
    cov, split = [], []
    for s in range(1500):
        p = np.random.default_rng([3, s]).permutation(N); C, T = p[:300], p[300:]
        sel = ms.top_k(mu[C], mu[T], 15); S = np.abs(y - mu)
        cov.append((S[T[sel.selected]] <= ms.jomi(S[C], sel, .1)).mean())
        split.append((S[T[sel.selected]] <= np.sort(S[C])[int(np.ceil(.9 * 301)) - 1]).mean())
    assert np.mean(cov) >= .89 and np.mean(split) < .85


def test_dual_and_primal_share_objective():
    """Both forms solve one programme: the dual's beta must attain the primal's optimum, and infinities must agree."""
    r = np.random.default_rng(4); checked = 0
    for s in range(60):
        n, p = int(r.integers(3, 80)), int(r.integers(2, 12)); tau = .9
        Phi = np.c_[np.ones(n), r.random((n, p - 1))]; S = r.random(n); phi = np.r_[1, r.random(p - 1)]; pen = np.r_[False, np.ones(p - 1, bool)]
        a, b = _primal_beta(S, Phi, phi, tau, .95, pen), _dual_beta(S, Phi, phi, tau, .95, pen)
        assert (a is None) == (b is None) == (n < 9), (n, a, b)
        if a is not None:
            assert abs(objective(a, S, Phi, phi, tau, .95, pen) - objective(b, S, Phi, phi, tau, .95, pen)) < 1e-7
            checked += 1
    assert checked > 30


def test_sc_geoccp_intercept_only_is_jomi():
    r = np.random.default_rng(5); n, m = 800, 100
    sel = ms.top_k(r.random(n), r.random(m), 5); S = r.random(n)
    target = ms.jomi(S, sel, .1)
    assert np.isfinite(target).all()                                    # otherwise this comparison proves nothing
    for form in ("dual", "primal"):
        q = ms.sc_geoccp(S, sel, np.ones((n, 1)), np.ones((len(sel), 1)), .1, form=form)
        assert np.allclose(q, target, atol=1e-8)


def test_active_set_attains_the_optimum():
    """The active-set answer must reach the full programme's objective, warm or cold, on problems large enough to prune."""
    r = np.random.default_rng(8); tau = .9
    for s in range(25):
        n, p = int(r.integers(9, 3000)), int(r.integers(2, 40)); l1 = float(r.choice([.95, 2.0]))
        Phi = np.c_[np.ones(n), np.exp(-r.random((n, p - 1)) * 3)]; S = np.abs(r.normal(size=n)) * (1 + Phi[:, 1]); pen = np.r_[False, np.ones(p - 1, bool)]
        beta0 = np.zeros(p); beta0[0] = np.quantile(S, tau)
        for k in range(3):
            phi = np.r_[1, np.exp(-r.random(p - 1) * 3)]
            b, rounds, nf = _active_beta(S, Phi, phi, tau, l1, pen, beta0)
            a = _dual_beta(S, Phi, phi, tau, l1, pen)
            assert abs(objective(b, S, Phi, phi, tau, l1, pen) - objective(a, S, Phi, phi, tau, l1, pen)) < 1e-6, (n, p, rounds, nf)
            beta0 = b


def test_active_set_infinite_only_when_jomi_is():
    r = np.random.default_rng(9); n = 8; Phi = np.c_[np.ones(n), r.random(n)]
    assert _active_beta(r.random(n), Phi, np.r_[1, .5], .9, .95, np.array([False, True]), np.zeros(2))[0] is None


def test_mask_tree_equals_dense():
    for s in range(12):
        r = np.random.default_rng(400 + s); n, m = int(r.integers(300, 2500)), 500
        sa, st = r.random((n, 2)), r.random((m, 2)); h = float(np.sqrt(30 / (2 * np.pi * n)))
        a, b = ms.publication_mask(sa, st, h, .1, method="dense"), ms.publication_mask(sa, st, h, .1, method="tree")
        assert np.array_equal(a.selected, b.selected) and np.array_equal(a.sets[0], b.sets[0])
        assert all(np.array_equal(np.sort(a.reference(i)), np.sort(b.reference(i))) for i in range(0, len(a), 7))


def test_cfbh_pvalues():
    r = np.random.default_rng(6); n, m = 50, 30
    audit_key, test_key = r.random(n), r.random(m)
    psi, null, ps = r.random(n), r.random(n) < .5, r.random(6)
    for largest in (True, False):
        sel = ms.top_k(audit_key, test_key, 6, largest=largest)
        p = ms.cf_pvalues(psi, null, ps, sel); R = sel.sets[0]
        for a in range(6):
            assert p[a] == (1 + ((psi[R] < ps[a]) & null[R]).sum()) / (len(R) + 1)
        rejected, funded_p = ms.cfbh(psi, null, ps, sel, q=.2)
        assert np.array_equal(funded_p, p) and np.array_equal(rejected, ms.bh(p, .2))


def test_cfbh_rejects_mixed_quotas():
    """One selected group does not make a quota's mixed reference set a top-K pool."""
    audit_key = np.array([.1, .2, .3, .4]); audit_group = np.array([0, 0, 1, 1])
    for test_key, test_group, expected_shared in (([.15, .25, .35, .45], [0, 0, 1, 1], False),
                                                ([.15, .25], [0, 0], True)):
        sel = ms.quota(audit_key, test_key, audit_group, test_group, {0: 1, 1: 1})
        assert sel.shared == expected_shared
        psi, null, ps = np.arange(4.), np.ones(4, bool), 4. + np.arange(len(sel))
        for call in (ms.cf_pvalues, ms.cfbh):
            with np.testing.assert_raises_regex(ValueError, "top- or bottom-K"):
                call(psi, null, ps, sel)


def test_cfbh_requires_shared_topk_pool():
    """Restricting a top-K reference set by group must not bypass the rule check."""
    sel = ms.top_k([.1, .2, .3, .4], [.15, .25], 1)
    grouped = ms.within_groups(sel, [0, 0, 1, 1], [0, 0])
    assert grouped.shared
    with np.testing.assert_raises_regex(ValueError, "top- or bottom-K"):
        ms.cfbh(np.arange(4.), np.ones(4, bool), [4.], grouped)
    malformed = ms.Selection(sel.selected, sel.sets * 2, sel.set_id, sel.name, sel.n_audit)
    with np.testing.assert_raises_regex(ValueError, "top- or bottom-K"):
        ms.cfbh(np.arange(4.), np.ones(4, bool), [4.], malformed)


def test_cfbh_rejects_tied_scores():
    sel = ms.top_k([.1, .2, .3, .4], [.15, .25, .35], 2)
    for psi, ps in (([0., 1., 2., 3.], [1., 4.]),  # reference--selected tie
                    ([0., 1., 1., 3.], [4., 5.]),  # reference--reference tie
                    ([0., 1., 2., 3.], [4., 4.])): # selected--selected tie
        for call in (ms.cf_pvalues, ms.cfbh):
            with np.testing.assert_raises_regex(ValueError, "tied values"):
                call(psi, np.ones(4, bool), ps, sel)


def test_cfbh_empty_selection():
    """No hypotheses means no p-values or rejections, even if unused audit scores tie."""
    for largest in (True, False):
        sel = ms.top_k([.1, .2, .3, .4], [.15, .25], 0, largest=largest)
        p = ms.cf_pvalues(np.zeros(4), np.ones(4, bool), [], sel)
        rejected, funded_p = ms.cfbh(np.zeros(4), np.ones(4, bool), [], sel)
        assert p.shape == funded_p.shape == rejected.shape == (0,)
        assert p.dtype.kind == funded_p.dtype.kind == "f" and rejected.dtype.kind == "b"
    unsupported = ms.quota([.1, .2], [.15, .25], [0, 1], [0, 1], {0: 0, 1: 0})
    with np.testing.assert_raises_regex(ValueError, "top- or bottom-K"):
        ms.cfbh([0., 1.], [True, True], [], unsupported)


def test_exact_total_bound_covers():
    r = np.random.default_rng(7); N = 400; mu = r.normal(size=N) + 5; y = np.maximum(mu + r.normal(size=N), 0); e = y - mu; miss = 0; M = 800
    for s in range(M):
        p = np.random.default_rng([7, s]).permutation(N); C, T = p[:200], p[200:]
        sel = ms.top_k(mu[C], mu[T], 10); J = T[sel.selected]
        miss += y[J].sum() < ms.exact_lower_total(mu[J], e[C], sel, .1, y_min=0.0)
    assert miss / M <= .1 + 3 * np.sqrt(.09 / M)


def test_good_paths_counts_subsets():
    """The lattice-path count against listing every K-subset of the slots."""
    from itertools import combinations
    from mapselect._envelope import good_paths
    r = np.random.default_rng(10)
    for _ in range(200):
        ell = int(r.integers(2, 13)); K = int(r.integers(1, ell)); nR = ell - K
        C = list(np.maximum.accumulate(r.integers(0, K + 1, nR))) + [K]
        n = 0
        for marks in combinations(range(ell), K):
            N, ok, b = 0, True, 0                                       # N_b = marks before the (b+1)-th unmarked slot
            for slot in range(ell):
                if slot in marks:
                    N += 1
                else:
                    ok &= N <= C[b]; b += 1
            n += ok
        assert good_paths(ell, K, C) == n


def test_planner():
    assert ms.plan.n_budget_for_finite(6000, 30) == 1869 and ms.plan.n_budget_for_finite(300, 30) == 92
    assert ms.plan.n_for_floor(3000, 150, 2, .8) == 14249
    assert abs(ms.plan.inv_moment(3000, 3000, 30) - 0.03332223) < 1e-7
    assert ms.plan.n_for_finite(10 ** 6, 10 ** 4) > 0                   # large pools stay fast


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
