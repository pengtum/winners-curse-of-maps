"""Closed-form reference sets for covariate-only selection rules, in compact form.

The reference set of a selected unit j is R_j = {audited i : j is still selected after swapping i and j}. Most selected units
share most of their reference set, so a `Selection` stores shared index arrays, plus, for rules where R_j really differs by
unit (budget, mask), the few audited units that can enter it. Memory is about the size of the reference sets instead of
|selected| x n, which is what makes a rule on 10^6 candidates routine.

Keys must be distinct for the guarantees of the manuscript's Theorems; ties do not break these functions.
All functions are checked against literal swaps in tests/test_refsets.py.
"""
import numpy as np


class Selection:
    """Selected test units and their reference sets.

    selected   indices into the test arrays
    sets       list of int arrays of audited-unit indices, shared by the units that point to them
    set_id     for each selected unit, the entry of `sets` it uses
    extras     optional function r -> audited-unit indices in R_j beyond the shared set, for the r-th selected unit
    """

    def __init__(self, selected, sets, set_id, name, n_audit, extras=None):
        self.selected = np.asarray(selected, int)
        self.sets = [np.asarray(s, int) for s in sets]
        self.set_id = np.asarray(set_id, int)
        self.name = name
        self.n_audit = int(n_audit)
        self.extras = extras

    def __len__(self):
        return len(self.selected)

    def __repr__(self):
        return f"Selection({self.name}, {len(self)} selected, {len(self.sets)} shared reference set(s))"

    def reference(self, r):
        """Audited-unit indices of R_j for the r-th selected unit."""
        base = self.sets[self.set_id[r]]
        return np.r_[base, self.extras(r)] if self.extras is not None else base

    @property
    def sizes(self):
        base = np.array([len(s) for s in self.sets], int)[self.set_id] if len(self) else np.zeros(0, int)
        if self.extras is None:
            return base
        return base + np.array([len(self.extras(r)) for r in range(len(self))], int)

    @property
    def shared(self):
        """True if all selected units have one reference set."""
        return self.extras is None and len(self.sets) == 1

    def dense(self):
        """Boolean matrix |selected| x n_audit (for checks and small problems)."""
        M = np.zeros((len(self), self.n_audit), bool)
        for r in range(len(self)):
            M[r, self.reference(r)] = True
        return M


def _kth_largest(x, k):
    """k-th largest of x (1-indexed); -inf if fewer than k values, +inf if k <= 0."""
    if k <= 0:
        return np.inf
    if len(x) < k:
        return -np.inf
    return np.partition(x, len(x) - k)[len(x) - k]


def top_k(key_audit, key_test, K, largest=True):
    """The K candidates with the largest (or smallest) key. R_j = {i : x_i > (K+1)-th largest candidate key} for every j."""
    xc = np.asarray(key_audit, float); xt = np.asarray(key_test, float)
    if not largest:
        xc, xt = -xc, -xt
    K = int(K)
    sel = np.argpartition(-xt, K - 1)[:K] if 0 < K < len(xt) else np.arange(min(K, len(xt)))
    sel = np.sort(sel)
    return Selection(sel, [np.flatnonzero(xc > _kth_largest(xt, K + 1))], np.zeros(len(sel), int),
                     f"{'top' if largest else 'bottom'}-{K}", len(xc))


def quota(key_audit, key_test, group_audit, group_test, K_by_group, largest=True):
    """The K_g candidates with the largest key in each group g.

    R_j = {i : x_i > T_{g_i}}, where T_g is the (K_g+1)-th largest candidate key of g if g is j's group and the K_g-th largest
    otherwise. All selected units of one group share R_j, so there is one set per group."""
    xc = np.asarray(key_audit, float); xt = np.asarray(key_test, float)
    if not largest:
        xc, xt = -xc, -xt
    gc = np.asarray(group_audit); gt = np.asarray(group_test)
    ugt, inv_t = np.unique(gt, return_inverse=True)
    order = np.lexsort((-xt, inv_t)); starts = np.searchsorted(inv_t[order], np.arange(len(ugt)))
    ends = np.r_[starts[1:], len(xt)]
    picked = []
    other, own = {}, {}
    for a, u in enumerate(ugt):
        k = int(K_by_group.get(u if not isinstance(u, np.generic) else u.item(), 0))
        members = order[starts[a]:ends[a]]
        if k > 0:
            picked.append(members[:k])
        other[a] = xt[members[k - 1]] if 0 < k <= len(members) else (np.inf if k <= 0 else -np.inf)
        own[a] = xt[members[k]] if (k > 0 and k < len(members)) else (np.inf if k <= 0 else -np.inf)
    sel = np.sort(np.concatenate(picked)) if picked else np.zeros(0, int)
    # audited units of a group with no candidates: K_g-th largest of nothing is -inf if K_g > 0, +inf otherwise
    ca = np.searchsorted(ugt, gc); known = (ca < len(ugt)) & (ugt[np.minimum(ca, len(ugt) - 1)] == gc)
    T_other = np.full(len(xc), np.inf)
    for a in other:
        T_other[(ca == a) & known] = other[a]
    for u in np.unique(gc[~known]):
        k = int(K_by_group.get(u if not isinstance(u, np.generic) else u.item(), 0))
        T_other[gc == u] = -np.inf if k > 0 else np.inf
    base = xc > T_other
    sets, set_of = [], {}
    for a in np.unique(inv_t[sel]):
        in_g = (ca == a) & known
        m = base.copy(); m[in_g] = xc[in_g] > own[a]
        set_of[a] = len(sets); sets.append(np.flatnonzero(m))
    return Selection(sel, sets, np.array([set_of[a] for a in inv_t[sel]], int), "quota", len(xc))


def budget(value_audit, cost_audit, value_test, cost_test, B):
    """Greedy budget: candidates in decreasing value / cost while cumulative cost stays within B.

    R_j = {i : Q_{-j}(r_i) + t_i <= B}, Q_{-j}(r) the cost of the other candidates with a ratio above r. Audited units meeting
    this with j's cost left in form the shared set; the rest can enter only if removing j's cost admits them."""
    vc, cc = np.asarray(value_audit, float), np.asarray(cost_audit, float)
    vt, ct = np.asarray(value_test, float), np.asarray(cost_test, float)
    rt, rc = vt / ct, vc / cc
    o = np.argsort(-rt, kind="stable")
    keep = np.cumsum(ct[o]) <= B
    sel = np.sort(o[keep])
    a = np.argsort(rt); rs = rt[a]; suf = np.r_[np.cumsum(ct[a][::-1])[::-1], 0.0]
    load = suf[np.searchsorted(rs, rc, side="right")] + cc             # Q(r_i) + t_i, with j's cost still counted if r_j > r_i
    base = np.flatnonzero(load <= B)
    cmax = ct[sel].max() if len(sel) else 0.0
    cand = np.flatnonzero((load > B) & (load - cmax <= B))
    rj, cj = rt[sel], ct[sel]

    def extras(r):
        return cand[(rj[r] > rc[cand]) & (load[cand] - cj[r] <= B)]
    return Selection(sel, [base], np.zeros(len(sel), int), "budget", len(vc), extras=extras)


def _gauss(A, Bm, h):
    D2 = np.maximum((A ** 2).sum(1)[:, None] + (Bm ** 2).sum(1)[None, :] - 2 * A @ Bm.T, 0)
    return np.exp(-0.5 * D2 / h ** 2)


def _kernel_sums(pts, sc, h, tree, radius, chunk):
    """Sum over audited units of K_h(pts, s_i): all of them (tree is None), or those within `radius`."""
    out = np.empty(len(pts))
    if tree is None:
        for a in range(0, len(pts), chunk):
            out[a:a + chunk] = _gauss(pts[a:a + chunk], sc, h).sum(1)
        return out
    for a in range(0, len(pts), chunk):
        P = pts[a:a + chunk]
        nb = tree.query_ball_point(P, radius, workers=1)
        counts = np.fromiter((len(x) for x in nb), int, len(nb))
        rows = np.repeat(np.arange(len(P)), counts)
        cols = np.concatenate([np.asarray(x, int) for x in nb]) if counts.sum() else np.zeros(0, int)
        d2 = ((P[rows] - sc[cols]) ** 2).sum(1)
        out[a:a + chunk] = np.bincount(rows, np.exp(-0.5 * d2 / h ** 2), len(P))
    return out


def publication_mask(s_audit, s_test, h, alpha, chunk=2000, method="auto", tol=1e-9):
    """Publish candidate k when sum_i K_h(s_k, s_i) >= (1-alpha)/alpha over audited units, K_h Gaussian.

    R_j = {i : W_i + K_h(s_i, s_j) >= (1-alpha)/alpha}, W_i = sum_{i' != i} K_h(s_i, s_i').

    method="dense" visits every audited unit for every kernel sum, quadratic in the audit. method="tree" prunes with a k-d tree
    at a radius where the omitted mass is below tol, then recomputes densely the few sums within tol of the threshold, so the
    selected units and reference sets are the same as dense. "auto" uses the tree above 20,000 audited units."""
    sc = np.asarray(s_audit, float); st = np.asarray(s_test, float)
    thr = (1 - alpha) / alpha
    use_tree = method == "tree" or (method == "auto" and len(sc) > 20000)
    tree, radius = None, None
    if use_tree:
        from scipy.spatial import cKDTree
        tree = cKDTree(sc); radius = h * np.sqrt(2 * np.log(max(len(sc), 2) / tol))

    W = _kernel_sums(sc, sc, h, tree, radius, chunk) - 1.0              # drop K(s_i, s_i) = 1
    if use_tree:
        for level in (thr, thr - 1.0):                                  # the two comparisons that define base and cand below
            near = np.flatnonzero(np.abs(W - level) <= tol)
            if len(near):
                W[near] = _kernel_sums(sc[near], sc, h, None, None, chunk) - 1.0
    mass = _kernel_sums(st, sc, h, tree, radius, chunk)
    if use_tree:
        near = np.flatnonzero(np.abs(mass - thr) <= tol)
        if len(near):
            mass[near] = _kernel_sums(st[near], sc, h, None, None, chunk)
    sel = np.flatnonzero(mass >= thr)
    base = np.flatnonzero(W >= thr)
    cand = np.flatnonzero((W < thr) & (W + 1.0 >= thr))

    def extras(r):
        if not len(cand):
            return cand
        return cand[W[cand] + _gauss(st[sel[r]][None, :], sc[cand], h)[0] >= thr]
    return Selection(sel, [base], np.zeros(len(sel), int), "publication mask", len(sc), extras=extras)


def threshold(key_audit, key_test, c, above=True):
    """Candidates past a fixed cut-off c; R_j holds the audited units past it."""
    xc = np.asarray(key_audit, float); xt = np.asarray(key_test, float)
    sel = np.flatnonzero(xt >= c if above else xt <= c)
    return Selection(sel, [np.flatnonzero(xc >= c if above else xc <= c)], np.zeros(len(sel), int), "threshold", len(xc))


def within_groups(sel, group_audit, group_test):
    """R_j intersected with j's group: coverage given the group, and the sets R_g of the pool lemma under a quota."""
    if sel.extras is not None:
        raise ValueError("within_groups needs a rule with shared reference sets")
    gc = np.asarray(group_audit); gj = np.asarray(group_test)[sel.selected]
    key = {}; sets = []; ids = np.empty(len(sel), int)
    for r in range(len(sel)):
        k = (int(sel.set_id[r]), gj[r].item() if isinstance(gj[r], np.generic) else gj[r])
        if k not in key:
            base = sel.sets[sel.set_id[r]]
            key[k] = len(sets); sets.append(base[gc[base] == gj[r]])
        ids[r] = key[k]
    return Selection(sel.selected, sets, ids, sel.name + " within group", sel.n_audit)
