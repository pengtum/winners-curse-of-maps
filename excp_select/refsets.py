"""Closed-form JOMI reference sets for the geographic selection rules in `rules`.

Each function returns a `Selection`: the selected test indices and, for every selected unit j,
the reference membership R_j over the calibration units (bool matrix |S| x n_cal). Every
formula is checked case by case against the brute-force swap in tests/test_swap.py.

  topk       R = {i : x_i > T},  T = (K+1)-th largest test key            (JOMI Prop. 6)
  quota      R_j = {i : x_i > T_{g_i}^{(-j)}},  T_b^{(-j)} = K_b-th largest test key in
             group b with j removed                                           (new, verified)
  knapsack   R_j = {i : C_{-j}(r_i) + c_i <= B},  r = v / c, C_{-j}(r) = cost of the other
             test units with ratio > r                                       (new, verified)
  kernel_mask R_j = {i : M_i + K_h(s_i, s_j) >= (1-a)/a},  M_i = sum_{l != i} K_h(s_i, s_l)
             over calibration                                                 (new, verified)
  threshold  R = {i : x_i <= c}                                               (JOMI Prop. 7 type)

Group-conditional coverage (selected AND in state s) = compose the selection with
excp.localizers.mondrian(g_cal, g_test[selected]).
"""
import numpy as np
from excp.localizers import Localizer, VALID


class Selection:
    def __init__(self, selected, member, name):
        self.selected = np.asarray(selected, int)       # indices into the test set
        self.member = np.asarray(member, bool)          # (|S|, n_cal) reference membership
        self.name = name

    def __len__(self):
        return len(self.selected)

    @property
    def sizes(self):
        """|R_j| per selected unit."""
        return self.member.sum(1)

    def localizer(self):
        M = self.member
        return Localizer(lambda sl: M[sl].astype(float), lambda sl: np.ones(sl.stop - sl.start),
                         VALID + " given selection (JOMI)", f"select[{self.name}]")

    def restrict(self, mask_cal_rows):
        """Intersect each R_j with a per-j calibration mask (e.g. same group)."""
        return Selection(self.selected, self.member & np.asarray(mask_cal_rows, bool), self.name + "|g")


def _kth_largest(x, k):
    """k-th largest of x (1-indexed); -inf if fewer than k values, +inf if k <= 0."""
    if k <= 0:
        return np.inf
    if len(x) < k:
        return -np.inf
    return np.partition(x, len(x) - k)[len(x) - k]


def topk(x_cal, x_test, K, largest=True):
    xc = np.asarray(x_cal, float); xt = np.asarray(x_test, float)
    if not largest:
        xc, xt = -xc, -xt
    sel = np.zeros(len(xt), bool)
    sel[np.argsort(-xt, kind="stable")[:K]] = True
    T = _kth_largest(xt, K + 1)
    row = xc > T
    idx = np.flatnonzero(sel)
    return Selection(idx, np.broadcast_to(row, (len(idx), len(xc))).copy(),
                     f"{'top' if largest else 'bottom'}-{K}")


def quota(x_cal, x_test, g_cal, g_test, K_by_group, largest=True):
    xc = np.asarray(x_cal, float); xt = np.asarray(x_test, float)
    if not largest:
        xc, xt = -xc, -xt
    gc = np.asarray(g_cal); gt = np.asarray(g_test)
    groups = np.unique(np.r_[gc, gt])
    sel = np.zeros(len(xt), bool)
    T_all, T_minus = {}, {}
    for u in groups:
        idx = np.flatnonzero(gt == u); k = int(K_by_group.get(u, 0))
        if k > 0 and len(idx):
            sel[idx[np.argsort(-xt[idx], kind="stable")[:k]]] = True
        T_all[u] = _kth_largest(xt[idx], k)                  # j not in this group
        T_minus[u] = _kth_largest(xt[idx], k + 1) if k > 0 else np.inf   # selected j removed
    sidx = np.flatnonzero(sel)
    Tc_other = np.array([T_all[u] for u in gc])
    Tc_own = np.array([T_minus[u] for u in gc])
    M = np.empty((len(sidx), len(xc)), bool)
    for r, j in enumerate(sidx):
        T = np.where(gc == gt[j], Tc_own, Tc_other)
        M[r] = xc > T
    return Selection(sidx, M, "quota")


def knapsack_greedy(v_cal, c_cal, v_test, c_test, B):
    vc, cc = np.asarray(v_cal, float), np.asarray(c_cal, float)
    vt, ct = np.asarray(v_test, float), np.asarray(c_test, float)
    rt, rc = vt / ct, vc / cc
    o = np.argsort(-rt, kind="stable")
    sel = np.zeros(len(vt), bool); sel[o[np.cumsum(ct[o]) <= B]] = True
    sidx = np.flatnonzero(sel)
    # cost of all test units with ratio > r_i, via ascending sort + suffix sums
    a = np.argsort(rt); rs = rt[a]; suf = np.r_[np.cumsum(ct[a][::-1])[::-1], 0.0]
    above = suf[np.searchsorted(rs, rc, side="right")]          # sum_{l: r_l > r_i} c_l
    M = np.empty((len(sidx), len(vc)), bool)
    for r, j in enumerate(sidx):
        Cm = above - ct[j] * (rt[j] > rc)
        M[r] = Cm + cc <= B
    return Selection(sidx, M, "knapsack")


def kernel_mask(s_cal, s_test, h, alpha, chunk=2000):
    sc = np.asarray(s_cal, float); st = np.asarray(s_test, float)
    thr = (1 - alpha) / alpha

    def K(A, Bm):
        D2 = np.maximum((A ** 2).sum(1)[:, None] + (Bm ** 2).sum(1)[None, :] - 2 * A @ Bm.T, 0)
        return np.exp(-0.5 * D2 / h ** 2)
    Mi = np.empty(len(sc))
    for a in range(0, len(sc), chunk):
        Mi[a:a + chunk] = K(sc[a:a + chunk], sc).sum(1) - 1.0          # drop K(s_i, s_i) = 1
    mass = np.empty(len(st))
    for a in range(0, len(st), chunk):
        mass[a:a + chunk] = K(st[a:a + chunk], sc).sum(1)
    sidx = np.flatnonzero(mass >= thr)
    M = np.empty((len(sidx), len(sc)), bool)
    for a in range(0, len(sidx), chunk):
        M[a:a + chunk] = Mi[None, :] + K(st[sidx[a:a + chunk]], sc) >= thr
    return Selection(sidx, M, "geocp_mask")


def threshold(x_cal, x_test, c, below=True):
    xc = np.asarray(x_cal, float); xt = np.asarray(x_test, float)
    sel = xt <= c if below else xt >= c
    row = xc <= c if below else xc >= c
    sidx = np.flatnonzero(sel)
    return Selection(sidx, np.broadcast_to(row, (len(sidx), len(xc))).copy(), "threshold")
