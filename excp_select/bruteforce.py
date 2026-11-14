"""Generic JOMI reference sets by literal swapping (Jin & Ren 2025, Sec. 3.2).

For a selected test unit j and each calibration unit i: put X_i in test slot j and (X_j) in
calibration slot i, rerun the rule, keep i if slot j is selected. Cost O(|S| n) rule calls --
the ground truth for the closed forms and the fallback for any label-free rule.
"""
import numpy as np

from .refsets import Selection


def _swap(cal, test, i, j):
    c2, t2 = {}, {}
    for k in test:
        cv, tv = np.array(cal[k], copy=True), np.array(test[k], copy=True)
        cv[i], tv[j] = test[k][j], cal[k][i]
        c2[k], t2[k] = cv, tv
    return c2, t2


def reference_sets(rule, cal, test):
    sel = np.asarray(rule(cal, test), bool)
    sidx = np.flatnonzero(sel)
    n = len(next(iter(cal.values())))
    M = np.zeros((len(sidx), n), bool)
    for r, j in enumerate(sidx):
        for i in range(n):
            c2, t2 = _swap(cal, test, i, j)
            M[r, i] = bool(rule(c2, t2)[j])
    return Selection(sidx, M, getattr(rule, "name", "rule") + "[brute]")
