"""Reference sets by literal swapping (Jin & Ren 2025, Sec. 3.2): the ground truth for the closed forms, and the fallback for
any covariate-only rule. Cost: |selected| x n reruns of the rule, each at least linear in the candidates."""
import numpy as np


def _swap(audit, test, i, j):
    a2, t2 = {}, {}
    for k in test:
        av, tv = np.array(audit[k], copy=True), np.array(test[k], copy=True)
        av[i], tv[j] = test[k][j], audit[k][i]
        a2[k], t2[k] = av, tv
    return a2, t2


def reference_matrix(rule, audit, test, rows=None):
    """Boolean matrix |selected| x n. `rule(audit, test)` returns a boolean array over test units; audit and test are dicts of
    arrays with the same keys. `rows` limits the work to some selected units, for timing a sample of swaps."""
    sel = np.flatnonzero(np.asarray(rule(audit, test), bool))
    rows = np.arange(len(sel)) if rows is None else np.asarray(rows, int)
    n = len(next(iter(audit.values())))
    M = np.zeros((len(rows), n), bool)
    for a, r in enumerate(rows):
        for i in range(n):
            a2, t2 = _swap(audit, test, i, sel[r])
            M[a, i] = bool(rule(a2, t2)[sel[r]])
    return sel, M
