"""Bounds on the total of a top- or bottom-K selection (the manuscript's Lemma 1 and Theorem 2)."""
import numpy as np
from ._envelope import envelope_exact, lower_total, valid_prob_fast, violation_exact

Z = {1 / 3: 0.4307, 0.30: 0.524417}


def _pool(sel):
    if not sel.shared:
        raise ValueError("total bounds need one shared reference set (a top- or bottom-K rule, or a fixed quota group by group)")
    return sel.sets[0]


def selection_aware_total(pred_selected, resid_audit, sel):
    """Difference estimate of the selected total, sum(pred) + K mean(e_R), with its standard error s_R (K + K^2 / n_R)^(1/2).
    A central-limit result: no finite-sample guarantee. Returns (estimate, standard error); +inf if |R| < 2."""
    pred = np.asarray(pred_selected, float); e = np.asarray(resid_audit, float)[_pool(sel)]
    K, nR = len(pred), len(e)
    if nR < 2:
        return float(pred.sum()), np.inf
    return float(pred.sum() + K * e.mean()), float(e.std(ddof=1) * np.sqrt(K + K ** 2 / nR))


def exact_lower_total(pred_selected, resid_audit, sel, delta, y_min=0.0, iters=16):
    """Lower bound L with Pr(selected total >= L) >= 1 - delta in finite samples, for outcomes known to be >= y_min."""
    pred = np.asarray(pred_selected, float); e = np.asarray(resid_audit, float)[_pool(sel)]
    K, nR = len(pred), len(e)
    C = envelope_exact(nR + K, K, delta, iters=iters, fast=True)
    return float(pred.sum() + lower_total(e, K, y_min - pred.max(), C))


def vcs_credit(estimate, se):
    """Verified Carbon Standard rule: no deduction at relative uncertainty U <= 10%, none creditable above 100%, else 0.4307 SE."""
    U = 1.6449 * se / estimate if estimate > 0 else np.inf
    return 0.0 if U > 1 else (estimate if U <= 0.10 else estimate - 0.4307 * se)
