"""Selection rules as plain functions rule(cal, test) -> bool array over the test units.

cal / test are dicts of per-unit covariate arrays (never labels): 'mu' prediction, 'g' group
(state), 'v' value, 'c' cost, 's' (n, 2) coordinates. Every rule here is invariant to
permutations of the calibration units, which is JOMI's Theorem 1 assumption 1. These are the
reference implementations used by `bruteforce.reference_sets`; `refsets` has the closed forms.
"""
import numpy as np


def topk(K, key="mu", largest=True):
    """Select the K test units with the largest (smallest) key."""
    def rule(cal, test):
        x = np.asarray(test[key], float); x = x if largest else -x
        out = np.zeros(len(x), bool)
        if K > 0:
            out[np.argsort(-x, kind="stable")[:K]] = True
        return out
    rule.name = f"{'top' if largest else 'bottom'}-{K}"
    return rule


def quota(K_by_group, key="mu", largest=True):
    """State quota: within each group g select the K_g test units with the largest key.
    K_by_group is FIXED (e.g. proportional to population counts), not a function of the split."""
    def rule(cal, test):
        x = np.asarray(test[key], float); x = x if largest else -x
        g = np.asarray(test["g"]); out = np.zeros(len(x), bool)
        for u in np.unique(g):
            idx = np.flatnonzero(g == u)
            k = int(K_by_group.get(u, 0))
            if k > 0:
                out[idx[np.argsort(-x[idx], kind="stable")[:k]]] = True
        return out
    rule.name = "quota"
    return rule


def knapsack_greedy(B, value="v", cost="c"):
    """Budgeted selection: rank by value / cost and fund the longest prefix whose total cost
    stays within the fixed budget B (the usual 'fund until the money runs out' rule)."""
    def rule(cal, test):
        v = np.asarray(test[value], float); c = np.asarray(test[cost], float)
        o = np.argsort(-(v / c), kind="stable")
        out = np.zeros(len(v), bool)
        out[o[np.cumsum(c[o]) <= B]] = True
        return out
    rule.name = "knapsack"
    return rule


def kernel_mask(h, alpha):
    """'Publish only where GeoCP v2 is finite': the Gaussian kernel mass of the calibration
    set around the test location reaches (1 - alpha) / alpha (atom K(0) = 1). Uses calibration
    COORDINATES only (no labels), symmetrically in the calibration units."""
    thr = (1 - alpha) / alpha

    def rule(cal, test):
        sc = np.asarray(cal["s"], float); st = np.asarray(test["s"], float)
        D2 = ((st[:, None, :] - sc[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * D2 / h ** 2).sum(1) >= thr
    rule.name = "geocp_mask"
    return rule


def threshold(c, key="sigma", below=True):
    """Fixed-threshold mask, e.g. publish where the training-fold scale model sigma <= c."""
    def rule(cal, test):
        x = np.asarray(test[key], float)
        return x <= c if below else x >= c
    rule.name = "threshold"
    return rule
