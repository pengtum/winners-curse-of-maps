"""Conformal Benjamini-Hochberg on reference sets (the manuscript's Theorem 3)."""
import numpy as np


def _check_scope(psi, ps, sel):
    """This API supports top-/bottom-K pools and scores without ties."""
    # A mixed quota can also have one shared set when only one group has candidates.
    # Check the rule created by top_k, not just the number of reference sets.
    direction, separator, count = sel.name.partition("-")
    is_topk = direction in ("top", "bottom") and separator == "-" and count.isdecimal()
    if not is_topk or not sel.shared:
        raise ValueError(
            f"This cfBH API supports only a top- or bottom-K rule with one shared reference set; "
            f"{sel!r} has {len(sel.sets)} shared set(s){' and unit-specific extras' if sel.extras is not None else ''}. "
            "For a quota, build a top_k selection within each group from that group's audited and candidate units and run "
            "cfbh on each: that controls the FDR within each group, not over all groups.")
    if not len(sel):
        return
    v = np.r_[psi[sel.sets[0]], ps]
    if len(np.unique(v)) < len(v):
        raise ValueError("psi has tied values among the reference set and the selected units, which Theorem 3 excludes; "
                         "break the ties first, e.g. psi + 1e-9 * rng.standard_normal(len(psi)) with a fixed seed")


def cf_pvalues(psi_audit, null_audit, psi_selected, sel):
    """p_j = (1 + #{i in R_j : unit i is ineligible, psi_i < psi_j}) / (|R_j| + 1).

    Supports top_k selections in either direction with one shared reference set; other rules (including quotas with
    only one selected group) and tied values of psi raise ValueError. An empty supported selection returns no p-values."""
    psi = np.asarray(psi_audit, float); null = np.asarray(null_audit, bool); ps = np.asarray(psi_selected, float)
    _check_scope(psi, ps, sel)
    p = np.empty(len(sel))
    for a, s in enumerate(sel.sets):
        v = np.sort(psi[s][null[s]]); r = sel.set_id == a
        p[r] = (1 + np.searchsorted(v, ps[r], side="left")) / (len(s) + 1)
    return p


def bh(p, q):
    """Benjamini-Hochberg rejections at level q."""
    p = np.asarray(p, float); m = len(p)
    if not m:
        return np.zeros(0, bool)
    o = np.argsort(p); ok = np.flatnonzero(p[o] <= q * np.arange(1, m + 1) / m)
    out = np.zeros(m, bool)
    if len(ok):
        out[o[:ok[-1] + 1]] = True
    return out


def cfbh(psi_audit, null_audit, psi_selected, sel, q=0.1):
    """Units funded by cfBH at FDR level q, as a boolean array over the selected units, with their p-values."""
    p = cf_pvalues(psi_audit, null_audit, psi_selected, sel)
    return bh(p, q), p
