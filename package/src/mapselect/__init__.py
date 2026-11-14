"""mapselect: uncertainty at the places a map selects.

Rank a map, select places, and calibrate each selected place only on audited places the same rule would have selected:

    import mapselect as ms
    sel = ms.top_k(mu[audited], mu[candidates], K=30)
    q = ms.jomi(scores[audited], sel, alpha=0.1)              # interval half-widths on the score scale
    ms.plan.n_for_finite(m=3000, K=30)                         # the audit needed, before any outcome is measured

Guarantees need a random audit of one finite population, a map fitted separately, and a rule that uses covariates alone and
treats audited units symmetrically.
"""
from .refsets import Selection, top_k, quota, budget, publication_mask, threshold, within_groups
from .calibrate import jomi, sc_geoccp
regioncp = sc_geoccp          # RegionCP, the name used in the paper (2026-11-08); sc_geoccp is kept
from .totals import selection_aware_total, exact_lower_total, vcs_credit
from .funding import cfbh, cf_pvalues, bh
from . import planner as plan, bruteforce, rules, basis

__version__ = "0.1.0"
__all__ = ["Selection", "top_k", "quota", "budget", "publication_mask", "threshold", "within_groups", "jomi", "regioncp", "sc_geoccp",
           "selection_aware_total", "exact_lower_total", "vcs_credit", "cfbh", "cf_pvalues", "bh", "plan", "bruteforce",
           "rules", "basis"]
