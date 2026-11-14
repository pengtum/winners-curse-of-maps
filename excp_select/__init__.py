"""excp_select -- selection-conditional calibration (JOMI reference sets) for excp.

A selection rule picks test units from model predictions / covariates (top-K, state quotas,
knapsack, publish masks). Conditioning on "unit j was selected" is a data-dependent Mondrian
(Jin & Ren, JRSSB 2025): the reference set R_j holds the calibration units that would also be
selected if they were swapped into test slot j. Here a reference set is an excp `Localizer`
(0/1 factors), so every excp engine / weight / valid localizer composes with it:

    sel = refsets.topk(mu_cal, mu_test, K)              # Selection
    q   = methods.jomi(sel).thresholds(ctx, alpha)      # exact JOMI (constant engine)
    q   = methods.jomi_excp(sel, kappa).thresholds(...)  # JOMI x RLCP+ (theory/SC_RLCP.md)

Validity: random split (or known design) + rule invariant to permutations of the calibration
set + rule free of calibration labels. Test-centred kernels (GeoCP v1/v2) must NOT be composed
with a selection localizer -- only Pi weights and random localization (RLCP/RLCP+) may be.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The results were produced with the 21 Oct 2026 state of excp. The live package has since been renamed
# (geounicp) and its family API changed (no geocp_v2 / excp_default), so the verified snapshot comes first
# (vendor/excp_20261021/README.md); QW2_EXCP=live uses the live package instead.
_EXCP = (os.path.join(os.path.dirname(_ROOT), "design_based_spatial_cp", "excp_pkg")
         if os.environ.get("QW2_EXCP") == "live" else os.path.join(_ROOT, "vendor", "excp_20261021"))
if _EXCP not in sys.path:
    sys.path.insert(0, _EXCP)

from . import rules, refsets, bruteforce, methods, basis, totals  # noqa: E402,F401

__version__ = "0.1.0"
