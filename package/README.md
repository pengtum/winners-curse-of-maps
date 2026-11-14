# mapselect

Uncertainty at the places a map selects.

Machine-learning maps rank places, and a rule picks some of them: the highest carbon stock, the poorest villages in each country,
the most valuable sites within a travel budget. Intervals that are valid on average are often wrong at exactly those places.
`mapselect` calibrates each selected place only on audited places that the same rule would have selected, and tells you before any
field work how large the audit has to be.

```python
import numpy as np
import mapselect as ms

# mu: the map's prediction; audited / candidates: a random split of one population; y known at audited places
sel = ms.top_k(mu[audited], mu[candidates], K=30)
scores = np.abs(y[audited] - mu[audited]) / sigma[audited]
q = ms.jomi(scores, sel, alpha=0.1)                     # may be +inf if too few audited places lie past the cutoff
J = candidates[sel.selected]
lower, upper = mu[J] - q * sigma[J], mu[J] + q * sigma[J]

ms.plan.n_for_finite(m=3000, K=30)                      # audited places for a 0.9 chance of finite intervals
ms.plan.n_budget_for_finite(population=300, K=30)       # the same when the audit is drawn from the map's top 300 places
```

| Task | Function | Guarantee |
|---|---|---|
| Rules | `top_k`, `quota`, `budget`, `publication_mask`, `threshold`, `within_groups` | closed-form reference sets, checked against literal swaps |
| Intervals at selected places | `jomi` | coverage at least 1 − α among selected places, on average over audits |
| Coverage within regions (RegionCP) | `regioncp` (also `sc_geoccp`), `basis.Multiscale` | Theorem 1's floors, which may be vacuous |
| Project totals | `exact_lower_total` | finite sample |
| | `selection_aware_total`, `vcs_credit` | asymptotic; the protocol rule itself |
| Funding | `cfbh` | false discovery rate at most q, for a top- or bottom-K rule without tied scores (checked) |
| Audit size | `plan.p_finite`, `plan.n_for_finite`, `plan.n_budget_for_finite`, `plan.floor`, `plan.n_for_floor` | no outcomes used |
| Any other rule | `bruteforce.reference_matrix` | literal swaps; slow |

**What the guarantees need.** A random audit of one finite population (or of a stratum fixed by the map in advance), a map fitted
separately, and a rule that uses covariates alone and treats audited places symmetrically. Rules that look at audited outcomes,
audits that are not random, and transfer to unsampled regions fall outside them. Coverage holds on average over audits, and counts
infinite intervals as covering.

`cfbh` accepts only `top_k` selections (either direction) with one shared reference set and distinct scores. It rejects
quota selections even when only one group is selected. An empty supported selection returns empty p-values and rejections.

**Install.** `pip install .` from this folder (not yet on PyPI; `RELEASE.md` has the steps). Needs numpy and scipy; `pip install .[basis]` adds scikit-learn for `basis.Multiscale`.

**Tests.** `python -m pytest tests`.

**Status.** Version 0.1.0, not yet on PyPI. Licence: MIT. Citation: [manuscript to record].

## Performance

One core of an Apple M6 laptop, synthetic data (`experiments/run_package_benchmark.py` in the project; `results/fia_db/RESULTS_PACKAGE_BENCHMARK.md`).

| Task | Size | Time | Memory |
|---|---|---|---|
| Top-1% rule, thresholds | 10⁸ candidates, 10⁷ audited | 1.4 s | 2.5 GB |
| Quota, 50 groups | same | 40 s | 5.7 GB |
| Budget | same | 35 s | 6.9 GB |
| Publication mask, k-d tree, selection only | 10⁶ audited | 2 min | 2.2 GB |
| RegionCP, later units (one reference set) | 10⁵ reference units, 60 basis functions | 0.036 s per unit | 0.9 GB |
| Exact total bound | 10⁴ selected | 25 s | 0.2 GB |

At small sizes these gains shrink or vanish: at 10³ reference units the active set is 1.6 times faster than the full dual, and the budget rule is no faster than a careful per-unit loop up to 10⁵ candidates.
