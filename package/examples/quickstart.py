"""mapselect in one minute: a synthetic map, a random audit, and intervals at the places a rule selects.

    python examples/quickstart.py
"""
import numpy as np
import mapselect as ms

rng = np.random.default_rng(0)
N = 4000
mu = rng.normal(size=N)                                             # the map's prediction at every place
sigma = 0.3 + 0.2 * np.abs(mu)                                      # its modelled error spread
y = mu + (0.2 + 0.5 * np.maximum(mu, 0)) * rng.standard_normal(N)   # the truth: errors grow at the top of the map

perm = rng.permutation(N)
audited, candidates = perm[:2000], perm[2000:]                      # a random audit; y is known only at audited places
scores = np.abs(y[audited] - mu[audited]) / sigma[audited]

sel = ms.top_k(mu[audited], mu[candidates], K=40)                   # the rule: the 40 highest-predicted candidates
q = ms.jomi(scores, sel, alpha=0.1)
J = candidates[sel.selected]

all_scores = np.abs(y[candidates] - mu[candidates]) / sigma[candidates]
q_split = np.sort(scores)[int(np.ceil(0.9 * (len(scores) + 1))) - 1]
print(f"{sel}")
print(f"reference places: {sel.sizes[0]}, half-width {q[0]:.2f} spreads (split CP: {q_split:.2f})")
print(f"coverage of split CP at all candidates: {(all_scores <= q_split).mean():.2f}")
print(f"coverage at the selected places: split CP {(all_scores[sel.selected] <= q_split).mean():.2f}, JOMI {(all_scores[sel.selected] <= q).mean():.2f}")
print(f"audit needed for a 0.9 chance of finite intervals: {ms.plan.n_for_finite(m=2000, K=40)} places")
