"""R -> Pi: exchange weights determined by the source of randomness (GeoCP_framework.md §2).

All functions return per-unit weights w_k with Pi(k | A) proportional to w_k (product-form
designs, Prop. 2), so the same vector serves calibration (w[C]) and test (w[T]).
"""
import numpy as np


def uniform(N):
    """Random split / SRSWOR / exchangeable data: Pi uniform."""
    return np.ones(N)


def design_odds(pi):
    """Poisson design with known inclusion probabilities pi: w = (1 - pi) / pi (M1-T1(a)).
    Target: ALL UNSAMPLED units. (1/pi corresponds to all units and is the wrong target.)"""
    pi = np.asarray(pi, float)
    return (1 - pi) / pi


def cps_odds(ptil):
    """Conditional Poisson / rejective design: odds of the WORKING probabilities (M1-T1(b))."""
    return design_odds(ptil)


def sampford_bag_weights(pi_bag, n):
    """Sampford design (not product form): Pi(k | A) proportional to (c_A + pi_k)(1 - pi_k)/pi_k,
    c_A = n - sum_{i in A} pi_i (M1-Prop. 2; form per Tille 2006 -- to be double-checked).
    pi_bag: inclusion probabilities of the units of ONE bag A (calibration + the test unit)."""
    pi_bag = np.asarray(pi_bag, float)
    cA = n - pi_bag.sum()
    return (cA + pi_bag) * (1 - pi_bag) / pi_bag


def covariate_shift(density_ratio):
    """Super-population covariate shift (Tibshirani et al. 2019): w = dQ/dP(x)."""
    return np.asarray(density_ratio, float)


def tilt(w, a):
    """Target tilt (III(c)): coverage for the target measure a (population, area, a target-region
    indicator, a density ratio). Likelihood-ratio weights are exactly such a tilt."""
    return np.asarray(w, float) * np.asarray(a, float)


# ------------------------------------------------------------------ estimated inclusion
def estimate_pi(X_pop, sampled, n_target=None, model=None, clip=(1e-4, 0.999)):
    """Estimate inclusion probabilities of a non-probability / unknown-design sample from the
    POPULATION covariates (known for every unit of a finite map): a probabilistic classifier of
    'sampled vs not', calibrated to sum to the sample size. Returns pi_hat.
    The resulting weights are only as good as the 'ignorability given X' assumption; report the
    diagnostics from overlap_diagnostics and use robust.gamma_box for sensitivity."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X_pop = np.asarray(X_pop, float); sampled = np.asarray(sampled, bool)
    clf = model or HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                  min_samples_leaf=40, l2_regularization=1.0)
    # cross-fitted probabilities so a unit's own label does not set its pi_hat
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(X_pop))
    p = np.empty(len(X_pop))
    for f in range(5):
        tr, te = fold != f, fold == f
        clf.fit(X_pop[tr], sampled[tr])
        p[te] = clf.predict_proba(X_pop[te])[:, 1]
    n = sampled.sum() if n_target is None else n_target
    p = np.clip(p * n / p.sum(), *clip)
    return p


def overlap_diagnostics(pi_hat, sampled, floor=None):
    """Diagnostics for estimated weights: AUC of the sample-vs-population classifier, Kish ESS
    of the calibration weights, share of unsampled units below the pi floor ('extrapolation')."""
    from sklearn.metrics import roc_auc_score
    pi_hat = np.asarray(pi_hat); sampled = np.asarray(sampled, bool)
    w = design_odds(pi_hat)
    floor = np.quantile(pi_hat[sampled], 0.01) if floor is None else floor
    ws = w[sampled]
    return dict(auc=float(roc_auc_score(sampled, pi_hat)),
                kish_ess=float(ws.sum() ** 2 / (ws ** 2).sum()),
                n_sample=int(sampled.sum()),
                extrapolation_share=float(np.mean(pi_hat[~sampled] < floor)),
                pi_floor=float(floor))
