"""Scores S (III(f)): any score fixed given the training fold keeps validity; the choice only
changes efficiency. Regression scores return arrays; classification scores return a function
of (probabilities, labels) so the same engines produce prediction SETS."""
import numpy as np


def abs_residual(y, mu):
    return np.abs(np.asarray(y) - np.asarray(mu))


def studentized(y, mu, sigma):
    """|y - mu| / sigma(x, s): scale / likelihood-ratio-type score; sigma from the training fold."""
    return np.abs(np.asarray(y) - np.asarray(mu)) / np.maximum(np.asarray(sigma), 1e-12)


def cqr(y, q_lo, q_hi):
    """Conformalized quantile regression score max(q_lo - y, y - q_hi)."""
    y = np.asarray(y)
    return np.maximum(np.asarray(q_lo) - y, y - np.asarray(q_hi))


def multi_max(Y, MU, SIG):
    """Joint multi-output score max_d |y_d - mu_d| / sigma_d (GeoXCP-type multi-output targets:
    one threshold gives a joint box valid for all outputs simultaneously)."""
    return np.max(np.abs(np.asarray(Y) - np.asarray(MU)) / np.maximum(np.asarray(SIG), 1e-12), axis=1)


def fit_sigma(X_train, y_train, X_all, mu_model_factory, seed=0, n_folds=2):
    """Cross-fitted local residual scale inside the training fold (never touches calibration
    labels): out-of-fold |residuals| regressed on X with a random forest."""
    from sklearn.ensemble import RandomForestRegressor
    X_train = np.asarray(X_train); y_train = np.asarray(y_train)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(y_train))
    oof = np.empty(len(y_train))
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        m = mu_model_factory().fit(X_train[tr], y_train[tr])
        oof[te] = np.abs(y_train[te] - m.predict(X_train[te]))
    sm = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.6,
                               random_state=seed, n_jobs=1).fit(X_train, oof)
    return np.maximum(sm.predict(X_all), 0.1 * oof.mean())


# ------------------------------------------------------------------ classification
def lac(probs, labels):
    """Least-ambiguous-classifier score 1 - p_hat(y | x). probs (n, K), labels (n,)."""
    probs = np.asarray(probs)
    return 1 - probs[np.arange(len(probs)), np.asarray(labels)]


def lac_all(probs):
    """LAC score of every candidate label, (n, K)."""
    return 1 - np.asarray(probs)


def aps_all(probs, rng=None):
    """Adaptive prediction sets score of every candidate label (randomized if rng given):
    S(x, y) = sum of probabilities of labels ranked at least as likely as y (minus U * p_y)."""
    probs = np.asarray(probs)
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    if rng is not None:
        cum = cum - rng.random((len(probs), 1)) * sorted_p
    S = np.empty_like(probs)
    np.put_along_axis(S, order, cum, axis=1)
    return S


def aps(probs, labels, rng=None):
    S = aps_all(probs, rng)
    return S[np.arange(len(S)), np.asarray(labels)]


def sets_from_threshold(S_all, q):
    """Prediction sets {y : S(x, y) <= q_j} as a boolean (m, K) matrix."""
    return np.asarray(S_all) <= np.asarray(q)[:, None]
