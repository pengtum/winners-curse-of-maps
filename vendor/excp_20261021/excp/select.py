"""Decision layer D: exhaustive screening of all unsampled units for H_j: y_j <= c_j
(M1-T4, GeoCP_framework III(e)).

bh / wbh        BH on (weighted) conformal p-values; valid for uniform Pi (cfBH), wbh has no
                finite-sample guarantee under heterogeneous weights
wcs             Weighted Conformalized Selection (exact vectorised); 'hete' (randomised) or 'dtm'
softrank_ebh    NEW deterministic procedure (O2): label-free thresholds tau (fixed given the
                training fold) + soft-rank conformal e-values + e-BH; FDR <= q for any Pi
"""
import numpy as np


def weighted_pvalues(V_cal, w_cal, Vhat, w_test):
    o = np.argsort(V_cal); Vs = np.asarray(V_cal)[o]
    cs = np.r_[0.0, np.cumsum(np.asarray(w_cal, float)[o])]
    base = cs[np.searchsorted(Vs, Vhat, side="left")]
    return (base + w_test) / (cs[-1] + w_test), base, cs[-1]


def bh(p, q):
    p = np.asarray(p); m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    ps = np.sort(p)
    ok = np.flatnonzero(ps <= q * np.arange(1, m + 1) / m)
    return p <= q * (ok[-1] + 1) / m if len(ok) else np.zeros(m, bool)


def ebh(e, q):
    """e-BH (Wang & Ramdas 2022): reject the k* largest e-values, k* = max{k: e_(k) >= m/(q k)}.
    Valid for compound e-values (Ignatiadis, Wang & Ramdas 2024)."""
    e = np.asarray(e, float); m = len(e)
    es = np.sort(e)[::-1]
    ok = np.flatnonzero(es >= m / (q * np.arange(1, m + 1)))
    if len(ok) == 0:
        return np.zeros(m, bool)
    return e >= es[ok[-1]]


def _prune(R1, size):
    if not R1.any():
        return R1.copy()
    kk = np.sort(size[R1])
    r_ok = np.flatnonzero(np.arange(1, len(kk) + 1) >= kk)
    return R1 & (size <= r_ok[-1] + 1) if len(r_ok) else np.zeros(len(R1), bool)


def wcs(V_cal, w_cal, Vhat, w_test, q, rng=None):
    """Exact vectorised WCS (identical to the per-j loop; see sim/dbcp.wcs_fast)."""
    m = len(Vhat)
    p, base, W = weighted_pvalues(V_cal, w_cal, Vhat, w_test)
    cand = np.flatnonzero(p <= q)
    z = np.zeros(m, bool)
    if len(cand) == 0:
        return dict(R1=z, dtm=z, hete=z)
    to = np.argsort(Vhat); bs = base[to]
    pos = np.empty(m, int); pos[to] = np.arange(m)
    l = np.arange(m)
    a_up = q * (l + 1) / m; h = (bs - a_up * W) / (a_up - 1.0)
    SM = np.maximum.accumulate(h[::-1])[::-1]
    a_lo = q * (l + 2) / m; f = (bs - a_lo * W) / a_lo
    wj, pj = np.asarray(w_test)[cand], pos[cand]
    L = np.searchsorted(-SM, -wj, side="right") - 1
    ksz = np.full(m, m + 1); kc = np.ones(len(cand), int)
    up = L > pj; kc[up] = L[up] + 1
    for t in np.flatnonzero(~up):
        ok = np.flatnonzero(f[:pj[t]] <= wj[t])
        if len(ok):
            kc[t] = ok[-1] + 2
    ksz[cand] = kc
    R1 = p <= q * ksz / m
    out = dict(R1=R1, dtm=_prune(R1, ksz.astype(float)))
    if rng is not None:
        out["hete"] = _prune(R1, rng.random(m) * ksz)
    return out


# ------------------------------------------------------------------ O2: soft-rank e-values
def softrank_evalues(V_cal, w_cal, Vhat, w_test, tau, g_cal=None, g_test=None):
    """Soft-rank conformal e-values at a threshold tau that is FIXED given the training fold:
        e_j = (W + w_j) 1{Vhat_j < tau} / (w_j + sum_{i in C} w_i 1{V_i < tau}).
    For a null j (V_j <= Vhat_j by monotonicity) e_j <= the oracle bag e-value, whose
    Pi-average over the bag equals 1, so the e_j are compound e-values over T for ANY Pi
    (GeoCP_framework III(e'), proof in theory). With groups (Mondrian Pi) the sums run over the
    calibration units of the test unit's group."""
    V_cal = np.asarray(V_cal); w_cal = np.asarray(w_cal, float)
    Vhat = np.asarray(Vhat); w_test = np.asarray(w_test, float)
    e = np.zeros(len(Vhat))
    if g_cal is None:
        W = w_cal.sum(); D = (w_cal * (V_cal < tau)).sum()
        hit = Vhat < tau
        e[hit] = (W + w_test[hit]) / (w_test[hit] + D)
        return e
    for g in np.unique(g_test):
        a = g_cal == g; b = np.flatnonzero((g_test == g) & (Vhat < tau))
        W = w_cal[a].sum(); D = (w_cal[a] * (V_cal[a] < tau)).sum()
        e[b] = (W + w_test[b]) / (w_test[b] + D)
    return e


def choose_taus(V_train_oof, w_train, Vhat_nontrain, q, levels=(0.5, 0.8)):
    """Label-free, calibration-free thresholds: for each level l pick the largest tau with
        FDPhat(tau) = |U \\ D| * F0hat(tau) / #{u in U \\ D: Vhat_u < tau} <= l * q,
    where F0hat(tau) = sum_{i in D} w_i 1{V_i^oof < tau} / sum_{i in D} w_i uses only the
    training fold (cross-fitted scores) and the counts use Vhat of ALL non-training units
    (calibration and test alike) -> symmetric in the bag, fixed given D."""
    V_train_oof = np.asarray(V_train_oof); w_train = np.asarray(w_train, float)
    Vn = np.sort(np.asarray(Vhat_nontrain)); M = len(Vn)
    cand = np.unique(np.quantile(Vn, np.linspace(0.001, 0.6, 300)))
    o = np.argsort(V_train_oof); Vt = V_train_oof[o]; cw = np.r_[0, np.cumsum(w_train[o])] / w_train.sum()
    F0 = cw[np.searchsorted(Vt, cand, side="left")]
    R = np.searchsorted(Vn, cand, side="left")
    fdp = M * F0 / np.maximum(R, 1)
    taus = []
    for lv in levels:
        ok = np.flatnonzero(fdp <= lv * q)
        taus.append(cand[ok[-1]] if len(ok) else -np.inf)
    return np.array(taus)


def softrank_ebh(V_cal, w_cal, Vhat, w_test, q, taus, g_cal=None, g_test=None):
    """Deterministic screening: average the soft-rank e-values over the label-free thresholds
    (an average of compound e-values is a compound e-value) and run e-BH at level q."""
    taus = [t for t in np.atleast_1d(taus) if np.isfinite(t)]
    if not taus:
        return np.zeros(len(Vhat), bool)
    e = np.mean([softrank_evalues(V_cal, w_cal, Vhat, w_test, t, g_cal, g_test) for t in taus], axis=0)
    return ebh(e, q)
