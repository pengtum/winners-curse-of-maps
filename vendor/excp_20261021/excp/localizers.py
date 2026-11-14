"""Localizers L: per-test multiplicative factors on the calibration weights and on the test
atom. In ExCP terms (theory/GeoCP_framework.md) a localizer either

* is part of the randomness R (random localization, III(a)) or a conditioning (III(b)) or a
  target tilt (III(c)) -> covered by Theorem I, or
* is centred at the test point itself (GeoCP v1/v2, GeoSIMCP, LSCP) -> NOT an exchange
  distribution; kept as faithful baselines, gap given by Theorem IV.

A Localizer exposes fac(sl) -> (len(sl), n_cal) and atom(sl) -> (len(sl),), sl a slice of the
test units, so engines can stream over large test sets.
"""
import numpy as np
from scipy.spatial import cKDTree

VALID = "valid (Theorem I)"
BASELINE = "no guarantee at the test point (Theorem IV gap)"


def sqd(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


class Localizer:
    def __init__(self, fac, atom, status, name):
        self.fac, self.atom, self.status, self.name = fac, atom, status, name

    def __repr__(self):
        return f"<Localizer {self.name}: {self.status}>"


def none(n_cal):
    return Localizer(lambda sl: np.ones((sl.stop - sl.start, n_cal)),
                     lambda sl: np.ones(sl.stop - sl.start), VALID, "none")


# ------------------------------------------------------------------ valid localizers
def _logphi(D2, h):
    return -np.log(2 * np.pi * h ** 2) - 0.5 * D2 / h ** 2


def rlcp(s_cal, s_test, h, rng, h_cal=None, h_test=None):
    """Randomly-localized kernel (Hore & Barber; ExCP III(a)): draw s~_j = s_j + h_j N(0, I_2);
    calibration factor phi_{h_i}(s~_j - s_i), atom phi_{h_j}(s~_j - s_j), phi the NORMALISED
    2-D Gaussian density (needed when h varies by unit). h may be a scalar or, via
    h_cal/h_test, per-unit bandwidths that are fixed given the training fold (RLCP+)."""
    s_cal = np.asarray(s_cal, float); s_test = np.asarray(s_test, float)
    hc = np.full(len(s_cal), float(h)) if h_cal is None else np.asarray(h_cal, float)
    ht = np.full(len(s_test), float(h)) if h_test is None else np.asarray(h_test, float)
    xi = rng.standard_normal(s_test.shape)
    stil = s_test + ht[:, None] * xi
    # a common per-row factor cancels in the weighted quantile; subtract the atom's log-density
    # (and multiply design weights later) to keep exp() in floating-point range
    ref = _logphi((ht ** 2) * (xi ** 2).sum(1), ht)

    def fac(sl):
        return np.exp(_logphi(sqd(stil[sl], s_cal), hc[None, :]) - ref[sl, None])

    def atom(sl):
        return np.ones(sl.stop - sl.start)          # = exp(ref - ref)

    return Localizer(fac, atom, VALID, "rlcp")


def knn_bandwidth(ref, w_ref, query, w_query, kappa, kmax=1024, floor=1e-9):
    """RLCP+ bandwidth (theory/M1_rlcp_plus.md, R2): h(s_k) = smallest radius holding
    kappa * w_k of the TRAINING-FOLD design weight. Depends on unit k only through (s_k, w_k)
    and on the training fold -> fixed given D, so Theorem R0 applies."""
    ref = np.asarray(ref, float); query = np.asarray(query, float)
    kmax = int(min(kmax, len(ref)))
    dist, idx = cKDTree(ref).query(query, k=kmax)
    dist = dist.reshape(len(query), kmax); idx = idx.reshape(len(query), kmax)
    cw = np.cumsum(np.asarray(w_ref)[idx], axis=1)
    pos = (cw < (kappa * np.asarray(w_query))[:, None]).sum(1)
    return np.maximum(dist[np.arange(len(pos)), np.minimum(pos, kmax - 1)], floor)


def rlcp_plus(s_cal, s_test, s_train, w_cal, w_test, w_train, kappa, rng, floor=1e-9):
    """RLCP+ localizer: rlcp with design-weighted kNN bandwidths from the training fold."""
    hc = knn_bandwidth(s_train, w_train, s_cal, w_cal, kappa, floor=floor)
    ht = knn_bandwidth(s_train, w_train, s_test, w_test, kappa, floor=floor)
    loc = rlcp(s_cal, s_test, 1.0, rng, h_cal=hc, h_test=ht)
    loc.name = "rlcp_plus"
    return loc


def mondrian(g_cal, g_test):
    """Condition on the test unit's group (III(b)): Mondrian / regime / stratum."""
    g_cal = np.asarray(g_cal); g_test = np.asarray(g_test)
    return Localizer(lambda sl: (g_cal[None, :] == g_test[sl][:, None]).astype(float),
                     lambda sl: np.ones(sl.stop - sl.start), VALID, "mondrian")


def soft_tilt(gamma_cal, gamma_test, p=1.0):
    """Target tilt by soft membership gamma^p (III(c)): coverage for the gamma-tilted target
    (GROUND's per-component weights). gamma_* are (n, ) arrays for ONE component."""
    gc_ = np.asarray(gamma_cal, float) ** p; gt = np.asarray(gamma_test, float) ** p
    return Localizer(lambda sl: np.broadcast_to(gc_, (sl.stop - sl.start, len(gc_))),
                     lambda sl: gt[sl], VALID + " for the gamma-tilted target", "soft_tilt")


def compose(*locs):
    """Product rule (III(a)): independent localizers multiply."""
    st = VALID if all(l.status.startswith("valid") for l in locs) else BASELINE
    return Localizer(lambda sl: np.prod([l.fac(sl) for l in locs], axis=0),
                     lambda sl: np.prod([l.atom(sl) for l in locs], axis=0), st,
                     "*".join(l.name for l in locs))


# ------------------------------------------------------------------ test-centred baselines
def gauss(D2, h):
    return np.exp(-0.5 * D2 / h ** 2)


def test_centered(s_cal, s_test, h, atom=True):
    """GeoCP kernel K_h(s_i, s_j) centred at the test point; atom K(0)=1 (v2) or none (v1)."""
    return Localizer(lambda sl: gauss(sqd(s_test[sl], s_cal), h),
                     (lambda sl: np.ones(sl.stop - sl.start)) if atom else
                     (lambda sl: np.zeros(sl.stop - sl.start)), BASELINE,
                     "geocp_v2_kernel" if atom else "geocp_v1_kernel")


def joint_test_centered(s_cal, x_cal, s_test, x_test, lam, h, atom=False):
    """GeoSIMCP joint kernel exp(-0.5 (lam d_geo^2 + (1-lam) d_feat^2) / h^2) centred at the test
    point (original code: no atom; d_feat += 1e-8 is immaterial)."""
    return Localizer(lambda sl: gauss(lam * sqd(s_test[sl], s_cal) + (1 - lam) * sqd(x_test[sl], x_cal), h),
                     (lambda sl: np.ones(sl.stop - sl.start)) if atom else
                     (lambda sl: np.zeros(sl.stop - sl.start)), BASELINE, "geosimcp_kernel")


def knn_test_centered(s_cal, s_test, k):
    """LSCP (Mao, Martin & Reich): the k nearest calibration points of the test location,
    equal weights, no atom."""
    tree = cKDTree(np.asarray(s_cal, float))

    def fac(sl):
        _, idx = tree.query(np.asarray(s_test, float)[sl], k=k)
        F = np.zeros((sl.stop - sl.start, len(s_cal)))
        F[np.arange(F.shape[0])[:, None], idx.reshape(F.shape[0], -1)] = 1.0
        return F
    return Localizer(fac, lambda sl: np.zeros(sl.stop - sl.start), BASELINE, "lscp_knn")
