"""The GeoCP family as parameter settings of one ExCP API.

Every preset returns a `Method` whose `.thresholds(ctx)` gives per-test thresholds q_j
(interval = {y: S(x_j, y) <= q_j}; for classification the same with candidate-label scores),
and whose `.guarantee` states what the framework can (or cannot) certify.

ctx is a dict with (only the keys a method needs):
  S_cal, w_cal, w_test            scores and exchange weights (weights.uniform / design_odds / ...)
  s_cal, s_test, s_train, w_train coordinates (projected) and training-fold design weights
  x_cal, x_test                   standardized attributes (GeoSIMCP / GWQR)
  g_cal, g_test                   groups / regimes / strata
  gamma_cal, gamma_test           (n, K) soft memberships (GROUND)
  Phi_cal, Phi_test               basis of the function class (GeoCCP)
  rng                             numpy Generator
"""
import numpy as np

from . import engines as E
from . import localizers as Lz


class Method:
    def __init__(self, name, fn, guarantee, theorem):
        self.name, self._fn, self.guarantee, self.theorem = name, fn, guarantee, theorem

    def thresholds(self, ctx, alpha):
        return self._fn(ctx, alpha)

    def __repr__(self):
        return f"<{self.name}: {self.guarantee} [{self.theorem}]>"


def _n(ctx):
    return len(ctx["S_cal"])


# ------------------------------------------------------------------ special cases (baselines)
def split_cp():
    return Method("split CP", lambda c, a: E.quantile(c["S_cal"], np.ones(_n(c)), np.ones(len(c["w_test"])), a),
                  "average/marginal coverage if Pi is uniform (random split, SRSWOR)", "Thm I")


def geocp_v1(h):
    """Published GeoCP (no atom; level ceil((1-a)(n+1))/n as in geocp 0.1.0 legacy mode)."""
    def fn(c, a):
        n = _n(c); lev = min(np.ceil((1 - a) * (n + 1)) / n, 1.0)
        return E.quantile(c["S_cal"], np.ones(n), np.ones(len(c["s_test"])), a,
                          Lz.test_centered(c["s_cal"], c["s_test"], h, atom=False), level=lev)
    return Method("GeoCP v1", fn, "none at the test point", "Thm IV (test-centred kernel)")


def geocp_v2(h):
    def fn(c, a):
        return E.quantile(c["S_cal"], np.ones(_n(c)), np.ones(len(c["s_test"])), a,
                          Lz.test_centered(c["s_cal"], c["s_test"], h, atom=True))
    return Method("GeoCP v2", fn, "coverage of the kernel-smoothed synthetic target only (baseLCP)",
                  "Thm IV")


def geosimcp(lam, h, atom=False):
    def fn(c, a):
        return E.quantile(c["S_cal"], np.ones(_n(c)), np.ones(len(c["s_test"])), a,
                          Lz.joint_test_centered(c["s_cal"], c["x_cal"], c["s_test"], c["x_test"], lam, h, atom))
    return Method("GeoSIMCP" + ("+atom" if atom else ""), fn, "none at the test point", "Thm IV")


def lscp(k):
    def fn(c, a):
        return E.quantile(c["S_cal"], np.ones(_n(c)), np.ones(len(c["s_test"])), a,
                          Lz.knn_test_centered(c["s_cal"], c["s_test"], k))
    return Method("LSCP (kNN)", fn, "asymptotic only (stationary mixing, infill)", "R axis")


def gwqr(k=50):
    def fn(c, a):
        return E.local_qr(c["S_cal"], c["s_cal"], c["x_cal"], c["s_test"], c["x_test"], a, k)
    return Method("GWQR-CP (local QR)", fn, "none (no calibration step)", "Thm IV")


def mondrian():
    def fn(c, a):
        return E.quantile(c["S_cal"], c["w_cal"], c["w_test"], a, Lz.mondrian(c["g_cal"], c["g_test"]))
    return Method("Mondrian / regime CP", fn, "per-group coverage", "III(b)")


def ground(p=1.0, tau=0.1, h=None):
    """GROUND core in ExCP terms: per component k a gamma_k^p tilt (x optional RLCP kernel),
    max over active components (gamma_k(test) >= tau, plus the argmax). Hard memberships give
    exactly Mondrian; soft memberships give gamma-tilted coverage; L3 is NOT certified."""
    def fn(c, a):
        G_c, G_t = c["gamma_cal"], c["gamma_test"]
        K = G_c.shape[1]; qs = []
        for k in range(K):
            loc = Lz.soft_tilt(G_c[:, k], G_t[:, k], p)
            if h is not None:
                loc = Lz.compose(loc, Lz.rlcp(c["s_cal"], c["s_test"], h, c["rng"]))
            qk = E.quantile(c["S_cal"], c["w_cal"], c["w_test"], a, loc)
            active = (G_t[:, k] >= tau) | (G_t.argmax(1) == k)
            qs.append(np.where(active, qk, -np.inf))
        return E.union_max(*qs)
    return Method("GROUND (max over regimes)", fn,
                  "per-regime coverage for hard memberships; gamma-tilted for soft; L3 open", "III(b)+III(c)")


def geocrc(loss_cal, lam_grid, B=1.0):
    """GeoCRC: regional (Mondrian) conformal risk control; loss_cal (n, G) monotone losses."""
    def fn(c, a):
        out = np.empty(len(c["g_test"]))
        for g in np.unique(c["g_test"]):
            ia = c["g_cal"] == g; ib = c["g_test"] == g
            out[ib] = E.crc(loss_cal[ia], c["w_cal"][ia], c["w_test"][ib], a, lam_grid, B)
        return out
    return Method("GeoCRC", fn, "per-region risk <= alpha", "III(b)+III(e)")


def wbcp(beta=0.9, draws=100):
    def fn(c, a):
        return E.bayes_layer(c["S_cal"], c["w_cal"], c["w_test"], a, beta, None, draws, c.get("rng"))
    return Method("WBCP layer", fn, "marginal (kept) + beta-credible data-conditional threshold",
                  "Thm I + D axis")


# ------------------------------------------------------------------ ExCP methods
def excp_design():
    def fn(c, a):
        return E.quantile(c["S_cal"], c["w_cal"], c["w_test"], a)
    return Method("ExCP-Design", fn, "average coverage over all unsampled units", "Thm I / M1-T1")


def excp_rlcp(h):
    def fn(c, a):
        return E.quantile(c["S_cal"], c["w_cal"], c["w_test"], a,
                          Lz.rlcp(c["s_cal"], c["s_test"], h, c["rng"]))
    return Method("ExCP-RLCP", fn, "average coverage + kernel-neighbourhood tilt", "III(a)")


def excp_default(kappa):
    """ExCP default = Pi (design or uniform) x RLCP+ (design-weighted kNN bandwidth) x any score
    (use a studentized score). Add groups to compose with Mondrian."""
    def fn(c, a):
        loc = None
        if kappa is not None:            # kappa=None: localization not affordable -> global
            loc = Lz.rlcp_plus(c["s_cal"], c["s_test"], c["s_train"], c["w_cal"], c["w_test"],
                               c["w_train"], kappa, c["rng"])
        if c.get("g_cal") is not None:
            mon = Lz.mondrian(c["g_cal"], c["g_test"])
            loc = mon if loc is None else Lz.compose(mon, loc)
        return E.quantile(c["S_cal"], c["w_cal"], c["w_test"], a, loc)
    return Method("ExCP-default (RLCP+)", fn, "average coverage + adaptive local tilt", "III(a), R0")


def excp_geoccp(l1=0.0):
    def fn(c, a):
        pen = np.ones(c["Phi_cal"].shape[1], bool); pen[0] = False
        return E.fclass(c["S_cal"], c["w_cal"], c["Phi_cal"], c["w_test"], c["Phi_test"], a,
                        l1=l1, penalize=pen)
    return Method("ExCP-GeoCCP", fn, "L2: coverage for every nonnegative basis direction", "III(d)")


TABLE = [
    # (family member, ExCP parameter setting, guarantee)
    ("split CP", "Pi uniform, no localizer, constant engine", "marginal / average coverage (Thm I)"),
    ("GeoCP v1", "test_centered(h, atom=False), level ceil((1-a)(n+1))/n", "none at the test point (Thm IV)"),
    ("GeoCP v2", "test_centered(h, atom=True)", "synthetic kernel target only (baseLCP; Thm IV)"),
    ("GeoSIMCP", "joint_test_centered(lam, h), no atom", "none (Thm IV)"),
    ("GeoXCP", "multi_max score on explanation targets + any localizer", "as the localizer (needs SHAP targets)"),
    ("LSCP", "knn_test_centered(k)", "asymptotic (R axis)"),
    ("GWQR-CP", "local_qr engine", "none"),
    ("Mondrian / regime CP", "mondrian(groups)", "per-group (III(b))"),
    ("GROUND", "soft_tilt(gamma_k^p) [x rlcp] per k, union_max over active k", "per-regime (hard) / tilted (soft); L3 open"),
    ("GeoCRC", "mondrian + crc engine", "per-region risk (III(b)+(e))"),
    ("WBCP", "covariate_shift(w_hat) weights + crc / bayes_layer", "marginal if w_hat = w; gap Thm IV; beta-credible layer"),
    ("GeoCCP", "fclass engine (basis Phi, optional l1)", "L2 (III(d))"),
    ("RLCP", "rlcp(h)", "marginal + local tilt (III(a))"),
    ("M1 design-based", "design_odds(pi) / cps_odds(ptil) weights", "average over all unsampled units (M1-T1)"),
    ("ExCP default", "Pi x rlcp_plus(kappa) x studentized score", "average + adaptive local tilt (III(a), R0)"),
    ("classification (LAC/APS)", "lac/aps scores + any of the above", "same as the chosen Pi / localizer"),
    ("screening", "softrank_ebh (deterministic) / wcs / bh", "FDR <= q over all unsampled units (M1-T4, III(e'))"),
]
