"""Calibration on reference sets. An interval for a selected unit j is {y : score(x_j, y) <= q_j}; q_j may be +inf."""
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def _rank(N, alpha):
    tau = 1 - alpha
    return np.ceil(tau * (N + 1) - 1e-9 * (N + 1)).astype(int)         # k-th smallest; infinite when k > N


def _kth_two_sorted(a, b, k):
    """k-th smallest (1-indexed) of the union of two sorted arrays, without merging them."""
    lo, hi = max(0, k - len(b)), min(k, len(a))
    while lo < hi:
        i = (lo + hi) // 2
        if a[i] < b[k - i - 1]:
            lo = i + 1
        else:
            hi = i
    x = a[lo - 1] if lo > 0 else -np.inf
    y = b[k - lo - 1] if k - lo > 0 else -np.inf
    return max(x, y)


def jomi(scores_audit, sel, alpha=0.1):
    """Joint Mondrian conformal inference (Jin & Ren): the ceil((1-alpha)(|R_j|+1))-th smallest score in R_j, or +inf.
    Coverage is at least 1 - alpha among selected units, on average over audits."""
    S = np.asarray(scores_audit, float)
    q = np.full(len(sel), np.inf)
    sorted_sets = [np.sort(S[s]) for s in sel.sets]
    if sel.extras is None:
        for a, ss in enumerate(sorted_sets):
            k = int(_rank(len(ss), alpha))
            if k <= len(ss):
                q[sel.set_id == a] = ss[k - 1]
        return q
    for r in range(len(sel)):
        base = sorted_sets[sel.set_id[r]]
        ext = np.sort(S[sel.extras(r)])
        N = len(base) + len(ext); k = int(_rank(N, alpha))
        if k <= N:
            q[r] = _kth_two_sorted(base, ext, k)
    return q


def objective(beta, S, Phi, phi_j, tau, l1, pen):
    """The programme's objective at beta."""
    r = S - Phi @ beta
    return float(np.maximum(tau * r, (tau - 1) * r).sum() - tau * phi_j @ beta + l1 * np.abs(beta[pen]).sum())


def _primal_beta(S, Phi, phi_j, tau, l1, pen):
    n, p = Phi.shape
    k = int(pen.sum()) if l1 > 0 else 0
    Aeq = sparse.hstack([sparse.csr_matrix(Phi), sparse.csr_matrix((n, k)), sparse.eye(n), -sparse.eye(n)], format="csc")
    Aub = None; bub = None
    if k:
        P = sparse.csr_matrix((np.ones(k), (np.arange(k), np.flatnonzero(pen))), shape=(k, p))
        Z = sparse.csr_matrix((k, 2 * n)); I = sparse.eye(k)
        Aub = sparse.vstack([sparse.hstack([P, -I, Z]), sparse.hstack([-P, -I, Z])], format="csc"); bub = np.zeros(2 * k)
    c = np.r_[-tau * phi_j, np.full(k, l1), np.full(n, tau), np.full(n, 1 - tau)]
    bounds = [(None, None)] * p + [(0, None)] * (k + 2 * n)
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=S, bounds=bounds, method="highs")
    return r.x[:p] if r.status == 0 else None


def _dual_beta(S, Phi, phi_j, tau, l1, pen):
    """The same programme through its dual: p rows instead of n, so its size is set by the basis and not the reference set.
    max sum u_i S_i, u in [tau - 1, tau]^n, sum u_i Phi_i + tau phi_j = 0 unpenalized, |.| <= l1 penalized; beta = row duals."""
    n, p = Phi.shape
    if n < tau / (1 - tau) - 1e-12:                                     # Proposition 2: the primal is unbounded
        return None
    A = Phi.T; f = np.flatnonzero(~pen); g = np.flatnonzero(pen)
    r = linprog(-S, A_ub=np.vstack([A[g], -A[g]]) if len(g) else None,
                b_ub=np.r_[l1 - tau * phi_j[g], l1 + tau * phi_j[g]] if len(g) else None,
                A_eq=A[f] if len(f) else None, b_eq=-tau * phi_j[f] if len(f) else None,
                bounds=(tau - 1, tau), method="highs")
    if r.status != 0:
        warnings.warn(f"linear programme failed with HiGHS status {r.status}; returning an infinite interval")
        return None
    beta = np.zeros(p)
    if len(f):
        beta[f] = -r.eqlin.marginals                                    # beta = -(row duals of the minimization of -S'u)
    if len(g):
        d = r.ineqlin.marginals
        beta[g] = d[len(g):] - d[:len(g)]
    return beta


def _active_beta(S, Phi, phi_j, tau, l1, pen, beta0, free=None, max_rounds=30):
    """The dual solved on an active set, after Portnoy and Koenker (1997). In the dual, u_i sits at tau where the residual is
    positive and at tau - 1 where it is negative, so only units near the fitted surface are free. Fix the others by the signs of
    the residuals at a warm start, solve the small programme, and free any unit whose sign turns out wrong. When none is wrong
    the reduced solution satisfies the optimality conditions of the full programme, up to the 1e-9 sign check on fixed units and the solver's tolerance on free ones.
    Returns (beta, rounds, size of the last free set)."""
    n, p = Phi.shape
    if n < tau / (1 - tau) - 1e-12:
        return None, 0, 0
    nfree = min(n, max(200, 10 * p) if free is None else free)
    r0 = S - Phi @ beta0
    F = np.zeros(n, bool); F[np.argpartition(np.abs(r0), nfree - 1)[:nfree]] = True
    u = np.where(r0 > 0, tau, tau - 1.0)
    f = np.flatnonzero(~pen); g = np.flatnonzero(pen)
    for rounds in range(1, max_rounds + 1):
        fixed = Phi[~F].T @ u[~F] if (~F).any() else np.zeros(p)
        A = Phi[F].T; c = tau * phi_j + fixed
        r = linprog(-S[F], A_ub=np.vstack([A[g], -A[g]]) if len(g) else None,
                    b_ub=np.r_[l1 - c[g], l1 + c[g]] if len(g) else None,
                    A_eq=A[f] if len(f) else None, b_eq=-c[f] if len(f) else None, bounds=(tau - 1, tau), method="highs")
        if r.status != 0:                                               # these signs admit no feasible point: free more units
            if F.all():
                warnings.warn(f"linear programme failed with HiGHS status {r.status}; returning an infinite interval")
                return None, rounds, int(F.sum())
            k = min(n, 2 * int(F.sum()))
            F[np.argpartition(np.abs(S - Phi @ beta0), k - 1)[:k]] = True
            continue
        beta = np.zeros(p)
        if len(f):
            beta[f] = -r.eqlin.marginals
        if len(g):
            d = r.ineqlin.marginals
            beta[g] = d[len(g):] - d[:len(g)]
        res = S - Phi @ beta
        wrong = ~F & (((res > 1e-9) & (u < 0)) | ((res < -1e-9) & (u > 0)))
        if not wrong.any():
            return beta, rounds, int(F.sum())
        F |= wrong; u[wrong] = np.where(res[wrong] > 0, tau, tau - 1.0)
        beta0 = beta
    return _dual_beta(S, Phi, phi_j, tau, l1, pen), max_rounds, n       # never seen in the tests; the full dual is the fallback


def sc_geoccp(scores_audit, sel, Phi_audit, Phi_selected, alpha=0.1, l1=0.95, penalize=None, form="active"):
    """RegionCP (also exported as `regioncp`; SC-GeoCCP in earlier versions): per selected unit, the augmented quantile regression of Gibbs, Cherian and Candes on R_j over a basis fixed
    given the map, test score imputed at +inf, L1 penalty on the `penalize` columns (default: all but the first).

    form: "active" (default; warm-started active set, exact), "dual" or "primal" (full programmes, cold). Any minimizer carries
    the coverage floors of the manuscript's Theorem 1, so the forms share the guarantee; where the programme has several
    minimizers they may return different ones.

    Numerical safeguard: when every reference score of a selected unit is below 1e-6 (outcome and prediction both zero, so
    the scores are only a tie-breaking jitter), the programme is solved on the scores divided by their maximum and the
    threshold multiplied back; the programme is positively homogeneous in the scores, so the threshold is unchanged. Other
    reference sets take exactly the original path. The non-default form="primal" also relies on the LP solver's absolute
    tolerances for scores between 1e-6 and about 1e-4; use the default form there."""
    S = np.asarray(scores_audit, float); Pa = np.asarray(Phi_audit, float); Ps = np.asarray(Phi_selected, float)
    pen = np.r_[False, np.ones(Pa.shape[1] - 1, bool)] if penalize is None else np.asarray(penalize, bool)
    tau = 1 - alpha; q = np.empty(len(sel)); beta0 = None
    for r in range(len(sel)):
        idx = sel.reference(r)
        Sr = S[idx]
        sc = float(Sr.max()) if Sr.size else 0.0
        sc = sc if 0.0 < sc < 1e-6 else 1.0                    # safeguard (see docstring); 1.0 leaves the path unchanged
        Sr = Sr / sc
        if form == "active":
            if beta0 is None:                                  # warm start, kept in the original units
                beta0 = np.zeros(Pa.shape[1])
                if len(idx):
                    beta0[0] = np.quantile(S[idx], tau)
            b = _active_beta(Sr, Pa[idx], Ps[r], tau, l1, pen, beta0 / sc)[0]
            if b is not None:
                b = b * sc
                beta0 = b
        else:
            b = {"dual": _dual_beta, "primal": _primal_beta}[form](Sr, Pa[idx], Ps[r], tau, l1, pen)
            b = b * sc if b is not None else None
        q[r] = Ps[r] @ b if b is not None else np.inf
    return q
