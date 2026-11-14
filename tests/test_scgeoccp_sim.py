"""Monte Carlo check of SC-GeoCCP Theorem 1 (theory/SC_GeoCCP.md) on the fixed spatially
autocorrelated population of test_coverage_sim.py (only the calibration/test split is random).

    ~/miniforge3/envs/favela/bin/python tests/test_scgeoccp_sim.py [reps]

For top-K and state-quota selection and l1 in {1.0, 2.0}, with the multiscale bump basis
(+ 16 state indicators) built from the fixed training fold:
  (a) intercept (unpenalized): P(Y in C | selected) >= 1 - alpha
  (b) every basis direction f: E[f(X)(cov - (1-alpha)) | selected] >= -l1 E[1/|R u {j}|]
Reports the worst direction as a z-score (estimate - bound) / MC se; pass if >= -3.
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
from excp_select import refsets, basis, methods as Mt  # noqa: E402
from test_coverage_sim import population  # noqa: E402

ALPHA = 0.1; TAU = 1 - ALPHA


def main(R=600):
    s, y, g = population()
    N = len(y); rng0 = np.random.default_rng(7)
    perm = rng0.permutation(N); tr, rest = perm[:1000], perm[1000:]
    tree = cKDTree(s[tr])
    mu = y[tr][tree.query(s, k=15)[1]].mean(1) + 1e-9 * rng0.standard_normal(N)
    sig = np.maximum(np.abs(y[tr] - mu[tr])[tree.query(s, k=40)[1]].mean(1), 0.05)
    S = np.abs(y - mu) / sig
    B = basis.Multiscale(s[tr]).with_units(g); Phi = B(s, g); pen = B.penalize
    n_cal = 1000; m = len(rest) - n_cal
    L1 = [1.0, 2.0]
    acc = {}
    for r in range(R):
        rng = np.random.default_rng(5000 + r)
        p = rng.permutation(rest); C, T = p[:n_cal], p[n_cal:]
        sels = {"top-50": refsets.topk(mu[C], mu[T], 50),
                "quota 16x4": refsets.quota(mu[C], mu[T], g[C], g[T], {u: 4 for u in range(16)})}
        ctx = dict(S_cal=S[C], Phi_cal=Phi[C], Phi_test=Phi[T])
        for nm, sel in sels.items():
            J = T[sel.selected]; invN = 1.0 / (sel.sizes + 1)
            for l1 in L1:
                q = Mt.sc_geoccp(sel, l1, pen).thresholds(ctx, ALPHA)
                cov = (S[J] <= q).astype(float)
                a = acc.setdefault((nm, l1), dict(F=[], invN=[], fin=[]))
                a["F"].append(Phi[J] * (cov - TAU)[:, None]); a["invN"].append(invN)
                a["fin"].append(np.isfinite(q))
    print(f"population N={N}, n_cal={n_cal}, m={m}, reps={R}, basis p={Phi.shape[1]} "
          f"({(~pen).sum()} unpenalized)")
    bad = 0
    for (nm, l1), a in acc.items():
        Fm = np.vstack(a["F"]); invN = np.concatenate(a["invN"]); n = len(Fm)
        est = Fm.mean(0); se = Fm.std(0) / np.sqrt(n)      # pooled over units (conservative-ish se)
        bound = np.where(pen, -l1 * invN.mean(), 0.0)
        z = (est - bound) / np.maximum(se, 1e-12)
        cov = est[0] + TAU
        k = int(np.argmin(z))
        ok = z.min() >= -3
        bad += not ok
        print(f"{nm:11s} l1={l1:3.1f}  cov|sel={cov:.4f} (se {se[0]:.4f})  +inf={1-np.concatenate(a['fin']).mean():.4f}  "
              f"worst direction {B.names()[k]:14s} z={z.min():6.2f}  bound={bound[k]:.4f}  {'ok' if ok else 'FAIL'}")
    print("ALL SC-GeoCCP CHECKS PASS" if bad == 0 else f"{bad} FAILURES")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 600) else 1)
