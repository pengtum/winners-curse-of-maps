"""Carbon control: a domain (post-stratified) difference estimator (paper/CARBON_DOMAIN_DESIGN.md, written before the run).

The survey-calibrated series of experiments/run_carbon.py corrects the map total with the mean residual of ALL audit
plots. A practitioner would more likely correct within strata of the predicted value. Strata are G quantile bins of the
map's predictions on its own training fold (G = 10 main; 5 and 20 sensitivity), fixed before any audit is seen; strata
are merged downward from the top until each holds at least 10 audit plots. For the K selected plots,
  estimate = sum mu + sum_h K_h mean(e_audit,h),  SE = (sum_h s_h^2 (K_h + K_h^2 / n_h))^1/2,
i.e. the difference estimator within each stratum, again treating the selected residuals as fresh draws there. The
estimate goes through the same protocol rules as the survey-calibrated series (VCS in full, with the 10% waiver and the
U > 100% cutoff; ART TREES), plus the VCS formula without waiver or cutoff; credits are floored at 0.
Maps, folds, audits and selections are those of run_carbon.py (same seeds; mu is refitted and checked against the
stored 'point' and 'ma' columns of results/carbon_<map>[_random].csv.gz before anything is written). QRF is not refitted.
    ~/miniforge3/envs/favela/bin/python -u experiments/run_carbon_domain.py <folds> <splits> <workers> [random]
writes results/carbon_domain_<map>[_random].csv.gz (parts in results/cache/carbon_domain_parts/). With CARBON_OUT set (as for
run_carbon.py), the reference series are read from and the results written to that folder: the post-hoc 20-map random control
uses CARBON_OUT=results/carbon_random20.
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma                                     # noqa: E402
from excp_select import refsets                                       # noqa: E402
import run_carbon as RC                                               # noqa: E402

GS = (10, 5, 20)                      # strata: 10 main, 5 and 20 sensitivity
NMIN = 10                             # minimum audit plots per (merged) stratum
PARTS = os.path.join(DS.CACHE, f"carbon_domain_parts{DS.FIA_TAG}")
OUT = os.environ.get("CARBON_OUT", DS.FIA_RES)                         # reference series read from, results written to


def strata(mu, Dtr, G):
    """Stratum of every plot: quantile bins of the training-fold predictions (0 = lowest)."""
    edges = np.quantile(mu[Dtr], np.linspace(0, 1, G + 1)[1:-1])
    return np.searchsorted(edges, mu, side="right")


def merged(h, C, G):
    """Map strata to merged groups so that every group holds >= NMIN audit plots, merging downward from the top;
    a remainder at the bottom joins the group above it."""
    n = np.bincount(h[C], minlength=G); grp = np.empty(G, int); g, acc = 0, 0
    for k in range(G - 1, -1, -1):
        grp[k] = g; acc += n[k]
        if acc >= NMIN and k > 0:
            g, acc = g + 1, 0
    if acc < NMIN and g > 0:                       # bottom remainder too small: join the group above
        grp[grp == g] = g - 1
    return grp


def domain(mu, e, h, grp, J, C):
    """Post-stratified difference estimate and SE for the selected plots J with audit C."""
    gC, gJ = grp[h[C]], grp[h[J]]; ng = grp.max() + 1
    n = np.bincount(gC, minlength=ng); s1 = np.bincount(gC, e[C], ng); s2 = np.bincount(gC, e[C] ** 2, ng)
    K = np.bincount(gJ, minlength=ng); m = s1 / np.maximum(n, 1)
    var_e = np.where(n > 1, (s2 - n * m ** 2) / np.maximum(n - 1, 1), np.nan)
    use = K > 0
    est = mu[J].sum() + (K[use] * m[use]).sum()
    se = np.sqrt((var_e[use] * (K[use] + K[use] ** 2 / n[use])).sum())
    return float(est), float(se), int(n[use].sum())


def task(args):
    mp, f, ns, rnd = args
    tag = "_random" if rnd else ""
    part = os.path.join(PARTS, f"{RC.FILE[mp]}{tag}_f{f:02d}_s{ns}.csv.gz")
    if os.path.exists(part):
        return part
    t0 = time.time()
    d = DS.load(RC.DATA); y = d["y"]; N = d["N"]
    X = d["Xrf"][:, [c for c in range(d["Xrf"].shape[1]) if c not in RC.DROP[mp]]]
    perm = np.random.default_rng([5150, f]).permutation(N); ntr = int(0.4 * N)
    Dtr, rest = perm[:ntr], perm[ntr:]
    mu, _ = fit_mu_sigma(X, y, Dtr, np.random.default_rng([5151, f]))
    e = y - mu
    H = {G: strata(mu, Dtr, G) for G in GS}
    rows = []
    for s in range(ns):
        p = np.random.default_rng([5152, f, s]).permutation(rest)
        u = np.random.default_rng([5161, f, s]).random(N) if rnd else mu            # selection score, as run_carbon
        for a in RC.AUDIT:
            na = int(a * len(p)); C, T = p[:na], p[na:]
            grp = {G: merged(H[G], C, G) for G in GS}
            for frac in RC.FRACS:
                K = max(1, int(round(frac * len(T))))
                sel = refsets.topk(u[C], u[T], K)
                J = T[sel.selected]
                r = dict(map=mp, fold=f, split=s, audit=a, frac=frac, K=K, total=float(y[J].sum()),
                         point=float(mu[J].sum()), ma=float(mu[J].sum() + K * e[C].mean()))
                for G in GS:
                    est, se, nu = domain(mu, e, H[G], grp[G], J, C)
                    r.update({f"dom{G}_est": est, f"dom{G}_se": se, f"dom{G}_n": nu,
                              f"VCS domain{G}": max(0.0, RC.vcs(est, se)),
                              f"VCS domain{G}, formula": max(0.0, est - RC.Z_VCS * se),
                              f"ART domain{G}": max(0.0, RC.art(est, se))})
                rows.append(r)
    df = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(OUT, f"carbon_{RC.FILE[mp]}{tag}.csv.gz"), usecols=["fold", "split", "audit", "frac", "point", "ma"])
    ref = ref[ref.fold == f].merge(df[["split", "audit", "frac", "point", "ma"]], on=["split", "audit", "frac"], suffixes=("", "_new"))
    assert len(ref) == len(df), (mp, f, len(ref), len(df))
    for c in ("point", "ma"):                                    # stored with %.6g: relative agreement
        dev = np.abs(ref[c] - ref[c + "_new"]) / np.maximum(np.abs(ref[c]), 1e-9)
        assert dev.max() < 2e-5, (mp, f, c, float(dev.max()))
    os.makedirs(PARTS, exist_ok=True)
    df.to_csv(part, index=False, float_format="%.6g")
    print(f"  {mp}{tag} fold {f}: {time.time() - t0:.0f}s (point/ma reproduce)", flush=True)
    return part


def main():
    nf, ns, W = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rnd = len(sys.argv) > 4 and sys.argv[4] == "random"
    t0 = time.time()
    with Pool(W) as pool:
        parts = pool.map(task, [(mp, f, ns, rnd) for mp in RC.DROP for f in range(nf)], chunksize=1)
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    tag = "_random" if rnd else ""
    for mp, g in df.groupby("map", sort=False):
        g.to_csv(os.path.join(OUT, f"carbon_domain_{RC.FILE[mp]}{tag}.csv.gz"), index=False, float_format="%.6g")
    a = df[df.audit == 0.5]
    cols = ["VCS domain10", "VCS domain5", "VCS domain20", "VCS domain10, formula", "ART domain10"]
    risk = pd.DataFrame({m: (a[m] > a.total).groupby([a["map"], a.frac]).mean() for m in cols}).T
    pd.set_option("display.width", 250)
    print(f"\nrealized over-crediting risk{tag}, audit share 0.5 (columns: map, selected share)\n" + risk.round(3).to_string())
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
