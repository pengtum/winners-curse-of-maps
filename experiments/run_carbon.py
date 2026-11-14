"""WP3 carbon anchor (paper/POLICY_ANCHORS.md): do carbon-protocol uncertainty deductions keep their
DECLARED over-crediting risk when the credited sites are picked with the same map that quantifies them?

Protocol formulas (checked against the original texts, paper/POLICY_ANCHORS.md sec. 1):
  VCS Methodology Requirements v5.0 sec. 2.4: U = 1.6449 SE / estimate; U <= 10%: no deduction;
      U > 100%: not creditable; otherwise credited = estimate - 0.4307 SE (declared risk 1/3).
  ART TREES v2.0 Eq. 10-11: credited = estimate - 0.524417 SE (declared risk 0.30).
Maps (random forests on a fixed 40% training fold, run_main.fit_mu_sigma; 10,000 FIA plots):
  site         lon, lat, elevation / slope class, aspect, heat load, forest type, site class, physiographic class
  site+height  + mean tree height (stand-in for lidar canopy height)
  full         + basal area, trees / ha, mean DBH, stand-size class (ground-measured: the method benchmark,
               kept as the main FIA setting of the paper; not available for unvisited land)
Selection: top-K candidate plots by the map. Truth: FIA plot biomass. Audit: random plots with ground truth.
Developer estimates fed into the SAME protocol formulas:
  map            estimate = sum mu over the selected; SE from the map's error layer, either the RF |residual|
                 model (s = sqrt(pi/2) sigma) or QRF (Meinshausen 2006: 90% interval / 3.29), combined as
                 independent errors (sqrt sum s^2, IPCC Approach 1) or fully correlated (sum s)
  model-assisted estimate = sum mu + K mean(e_audit), SE = s_audit sqrt(K + K^2 / n_audit): survey-calibrated
                 difference estimator that treats the selected residuals as fresh draws (ignores selection)
  selection-aware (ours) estimate = sum mu + K mean(e_R), SE = K s_R sqrt(L / (n_R K)) on the JOMI reference set
                 R (pool lemma, theory/POOL_LEMMA.md T1(c)); the protocol formula without the 10% waiver is
                 the reference-set CLT bound at the declared risk; "+ waiver" keeps the VCS waiver
  exact (ours)   pool-lemma lower bound at the declared risk (excp_select.totals, support y >= 0; envelope from
                 the floating-point DP with 16 bisection steps: still exactly within delta, tests/test_pool_lemma.py 5)
Sensitivity (CARBON_DATA=fia_post2000, with CARBON_OUT / CARBON_PARTS set to other folders): the FIA plots
inventoried in 2000 or later only (experiments/data.py).
Control (CARBON_SELECT=random): the same K candidates chosen at random instead of by the map (random
scores redrawn per split), to separate the effect of map-based selection from that of the error layer.
All credited amounts are floored at 0. Design: <folds> training folds (= maps) per map type x <splits>
random audit / candidate splits x audit share 0.2 / 0.5 / 0.8 x top 0.5 / 1 / 2 / 5 / 10% of candidates.
    ~/miniforge3/envs/favela/bin/python -u experiments/run_carbon.py <folds> <splits> <workers> [maps...]
writes results/carbon_<map>[_random].csv.gz: one row per selection, one column per method (credited amount);
each (map, training fold) is first saved to results/cache/carbon_parts/ so a late failure loses nothing
and a rerun skips the finished parts
"""
import os, sys, time
from functools import lru_cache
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma                                     # noqa: E402
from run_baselines import qrf_bounds, Z                               # noqa: E402
from excp_select import refsets, totals as TOT                        # noqa: E402

OUT = os.environ.get("CARBON_OUT", DS.FIA_RES)                               # env: inherited by spawned workers
PARTS = os.environ.get("CARBON_PARTS", os.path.join(DS.CACHE, f"carbon_parts{DS.FIA_TAG}"))
RANDOM = os.environ.get("CARBON_SELECT", "map") == "random"
DATA = os.environ.get("CARBON_DATA", DS.FIA)                                # fia_post2000: sensitivity run
TAG = "_random" if RANDOM else ""
DROP = {"site": [6, 7, 8, 9, 11],            # Xrf columns of X5 BA, X6 TPH, X7 DIA, X8 HT, X10 stand size
        "site+height": [6, 7, 8, 11], "full": []}
FILE = {"site": "site", "site+height": "siteheight", "full": "full"}
AUDIT = (0.2, 0.5, 0.8)
FRACS = (0.005, 0.01, 0.02, 0.05, 0.10)
Z_VCS, Z_ART = 0.4307, 0.524417
DELTA = {"VCS": 1 / 3, "ART": 0.30}


def vcs(est, se):
    U = 1.6449 * se / est if est > 0 else np.inf
    return 0.0 if U > 1 else (est if U <= 0.10 else est - Z_VCS * se)


def art(est, se):
    return est - Z_ART * se


@lru_cache(maxsize=None)
def env(L, K, delta):
    return np.array(TOT.envelope_exact(L, K, delta, iters=16, fast=True))   # exact law, float DP (tests/test_pool_lemma.py 5)


def credits(mu, e, s_rf, s_qrf, J, R, C):
    """Credited amount per method (floored at 0) and the (estimate, SE) pairs behind them, so that
    Gaussian bounds at any other declared risk can be recomputed from the output."""
    K, nR = len(J), len(R); L = nR + K; est = mu[J].sum()
    se = {"rf": np.sqrt((s_rf[J] ** 2).sum()), "qrf": np.sqrt((s_qrf[J] ** 2).sum()), "rf_corr": s_rf[J].sum()}
    ma, se_ma = est + K * e[C].mean(), e[C].std(ddof=1) * np.sqrt(K + K ** 2 / len(C))
    sa, se_sa = (est + K * e[R].mean(), K * e[R].std(ddof=1) * np.sqrt(L / (nR * K))) if nR >= 2 else (0.0, np.inf)
    out = {"point": est,
           "VCS map (RF)": vcs(est, se["rf"]), "ART map (RF)": art(est, se["rf"]),
           "VCS map (QRF)": vcs(est, se["qrf"]), "ART map (QRF)": art(est, se["qrf"]),
           "VCS map (RF, corr.)": vcs(est, se["rf_corr"]), "ART map (RF, corr.)": art(est, se["rf_corr"]),
           "VCS model-assisted": vcs(ma, se_ma), "ART model-assisted": art(ma, se_ma),
           "ours: VCS formula, sel.-aware": sa - Z_VCS * se_sa, "ours: ART formula, sel.-aware": sa - Z_ART * se_sa,
           "ours: VCS formula + waiver": vcs(sa, se_sa) if nR >= 2 else 0.0}
    for p, d in DELTA.items():
        out[f"ours: exact, risk {p}"] = est + TOT.lower_total(e[R], K, -mu[J].max(), env(L, K, d))
    aux = dict(se_rf=se["rf"], se_qrf=se["qrf"], se_rf_corr=se["rf_corr"], ma=ma, se_ma=se_ma, sa=sa, se_sa=se_sa)
    return {k: max(0.0, float(v)) for k, v in out.items()}, {k: float(v) for k, v in aux.items()}


def task(args):
    mp, f, ns = args
    part = os.path.join(PARTS, f"{FILE[mp]}{TAG}_f{f:02d}_s{ns}.csv.gz")
    if os.path.exists(part):
        return part
    t0 = time.time()
    d = DS.load(DATA); y = d["y"]; N = d["N"]
    X = d["Xrf"][:, [c for c in range(d["Xrf"].shape[1]) if c not in DROP[mp]]]
    perm = np.random.default_rng([5150, f]).permutation(N); ntr = int(0.4 * N)
    Dtr, rest = perm[:ntr], perm[ntr:]
    mu, sig = fit_mu_sigma(X, y, Dtr, np.random.default_rng([5151, f]))
    lo, hi = qrf_bounds(X[Dtr], y[Dtr], X[rest], int(np.random.default_rng([5153, f]).integers(2 ** 31)))
    s_rf = np.sqrt(np.pi / 2) * sig
    s_qrf = np.zeros(N); s_qrf[rest] = (hi - lo) / (2 * Z)
    e = y - mu
    r2 = 1 - (e[rest] ** 2).sum() / ((y[rest] - y[rest].mean()) ** 2).sum()
    rows = []
    for s in range(ns):
        p = np.random.default_rng([5152, f, s]).permutation(rest)
        u = np.random.default_rng([5161, f, s]).random(N) if RANDOM else mu        # selection score
        for a in AUDIT:
            na = int(a * len(p)); C, T = p[:na], p[na:]
            for frac in FRACS:
                K = max(1, int(round(frac * len(T))))
                sel = refsets.topk(u[C], u[T], K)
                J = T[sel.selected]; R = C[sel.member[0]]
                cr, aux = credits(mu, e, s_rf, s_qrf, J, R, C)
                rows.append(dict(map=mp, fold=f, split=s, audit=a, frac=frac, K=K, nR=len(R), r2=r2,
                                 total=float(y[J].sum()), U_rf=1.6449 * aux["se_rf"] / cr["point"], **cr, **aux))
    pd.DataFrame(rows).to_csv(part, index=False, float_format="%.6g")
    print(f"  {mp} fold {f}: R2={r2:.3f}, {time.time()-t0:.0f}s", flush=True)
    return part


def main():
    nf, ns, W = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    maps = sys.argv[4:] or list(DROP)
    t0 = time.time(); os.makedirs(PARTS, exist_ok=True)
    with Pool(W) as pool:
        parts = pool.map(task, [(mp, f, ns) for mp in maps for f in range(nf)], chunksize=1)
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    for mp, g in df.groupby("map", sort=False):          # df.map is the DataFrame method, not the column
        g.to_csv(os.path.join(OUT, f"carbon_{FILE[mp]}{TAG}.csv.gz"), index=False, float_format="%.6g")
    meth = [c for c in df.columns if c == "point" or " map (" in c or "model-assisted" in c or c.startswith("ours")]
    a = df[df.audit == 0.5]
    risk = pd.DataFrame({m: (a[m] > a.total).groupby([a["map"], a.frac]).mean() for m in meth}).T
    pd.set_option("display.width", 250)
    print("\nrealized over-crediting risk, audit share 0.5 (columns: map, selected share)\n" + risk.round(3).to_string())
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
