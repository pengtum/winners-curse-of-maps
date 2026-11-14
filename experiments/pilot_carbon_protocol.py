"""Feasibility pilot: do carbon-protocol uncertainty deductions keep their DECLARED risk on
map-selected sites? (paper/POLICY_ANCHORS.md)

Protocol rules (verified against the original texts):
  VCS Methodology Requirements v5.0 sec. 2.4: if U = 1.6449 SE / estimate > 10%, credited =
      estimate x (1 - U / 1.6449 x 0.4307) = estimate - 0.4307 SE (one-sided 66.67%, i.e. a
      declared over-crediting risk of 1/3, Neeff 2021); credited = 0 if U > 100%.
  ART TREES v2.0 Eq. 10-11: credited = estimate - 0.524417 SE (t at the "ART allowable risk",
      the 0.70 quantile, i.e. a declared risk of 30%).
"Map": a random forest on SITE attributes only (lon, lat, elevation / slope class, aspect, heat
load, forest type, site class, physiographic class: no plot-structure variables BA, TPH, DIA, HT,
stand size, which a map of unvisited land would not have); "site+height" adds mean tree height as a
stand-in for lidar canopy height; "full" keeps all ground variables (the main benchmark setting). The developer's SE is the map's own per-site error layer,
s_j = sqrt(pi/2) sigma_j (sigma = training-fold |residual| model), combined either as
independent errors, SE = sqrt(sum s_j^2) (IPCC Approach 1 style), or fully correlated, SE = sum s_j.
Sites: the top-K predicted carbon units among the unvisited (test) plots. Truth: FIA plot biomass.
Ours at the SAME declared risk delta, calibrated on the random audit (calibration) plots:
exact pool-lemma total bound (excp_select.totals, support floor y >= 0) and the reference-set
SRSWOR CLT (difference estimator).
Design: <folds> training folds x <splits> random audit/test splits; audit share 0.5 or 0.8.
    ~/miniforge3/envs/favela/bin/python -u experiments/pilot_carbon_protocol.py <folds> <splits> <workers>
writes results/pilot_carbon_protocol.csv
"""
import os, sys, time
from functools import lru_cache
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma                                     # noqa: E402
from excp_select import refsets, totals as TOT                        # noqa: E402

OUT = os.path.join(ROOT, "results")
DROP = {"site": [6, 7, 8, 9, 11],     # Xrf columns of X5 BA, X6 TPH, X7 DIA, X8 HT, X10 stand size
        "site+height": [6, 7, 8, 11], "full": []}
DELTAS = {"VCS (1/3)": 1 / 3, "ART (0.30)": 0.30}
FRACS = (0.005, 0.01, 0.02, 0.05)
AUDIT = (0.5, 0.8)


@lru_cache(maxsize=None)
def env(L, K, delta):
    return tuple(int(c) for c in TOT.envelope_exact(L, K, delta))


def fold(args):
    features, f = args
    d = DS.load("fia"); X = d["Xrf"]
    X = X[:, [c for c in range(X.shape[1]) if c not in DROP[features]]]
    rng = np.random.default_rng([4343, f])
    perm = rng.permutation(d["N"]); ntr = int(0.4 * d["N"])
    mu, sig = fit_mu_sigma(X, d["y"], perm[:ntr], rng)
    r = perm[ntr:]; r2 = 1 - ((d["y"][r] - mu[r]) ** 2).sum() / ((d["y"][r] - d["y"][r].mean()) ** 2).sum()
    return dict(features=features, rest=r, mu=mu, sig=sig, r2=r2)


def deductions(est, s):
    out = {}
    for tag, se in (("", np.sqrt((s ** 2).sum())), (" [SE corr.]", s.sum())):
        U = 1.6449 * se / est
        out["VCS MR v5.0 deduction" + tag] = 0.0 if U > 1 else (est if U <= 0.10 else est - 0.4307 * se)
        out["ART TREES deduction" + tag] = est - 0.524417 * se
    return out, 1.6449 * np.sqrt((s ** 2).sum()) / est


def task(args):
    fd, f, splits = args
    y = DS.load("fia")["y"]; mu, sig = fd["mu"], fd["sig"]; e = y - mu
    rows = []
    for s in splits:
        rng = np.random.default_rng([88, f, s])
        p = rng.permutation(fd["rest"])
        for a in AUDIT:
            na = int(a * len(p)); C, T = p[:na], p[na:]
            for frac in FRACS:
                K = max(1, int(round(frac * len(T))))
                sel = refsets.topk(mu[C], mu[T], K)
                J = T[sel.selected]; R = C[sel.member[0]]; nR = len(R); L = nR + K
                tot, est = y[J].sum(), mu[J].sum()
                out, U = deductions(est, np.sqrt(np.pi / 2) * sig[J])
                out = {"Point estimate (no deduction)": est, **out}
                for tag, dlt in DELTAS.items():
                    out[f"Ours exact, risk {tag}"] = est + TOT.lower_total(e[R], K, -mu[J].max(), np.array(env(L, K, dlt)))
                    sR = e[R].std(ddof=1) if nR > 1 else np.inf
                    out[f"Ours CLT, risk {tag}"] = (est + K * e[R].mean()
                                                    - norm.ppf(1 - dlt) * K * sR * np.sqrt(L / (max(nR, 1) * K)))
                base = dict(features=fd["features"], fold=f, split=s, audit=a, frac=frac, K=K, nR=nR,
                            total=tot, est=est, U_vcs=U)
                for mth, cr in out.items():
                    rows.append(dict(base, method=mth, credited=float(cr), overcredit=bool(cr > tot)))
    return rows


def main():
    nf, ns, W = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    with Pool(W) as pool:
        folds = pool.map(fold, [(ft, f) for ft in DROP for f in range(nf)], chunksize=1)
        chunk = max(1, ns // 2)
        jobs = [(fd, i % nf, list(range(a, min(ns, a + chunk)))) for i, fd in enumerate(folds) for a in range(0, ns, chunk)]
        res = pool.map(task, jobs, chunksize=1)
    df = pd.DataFrame([r for x in res for r in x])
    df.to_csv(os.path.join(OUT, "pilot_carbon_protocol.csv"), index=False)
    s = df.groupby(["features", "audit", "frac", "method"]).agg(risk=("overcredit", "mean"), cr=("credited", "sum"),
                                                                 true=("total", "sum"))
    s["credited/true"] = s.cr / s.true
    pd.set_option("display.width", 200)
    print(s[["risk", "credited/true"]].unstack("frac").round(3).to_string())
    b = df[df.method == "Point estimate (no deduction)"].groupby(["features", "frac", "fold"]).apply(
        lambda g: g.est.sum() / g.total.sum()).unstack("fold")
    print("\nsum(mu)/sum(Y) of the selected, per training fold:\n" + b.round(3).to_string())
    r2 = pd.DataFrame([(fd["features"], fd["r2"]) for fd in folds], columns=["features", "R2"])
    print("\nheld-out R2 by map:", r2.groupby("features").R2.agg(["mean", "min", "max"]).round(3).to_string())
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
