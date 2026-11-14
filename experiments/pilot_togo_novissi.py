"""Feasibility pilot: a Novissi-style geographic targeting rule on Togo DHS clusters
(paper/POLICY_ANCHORS.md, plan A). A stylised cluster-level analogue, NOT a replication of Novissi.

Novissi rural phase (Aiken et al., Nature 2022; NBER w29070): the 100 poorest of 397 cantons by the
Chi et al. relative-wealth map were prioritised, then funds reached the poorest 29% of registrants.
  units   the 330 Togo DHS 2013-14 clusters of PovertyMap v1.1; truth = cluster mean asset index
  map     random forest on the other 22 countries only (leave-Togo-out = an external map); sigma from
          training out-of-fold |residuals| (run_main.fit_mu_sigma hyperparameters); S = |Y - mu| / sigma
  audit   a random half of Togo's clusters is surveyed (calibration); the rest are mapped only (test)
  rule    stage 1 = poorest ceil(0.252 m) test clusters by mu (100/397); stage 2 = poorest 29% of those
          by the same map (equals a bottom-K2 rule)
  report  coverage / width / +inf share of 90% intervals on the selected (the map's own error layer
          mu +- 1.645 sqrt(pi/2) sigma, split CP on the audit, BY, JOMI, JOMI | urban), selection bias
          mean(Y - mu), cfBH certification of "eligible" (Y < c; c = 25.2% quantile of mu over all
          Togo clusters: covariates only, fixed before any label is seen) and a 90% upper bound on the
          mean wealth of the stage-1 set (survey-corrected map with margin z s sqrt(K + K^2/n), corrected on
          2026-11-10 from z s sqrt(K); reference-set CLT; exact pool-lemma bound with support y <= max
          training wealth).
    ~/miniforge3/envs/favela/bin/python -u experiments/pilot_togo_novissi.py <splits> [poverty|poverty_lab]
writes results/pilot_togo_<features>.csv; caches the map in results/cache/togo_map_<features>.npz
"""
import os, sys, time
from math import ceil
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_cfbh import bh, cf_pvalues                                   # noqa: E402
from excp import engines as E                                         # noqa: E402
from excp_select import refsets, totals as TOT, methods as Mt         # noqa: E402

ALPHA, DELTA, QS = 0.1, 0.1, (0.1, 0.2)
OUT = os.path.join(ROOT, "results")
TABLE = os.path.join(ROOT, "data", "povertymap_clusters.csv")


def rf(X, y, seed, n=100, leaf=3):
    return RandomForestRegressor(n_estimators=n, min_samples_leaf=leaf, max_features=0.6, n_jobs=4,
                                 random_state=seed).fit(X, y)


def togo_map(features):
    f = os.path.join(DS.CACHE, f"togo_map_{features}.npz")
    if os.path.exists(f):
        return dict(np.load(f))
    d = pd.read_csv(TABLE)
    img = [c for c in d.columns if c.startswith("b") and "_" in c]
    X = np.c_[d.lon, d.lat, d.urban.astype(float), d.year, d.nl_mean, d.nl_center]
    if features == "poverty":
        X = np.c_[X, d[img].values]
    y = d.wealthpooled.values.astype(float); tg = (d.country == "togo").values
    tr = np.flatnonzero(~tg); rng = np.random.default_rng(31)
    mu = rf(X[tr], y[tr], int(rng.integers(2 ** 31))).predict(X[tg])       # == run_main.fit_mu_sigma, 4 threads
    perm = rng.permutation(tr); a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    oof = np.r_[np.abs(y[a] - rf(X[b], y[b], 1, 60).predict(X[a])), np.abs(y[b] - rf(X[a], y[a], 2, 60).predict(X[b]))]
    sig = rf(X[np.r_[a, b]], oof, 3, 60, 10).predict(X[tg])
    out = dict(mu=mu + 1e-9 * rng.standard_normal(tg.sum()), sig=np.maximum(sig, 0.1 * oof.mean()), y=y[tg],
               urban=d.urban.values[tg].astype(int), y_sup=np.array(y[tr].max()))
    np.savez_compressed(f, **out)
    return out


def split(m, rep):
    mu, sig, y, urb = m["mu"], m["sig"], m["y"], m["urban"]; n = len(y)
    rng = np.random.default_rng([2026, rep])
    p = rng.permutation(n); C, T = p[: n // 2], p[n // 2:]
    S = np.abs(y - mu) / sig; e = y - mu
    c = np.quantile(mu, 0.252); poor = y <= np.quantile(y, 0.252)          # poor: evaluation only
    ctx = dict(S_cal=S[C], w_cal=np.ones(len(C)), w_test=np.ones(len(T)), g_cal=urb[C], g_test=urb[T])
    K1 = ceil(0.252 * len(T)); rows = []
    for stage, K in (("stage 1 (100/397)", K1), ("stage 2 (29% of stage 1)", ceil(0.29 * K1))):
        sel = refsets.topk(mu[C], mu[T], K, largest=False)
        J = T[sel.selected]; R = C[sel.member[0]]; nR = len(R); L = nR + K
        q = {"map error layer (Gaussian)": np.full(K, norm.ppf(1 - ALPHA / 2) * np.sqrt(np.pi / 2)),
             "split CP on audit (marginal)": np.full(K, E.quantile(S[C], np.ones(len(C)), np.ones(1), ALPHA)[0]),
             "BY split CP": Mt.by(sel, len(T)).thresholds(ctx, ALPHA),
             "JOMI": Mt.jomi(sel).thresholds(ctx, ALPHA),
             "JOMI | urban": Mt.jomi(sel, by_group=True).thresholds(ctx, ALPHA)}
        base = dict(rep=rep, stage=stage, K=K, nR=nR, bias=float(e[J].mean()), bias_all_test=float(e[T].mean()),
                    claimed=float(mu[J].mean()), truth=float(y[J].mean()), prec_poor=float(poor[J].mean()),
                    elig=float((y[J] < c).mean()), rural=float(1 - urb[J].mean()))
        for mth, qq in q.items():
            cov = S[J] <= qq
            rows.append(dict(base, method=mth, cover=float(cov.mean()), width=float(np.mean(2 * np.where(np.isfinite(qq), qq, np.nan) * sig[J])),
                             inf=float(np.mean(~np.isfinite(qq))), cover_rural=float(cov[urb[J] == 0].mean()) if (urb[J] == 0).any() else np.nan,
                             cover_urban=float(cov[urb[J] == 1].mean()) if (urb[J] == 1).any() else np.nan))
        if stage.startswith("stage 1"):
            sR = e[R].std(ddof=1)
            # margin: the selected clusters' own errors (K s^2) plus the error of the survey mean (K^2 s^2 / n), as in run_ood.py;
            # before 2026-11-10 it was z s sqrt(K), which left out the second term
            ub = {"marginal audit correction": mu[J].sum() + K * e[C].mean()
                  + norm.ppf(1 - DELTA) * np.sqrt(K + K ** 2 / len(C)) * e[C].std(ddof=1),
                  "reference-set CLT": mu[J].sum() + K * e[R].mean() + norm.ppf(1 - DELTA) * K * sR * np.sqrt(L / (nR * K)),
                  "exact pool bound": mu[J].sum() - TOT.lower_total(-e[R], K, mu[J].min() - float(m["y_sup"]),
                                                                    np.array(TOT.envelope_exact(L, K, DELTA)))}
            for mth, u in ub.items():
                rows.append(dict(base, method="mean-wealth bound: " + mth, cover=float(y[J].sum() <= u),
                                 width=float(u / K - y[J].mean())))
            g = (mu - c) / sig; cc = np.full(len(y), c)
            pv = cf_pvalues(g, y, cc, sel.member, J, C)
            for qv in QS:
                rej = bh(pv, qv); nf = int(rej.sum())
                rows.append(dict(base, method=f"cfBH eligible q={qv}", cover=np.nan, n_cert=nf,
                                 fdp=float((rej & (y[J] >= c)).sum() / max(nf, 1)),
                                 power=float((rej & (y[J] < c)).sum() / max((y[J] < c).sum(), 1))))
    return rows


def main():
    ns = int(sys.argv[1]); feats = sys.argv[2] if len(sys.argv) > 2 else "poverty"
    t0 = time.time(); m = togo_map(feats)
    r2 = 1 - ((m["y"] - m["mu"]) ** 2).sum() / ((m["y"] - m["y"].mean()) ** 2).sum()
    print(f"{feats}: Togo n={len(m['y'])}, leave-Togo-out R2={r2:.3f}, corr={np.corrcoef(m['y'], m['mu'])[0, 1]:.3f}, "
          f"mean(Y-mu)={np.mean(m['y'] - m['mu']):.3f}, share Y<c={np.mean(m['y'] < np.quantile(m['mu'], 0.252)):.3f} "
          f"[map {time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame([r for rep in range(ns) for r in split(m, rep)])
    df.to_csv(os.path.join(OUT, f"pilot_togo_{feats}.csv"), index=False)
    pd.set_option("display.width", 220)
    iv = df[~df.method.str.startswith(("mean-wealth", "cfBH"))]
    print(iv.groupby(["stage", "method"])[["cover", "cover_rural", "cover_urban", "width", "inf"]].mean().round(3).to_string())
    print(df.groupby("stage")[["K", "nR", "bias", "bias_all_test", "claimed", "truth", "prec_poor", "elig", "rural"]].mean().round(3).to_string())
    tb = df[df.method.str.startswith("mean-wealth")]
    print(tb.groupby("method")[["cover", "width"]].mean().round(3).rename(columns={"width": "bound - truth"}).to_string())
    cf = df[df.method.str.startswith("cfBH")]
    print(cf.groupby("method")[["n_cert", "fdp", "power"]].mean().round(3).to_string(), f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
