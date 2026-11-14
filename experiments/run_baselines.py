"""WP1: published baselines on the SAME splits / predictions / selections as run_main.py.

Same seeds as run_main.py (mu, sigma, Dtr/C/T and every selected set are identical; the mask
bandwidths h2/base are read from results/main_<ds>.csv). Methods (all published; evaluated on
the selected units, alpha = 0.1, two-sided):
  GeoCP (Lou et al. 2025)        published version (no test atom), |res| score, h tuned on the
                                  training fold (interval score)
  CQR (Romano et al. 2019)        HistGradientBoosting quantile models (training fold) + split CP
  QRF (Meinshausen 2006)          quantile regression forest on the training fold
  Regression kriging              RF trend + GP (Matern + white noise) on out-of-fold training
    (Hengl et al. 2007)           residuals; Gaussian predictive interval
  Bootstrap ensemble + residual   RF tree spread + OOF residual variance, Gaussian interval
    (Heskes 1997)                 ["Bootstrap ensemble only" = tree spread alone, for the SI]
  LSCP (Mao, Martin & Reich)      spatial kNN local conformal, k tuned on the training fold (SI)
  RLCP (Hore & Barber 2025)       randomly-localized CP, |res| score, h tuned on training (SI)
Outputs results/bl_main_<ds>.csv and results/bl_groups_<ds>.csv (pooled counts per state/cell).

    ~/miniforge3/envs/favela/bin/python -u experiments/run_baselines.py <R> <workers> [datasets...]
"""
import os, sys, time, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                      # noqa: E402
from run_main import fit_mu_sigma, quotas, rf, iscore, FRACS           # noqa: E402
from run_scgeoccp import cells                                          # noqa: E402
from excp_select import refsets                                         # noqa: E402
from excp import family as F                                            # noqa: E402

ALPHA = 0.1
Z = 1.6448536269514722                                                  # two-sided 90%
OUT = os.path.join(ROOT, "results")
_BW = {}


def bw_of(name, rep):
    if name not in _BW:
        m = pd.read_csv(os.path.join(OUT, f"main_{name}.csv"), usecols=["rep", "h2", "base"]).drop_duplicates("rep")
        _BW[name] = {int(r): (h, b) for r, h, b in zip(m.rep, m.h2, m.base)}
    return _BW[name][rep]


def q_conf(scores, alpha=ALPHA):
    n = len(scores); k = int(np.ceil((1 - alpha) * (n + 1)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def tune_local(X, y, s, Dtr, rng):
    """Training fold only: GeoCP (published) bandwidth, LSCP k, RLCP h by interval score."""
    perm = rng.permutation(Dtr); k = int(0.6 * len(perm)); D1, D2 = perm[:k], perm[k:]
    R2 = np.abs(y[D2] - rf(X[D1], y[D1], 5, 60).predict(X[D2]))
    base = float(np.median(cKDTree(s[Dtr]).query(s[Dtr], k=min(11, len(Dtr)))[0][:, -1]))
    halves = [np.random.default_rng(r).permutation(len(D2)) for r in range(2)]

    def ev(meth):
        sc, inf = [], []
        for p in halves:
            a, b = p[: len(p) // 2], p[len(p) // 2:]
            c = dict(S_cal=R2[a], w_cal=np.ones(len(a)), w_test=np.ones(len(b)), s_cal=s[D2][a], s_test=s[D2][b],
                     rng=np.random.default_rng(9))
            q = meth.thresholds(c, ALPHA); sc.append(iscore(R2[b], q)); inf.append(np.mean(~np.isfinite(q)))
        return np.mean(sc), np.mean(inf)
    hs = base * np.array([1, 2, 4, 8, 16, 32])
    h1 = hs[int(np.argmin([ev(F.geocp_v1(h))[0] for h in hs]))]
    ks = [kk for kk in [20, 50, 100] if kk < len(D2) // 2] or [max(5, len(D2) // 4)]
    kL = ks[int(np.argmin([ev(F.lscp(kk))[0] for kk in ks]))]
    res = [ev(F.excp_rlcp(h)) for h in hs]
    ok = [i for i in range(len(hs)) if res[i][1] <= 0.005]
    hR = hs[min(ok, key=lambda i: res[i][0])] if ok else hs[-1]
    return dict(h1=h1, k=kL, hR=hR)


def qrf_bounds(Xtr, ytr, Xq, seed):
    """Meinshausen (2006) QRF: leaf co-membership weights over the training fold."""
    f = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.6, n_jobs=1,
                              random_state=seed).fit(Xtr, ytr)
    Ltr, Lq = f.apply(Xtr), f.apply(Xq)
    W = np.zeros((len(Xq), len(Xtr)), np.float32)
    for t in range(Ltr.shape[1]):
        lt, lq = Ltr[:, t], Lq[:, t]
        _, inv, cnt = np.unique(lt, return_inverse=True, return_counts=True)
        W += (lq[:, None] == lt[None, :]) / cnt[inv][None, :].astype(np.float32)
    W /= Ltr.shape[1]
    o = np.argsort(ytr); cs = np.cumsum(W[:, o], axis=1); ys = ytr[o]
    lo = ys[np.minimum((cs < ALPHA / 2).sum(1), len(ys) - 1)]
    hi = ys[np.minimum((cs < 1 - ALPHA / 2).sum(1), len(ys) - 1)]
    return lo, hi


def kriging_bounds(X, y, s, Dtr, Q, mu, rng, nsub=800):
    """Regression kriging: OOF RF residuals on the training fold, GP on coordinates."""
    perm = rng.permutation(Dtr); a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    oof = np.empty(len(perm))
    oof[: len(a)] = y[a] - rf(X[b], y[b], 6, 60).predict(X[a])
    oof[len(a):] = y[b] - rf(X[a], y[a], 7, 60).predict(X[b])
    idx = np.r_[a, b]; sub = rng.choice(len(idx), min(nsub, len(idx)), replace=False)
    S, r = s[idx][sub], oof[sub]
    ls0 = float(np.median(cKDTree(S).query(S, k=min(11, len(S)))[0][:, -1])) * 5 + 1e-6
    kern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(ls0, (ls0 / 100, ls0 * 100), nu=1.5) + WhiteKernel(1.0, (1e-5, 1e2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(S, r)
        m_, sd = gp.predict(s[Q], return_std=True)
    c = mu[Q] + m_
    return c - Z * sd, c + Z * sd


def bootstrap_bounds(X, y, Dtr, Q, seed):
    f = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.6, n_jobs=1, oob_score=True,
                              random_state=seed).fit(X[Dtr], y[Dtr])
    P = np.stack([t.predict(X[Q]) for t in f.estimators_], 1)
    c, vt = P.mean(1), P.var(1)
    vr = np.mean((y[Dtr] - f.oob_prediction_) ** 2)
    sd_full, sd_ens = np.sqrt(vt + vr), np.sqrt(vt)
    return (c - Z * sd_full, c + Z * sd_full), (c - Z * sd_ens, c + Z * sd_ens)


def one(args):
    name, rep = args
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    cl = cells(name, d)
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # == run_main.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    m = len(T); t0 = time.time()
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    h2, base = bw_of(name, rep)
    rng2 = np.random.default_rng([11, rep, DS.ALL.index(name)])
    bw = tune_local(X, y, s, Dtr, rng2)
    Ra = np.abs(y - mu)
    # selections (identical construction to run_main.py)
    cost = 1 + np.linalg.norm(s - s[Dtr].mean(0), axis=1) / np.median(np.linalg.norm(s[Dtr] - s[Dtr].mean(0), axis=1))
    val = mu - mu[Dtr].min() + 0.05 * mu[Dtr].std()
    B = 0.05 * m * cost[Dtr].mean()
    sels = {}
    for f in FRACS:
        K = max(1, int(round(f * m)))
        sels[f"top{int(f*100)}"] = refsets.topk(mu[C], mu[T], K)
        sels[f"bottom{int(f*100)}"] = refsets.topk(mu[C], mu[T], K, largest=False)
    Kg = quotas(st, m)
    sels["quota"] = refsets.quota(mu[C], mu[T], st[C], st[T], Kg)
    sels["knapsack"] = refsets.knapsack_greedy(val[C], cost[C], val[T], cost[T], B)
    if DS.POLICY[name] == "bottom" and name != "era5":
        sels["quota_bottom"] = refsets.quota(mu[C], mu[T], st[C], st[T], Kg, largest=False)
        need = mu[Dtr].max() - mu + 0.05 * mu[Dtr].std()
        sels["knapsack_need"] = refsets.knapsack_greedy(need[C], cost[C], need[T], cost[T], B)
    sels["mask_h2"] = refsets.kernel_mask(s[C], s[T], h2, ALPHA)
    sels["mask_base"] = refsets.kernel_mask(s[C], s[T], base, ALPHA)
    # interval bounds on ALL test units (marginal methods), then read off the selected ones
    seed = int(rng2.integers(2 ** 31))
    hq_lo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=20, random_state=seed).fit(X[Dtr], y[Dtr])
    hq_hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=20, random_state=seed).fit(X[Dtr], y[Dtr])
    qlo, qhi = hq_lo.predict(X), hq_hi.predict(X)
    qc = q_conf(np.maximum(qlo[C] - y[C], y[C] - qhi[C]))
    bounds = {"CQR": (qlo[T] - qc, qhi[T] + qc)}
    bounds["QRF"] = qrf_bounds(X[Dtr], y[Dtr], X[T], seed)
    bounds["Regression kriging"] = kriging_bounds(X, y, s, Dtr, T, mu, rng2)
    bounds["Bootstrap ensemble + residual"], bounds["Bootstrap ensemble only"] = bootstrap_bounds(X, y, Dtr, T, seed)
    ctx = dict(S_cal=Ra[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T], rng=rng2)
    for nm, meth in [("GeoCP", F.geocp_v1(bw["h1"])), ("LSCP", F.lscp(min(bw["k"], len(C) - 1))),
                     ("RLCP", F.excp_rlcp(bw["hR"]))]:
        q = meth.thresholds(ctx, ALPHA)
        bounds[nm] = (mu[T] - q, mu[T] + q)
    rows, groups = [], defaultdict(lambda: np.zeros(3))
    for rn, sel in sels.items():
        if len(sel) == 0:
            continue
        k = sel.selected; J = T[k]
        for mn, (L, U) in bounds.items():
            L, U = L[k], U[k]
            cov = (y[J] >= L) & (y[J] <= U); fin = np.isfinite(U - L); hw = (U - L) / 2
            rows.append(dict(dataset=name, rep=rep, rule=rn, method=mn, n_sel=len(J), n_cov=int(cov.sum()),
                             n_inf=int((~fin).sum()), hw_med=float(np.median(hw)),
                             hw_mean_fin=float(hw[fin].mean()) if fin.any() else np.nan,
                             sumY=float(y[J].sum()), sumU=float(U.sum()), sumL=float(L.sum()),
                             up_breach=int((y[J] > U).sum()), lo_breach=int((y[J] < L).sum()),
                             h1=bw["h1"], k=bw["k"], hR=bw["hR"]))
            for gt, lab in (("state", st[J]), ("cell", cl[J])):
                for u in np.unique(lab):
                    kk = lab == u
                    groups[(rn, mn, gt, int(u))] += (kk.sum(), cov[kk].sum(), (~fin[kk]).sum())
    return rows, dict(groups), time.time() - t0


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[3:] or ["camels", "seattle", "voting", "health_stroke", "health_diabetes", "health_bphigh",
                             "health_obesity", "fia", "era5", "poverty_lab", "poverty"]
    for nm in names:
        t0 = time.time()
        with Pool(W) as p:
            res = p.map(one, [(nm, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a, _, _ in res for x in a])
        df.to_csv(os.path.join(OUT, f"bl_main_{nm}.csv"), index=False)
        tot = defaultdict(lambda: np.zeros(3))
        for _, g, _ in res:
            for k, v in g.items():
                tot[k] += v
        pd.DataFrame([dict(rule=k[0], method=k[1], gtype=k[2], group=k[3], n=int(v[0]), c=int(v[1]), inf=int(v[2]))
                      for k, v in tot.items()]).to_csv(os.path.join(OUT, f"bl_groups_{nm}.csv"), index=False)
        g = df.groupby(["rule", "method"]); cov = (g.n_cov.sum() / g.n_sel.sum()).unstack(1).round(3)
        print(f"== {nm}: R={R}, {time.time()-t0:.0f}s (mean {np.mean([t for *_, t in res]):.1f}s/rep)\n"
              f"{cov.loc[[r for r in ['top5', 'bottom5', 'quota'] if r in cov.index]]}", flush=True)


if __name__ == "__main__":
    main()
