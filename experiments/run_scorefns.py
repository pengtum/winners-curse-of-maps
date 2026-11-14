"""QRF / bootstrap as SCORE FUNCTIONS for selection-conditional calibration.

Same seeds, splits and selected sets as run_main.py (the selection still ranks the common RF
mu). Two score functions, both fitted on the training fold only:
  CQR-QRF    S = max(qlo - y, y - qhi), qlo/qhi = QRF (Meinshausen 2006) alpha/2 and 1-alpha/2
             quantiles; interval [qlo - q, qhi + q]
  norm-boot  S = |y - c| / sd, c and sd from the bootstrap ensemble (tree spread + OOF residual
             variance); interval c +/- q sd
Each score is calibrated by split CP (marginal), JOMI and SC-GeoCCP (l1 = 0.95, multiscale basis);
the raw QRF and bootstrap intervals are recomputed here for a like-for-like comparison.
Outputs results/sf_main_<ds>.csv and results/sf_groups_<ds>.csv.

    ~/miniforge3/envs/favela/bin/python -u experiments/run_scorefns.py <R> <workers> [datasets...]
"""
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma, quotas                             # noqa: E402
from run_scgeoccp import cells, ORDER                                 # noqa: E402
from run_baselines import qrf_bounds, Z                               # noqa: E402
from excp_select import refsets, basis, methods as Mt                 # noqa: E402
from excp import family as F                                          # noqa: E402

ALPHA, L1 = 0.1, 0.95
OUT = os.path.join(ROOT, "results")


def boot_center_scale(X, y, Dtr, Q, seed):
    f = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.6, n_jobs=1, oob_score=True,
                              random_state=seed).fit(X[Dtr], y[Dtr])
    P = np.stack([t.predict(X[Q]) for t in f.estimators_], 1)
    vr = np.mean((y[Dtr] - f.oob_prediction_) ** 2)
    return P.mean(1), np.sqrt(P.var(1) + vr)


def one(args):
    name, rep = args
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    cl = cells(name, d)
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # == run_main.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    m = len(T); t0 = time.time()
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    seed = int(np.random.default_rng([12, rep, DS.ALL.index(name)]).integers(2 ** 31))
    Q = np.r_[C, T]                                                       # calibration + test units
    lo, hi = np.full(N, np.nan), np.full(N, np.nan)
    lo[Q], hi[Q] = qrf_bounds(X[Dtr], y[Dtr], X[Q], seed)
    cb, sb = np.full(N, np.nan), np.full(N, np.nan)
    cb[Q], sb[Q] = boot_center_scale(X, y, Dtr, Q, seed)
    S_cqr = np.maximum(lo - y, y - hi)
    S_nb = np.abs(y - cb) / np.maximum(sb, 1e-12)
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st); pen = B.penalize
    # selections (identical construction to run_main.py; masks omitted)
    cost = 1 + np.linalg.norm(s - s[Dtr].mean(0), axis=1) / np.median(np.linalg.norm(s[Dtr] - s[Dtr].mean(0), axis=1))
    val = mu - mu[Dtr].min() + 0.05 * mu[Dtr].std()
    Bud = 0.05 * m * cost[Dtr].mean()
    sels = {}
    for f in (0.01, 0.05):
        K = max(1, int(round(f * m)))
        sels[f"top{int(f*100)}"] = refsets.topk(mu[C], mu[T], K)
        sels[f"bottom{int(f*100)}"] = refsets.topk(mu[C], mu[T], K, largest=False)
    Kg = quotas(st, m)
    sels["quota"] = refsets.quota(mu[C], mu[T], st[C], st[T], Kg)
    sels["knapsack"] = refsets.knapsack_greedy(val[C], cost[C], val[T], cost[T], Bud)
    if DS.POLICY[name] == "bottom" and name != "era5":
        sels["quota_bottom"] = refsets.quota(mu[C], mu[T], st[C], st[T], Kg, largest=False)
        need = mu[Dtr].max() - mu + 0.05 * mu[Dtr].std()
        sels["knapsack_need"] = refsets.knapsack_greedy(need[C], cost[C], need[T], cost[T], Bud)
    base_ctx = dict(w_cal=np.ones(len(C)), w_test=np.ones(m), g_cal=st[C], g_test=st[T],
                    Phi_cal=Phi[C], Phi_test=Phi[T])
    rows, groups = [], defaultdict(lambda: np.zeros(3))

    def record(rn, mn, J, cov, fin, hw):
        rows.append(dict(dataset=name, rep=rep, rule=rn, method=mn, n_sel=len(J), n_cov=int(cov.sum()),
                         n_inf=int((~fin).sum()), hw_med=float(np.median(hw)),
                         hw_mean_fin=float(hw[fin].mean()) if fin.any() else np.nan))
        for gt, lab in (("state", st[J]), ("cell", cl[J])):
            for u in np.unique(lab):
                k = lab == u
                groups[(rn, mn, gt, int(u))] += (k.sum(), cov[k].sum(), (~fin[k]).sum())

    for rn, sel in sels.items():
        if len(sel) == 0:
            continue
        J = T[sel.selected]
        # raw model intervals (no calibration)
        record(rn, "QRF (raw)", J, (y[J] >= lo[J]) & (y[J] <= hi[J]), np.ones(len(J), bool), (hi[J] - lo[J]) / 2)
        record(rn, "Bootstrap (raw)", J, np.abs(y[J] - cb[J]) <= Z * sb[J], np.ones(len(J), bool), Z * sb[J])
        for tag, S, width in [("CQR-QRF", S_cqr, lambda q: (hi[J] - lo[J]) / 2 + q),
                              ("norm-boot", S_nb, lambda q: q * sb[J])]:
            ctx = dict(base_ctx, S_cal=S[C])
            for cal, meth in [("split", Mt.on_selected(F.split_cp(), sel)), ("JOMI", Mt.jomi(sel)),
                              ("SC-GeoCCP", Mt.sc_geoccp(sel, L1, pen))]:
                q = meth.thresholds(ctx, ALPHA)
                record(rn, f"{cal} {tag}", J, S[J] <= q, np.isfinite(q), width(q))
    return rows, dict(groups), time.time() - t0


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[3:] or ORDER
    for nm in names:
        t0 = time.time()
        with Pool(W) as p:
            res = p.map(one, [(nm, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a, _, _ in res for x in a])
        df.to_csv(os.path.join(OUT, f"sf_main_{nm}.csv"), index=False)
        tot = defaultdict(lambda: np.zeros(3))
        for _, g, _ in res:
            for k, v in g.items():
                tot[k] += v
        pd.DataFrame([dict(rule=k[0], method=k[1], gtype=k[2], group=k[3], n=int(v[0]), c=int(v[1]), inf=int(v[2]))
                      for k, v in tot.items()]).to_csv(os.path.join(OUT, f"sf_groups_{nm}.csv"), index=False)
        g = df.groupby(["rule", "method"]); cov = (g.n_cov.sum() / g.n_sel.sum()).unstack(1).round(3)
        print(f"== {nm}: R={R}, {time.time()-t0:.0f}s (mean {np.mean([t for *_, t in res]):.1f}s/rep)\n"
              f"{cov.loc[[r for r in ['top5', 'bottom5', 'quota'] if r in cov.index]].T}", flush=True)


if __name__ == "__main__":
    main()
