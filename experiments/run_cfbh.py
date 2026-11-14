"""Aid triage with FDR control: conformal selection (Jin & Candes 2023, cfBH) INSIDE the selected set.

First stage (as in run_decisions.py): the budget rule picks the K predicted-poorest villages
(bottom 5%) or the country-quota set. Second stage: certify "fund now" villages with
H_j: Y_j >= c_j (not eligible), controlling FDR = E[#ineligible funded / #funded] <= q.
  score  V(x, y) = M 1{y < c(x)} + g(x),  g = (mu - c) / sigma  (decreasing in y; clipped form)
  p_j    = (1 + #{i in R : Y_i >= c_i, g_i < g_j}) / (|R| + 1)
  then BH at level q over the selected villages.
Calibration uses the JOMI reference set R only. Under bottom-K, R u S is a uniform random
partition of the pool (structural lemma, paper/innovation_GeoCERT.md), so calibration and test
are exchangeable and Jin & Candes (2023) Thm 2.6 gives FDR <= q (theory/POOL_LEMMA.md T2). Under
quotas the weak R_j mixes groups (heuristic); the group-wise version uses same-group reference sets
R_g (Note T2.2, FDR <= q).
Compared with the interval triage of run_decisions.py (fund iff U <= c) for JOMI / SC-GeoCCP and
with point funding (mu <= c). Same seeds as run_decisions.py.
    ~/miniforge3/envs/favela/bin/python -u experiments/run_cfbh.py <R> <workers> [targets...]
writes results/cfbh_<target>.csv
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                       # noqa: E402
from run_main import fit_mu_sigma, quotas                               # noqa: E402
import run_geofm as G                                                   # noqa: E402
from excp_select import refsets, basis, methods as Mt                   # noqa: E402

ALPHA, L1, QS = 0.1, 0.95, (0.05, 0.10)
OUT = os.path.join(ROOT, "results")
RF_T = ["poverty_lab", "poverty"]
FM_T = ["poverty_resnet", "poverty_dofa"]


def bh(p, q):
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    o = np.argsort(p); ok = np.flatnonzero(p[o] <= q * np.arange(1, m + 1) / m)
    rej = np.zeros(m, bool)
    if len(ok):
        rej[o[: ok[-1] + 1]] = True
    return rej


def cf_pvalues(g, y, c, R_rows, J, Cidx):
    """p_j with calibration = reference set rows (bool over calibration units)."""
    p = np.empty(len(J))
    for r, j in enumerate(J):
        Ri = Cidx[R_rows[r]]
        null = y[Ri] >= c[Ri]
        p[r] = (1 + np.sum(null & (g[Ri] < g[j]))) / (len(Ri) + 1)
    return p


def run(target, rep, y, mu, sig, C, T, st, c_of, ctx):
    g = (mu - c_of) / sig
    m = len(T); K = max(1, int(round(0.05 * m)))
    sels = {"bottom 5%": refsets.topk(mu[C], mu[T], K, largest=False),
            "quota": refsets.quota(mu[C], mu[T], st[C], st[T], quotas(st, m), largest=False)}
    rows = []
    for rn, sel in sels.items():
        J = T[sel.selected]; elig = y[J] < c_of[J]
        dec = {"point (mu < c)": mu[J] < c_of[J]}
        qj = Mt.jomi(sel).thresholds(ctx, ALPHA); dec["triage JOMI (U < c)"] = mu[J] + qj * sig[J] < c_of[J]
        qs = Mt.sc_geoccp(sel, L1, ctx["pen"]).thresholds(ctx, ALPHA)
        dec["triage SC-GeoCCP (U < c)"] = mu[J] + qs * sig[J] < c_of[J]
        p = cf_pvalues(g, y, c_of, sel.member, J, C)
        for q in QS:
            dec[f"cfBH q={q:.2f}" + ("" if rn == "bottom 5%" else " (heuristic)")] = bh(p, q)
        if rn == "quota":                 # theory/POOL_LEMMA.md Note T2.2: same-group reference sets R_g
            same = sel.member & (st[C][None, :] == st[J][:, None])
            pg = cf_pvalues(g, y, c_of, same, J, C)
            for q in QS:
                dec[f"cfBH q={q:.2f} (group-wise)"] = bh(pg, q)
        for mn, fund in dec.items():
            nf = int(fund.sum()); fi = int((fund & ~elig).sum())
            rows.append(dict(target=target, rep=rep, rule=rn, method=mn, n_sel=len(J), n_elig=int(elig.sum()),
                             n_fund=nf, n_fund_inelig=fi, fdp=fi / max(nf, 1), n_fund_elig=int((fund & elig).sum())))
    return rows


def one_rf(args):
    name, rep = args
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # == run_main.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    cc = {u: np.quantile(y[Dtr][st[Dtr] == u], 0.25) if (st[Dtr] == u).sum() >= 5 else np.quantile(y[Dtr], 0.25)
          for u in np.unique(st)}
    c_of = np.array([cc[u] for u in st])
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st)
    S = np.abs(y - mu) / sig
    ctx = dict(S_cal=S[C], w_cal=np.ones(len(C)), w_test=np.ones(len(T)), g_cal=st[C], g_test=st[T],
               Phi_cal=Phi[C], Phi_test=Phi[T], pen=B.penalize)
    return run(name, rep, y, mu, sig, C, T, st, c_of, ctx)


def one_fm(args):
    combo, rep = args
    d = G.load(combo); y, s, g = d["y"], d["s"], d["group"]; N = len(y)
    rng = np.random.default_rng([606, rep, list(G.COMBOS).index(combo)])   # == run_geofm.py
    perm = rng.permutation(N); a, b = int(0.25 * N), int(0.625 * N)
    Ds, C, T = perm[:a], perm[a:b], perm[b:]
    mu = d["mu"] + 1e-9 * rng.standard_normal(N)
    Ra = np.abs(y - mu); feat = np.c_[d["emb"], s / 1000.0]
    sm = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.6, n_jobs=1,
                               random_state=int(rng.integers(2 ** 31))).fit(feat[Ds], Ra[Ds])
    sig = np.maximum(sm.predict(feat), 0.1 * Ra[Ds].mean() + 1e-9)
    cc = {u: np.quantile(y[Ds][g[Ds] == u], 0.25) if (g[Ds] == u).sum() >= 5 else np.quantile(y[Ds], 0.25)
          for u in np.unique(g)}
    c_of = np.array([cc[u] for u in g])
    B = basis.Multiscale(s[Ds]).with_units(g); Phi = B(s, g)
    ctx = dict(S_cal=(Ra / sig)[C], w_cal=np.ones(len(C)), w_test=np.ones(len(T)), g_cal=g[C], g_test=g[T],
               Phi_cal=Phi[C], Phi_test=Phi[T], pen=B.penalize)
    return run(combo, rep, y, mu, sig, C, T, g, c_of, ctx)


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    targets = sys.argv[3:] or RF_T + FM_T
    for tg in targets:
        t0 = time.time()
        with Pool(W) as p:
            # the formal DOFA setting (dofa_poverty, QW2_MAP=probe) runs like the random-forest ones
            res = p.map(one_rf if tg in RF_T or tg.startswith("dofa_") else one_fm, [(tg, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a in res for x in a]); df.to_csv(os.path.join(OUT, f"cfbh_{tg}.csv"), index=False)
        s = df.groupby(["rule", "method"]).agg(FDR=("fdp", "mean"), fund=("n_fund", "mean"),
                                               reached=("n_fund_elig", "sum"), elig=("n_elig", "sum"))
        s["eligible reached"] = s.reached / s.elig
        print(f"== {tg}: R={R}, {time.time()-t0:.0f}s\n{s[['FDR', 'fund', 'eligible reached']].round(3)}", flush=True)


if __name__ == "__main__":
    main()
