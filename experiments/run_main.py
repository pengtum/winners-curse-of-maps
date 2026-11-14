"""QW2 main experiment: intervals at the places a budget rule picks.

Per dataset and repetition: random split train 40% / calibration 30% / test 30% (Pi uniform:
design-based validity under any spatial autocorrelation). Same random forest mu; sigma is a
cross-fitted training-fold scale model (excp convention). Bandwidths (GeoCP v2 h, RLCP+ kappa)
are tuned on the training fold only.

Rules (label-free, invariant to calibration permutations; reference sets in excp_select.refsets)
  top / bottom-K at K/m in {1%, 5%, 20%}
  quota      top 5% split into fixed state quotas K_s proportional to the state's unit count
  knapsack   value = mu (shifted > 0), cost = 1 + distance to the training centroid / median,
             fixed budget B = 5% of m x mean training cost; greedy ratio rule
  mask       publish only where GeoCP v2 is finite (kernel mass >= (1-a)/a), h = tuned h2 and
             h = base (median 10-NN distance in the training fold)
Methods evaluated on the selected units
  SCP |res|, SCP stud, GeoCP v2 (|res|, tuned h), ExCP-default (stud x RLCP+), BY stud,
  JOMI |res|, JOMI stud, JOMI-ExCP (stud x RLCP+ x reference set), [quota] JOMI|state

    ~/miniforge3/envs/favela/bin/python -u experiments/run_main.py <R> <workers> [datasets...]
writes results/main_<dataset>.csv (one row per rep x rule x method), results/state_<dataset>.csv
(quota per-state counts), results/units_<dataset>_rep0.npz (unit-level detail for maps).
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from excp_select import refsets, methods as Mt                        # noqa: E402
from excp import family as F                                          # noqa: E402

ALPHA = 0.1
FRACS = [0.01, 0.05, 0.20]
# a-priori selected fraction of each rule (fixed by design, used only to size JOMI-ExCP's kappa)
SEL_FRACS = {**{f"{d}{int(f*100)}": f for f in FRACS for d in ("top", "bottom")},
             "quota": 0.05, "knapsack": 0.05, "mask_h2": 1.0, "mask_base": 1.0,
             "quota_bottom": 0.05, "knapsack_need": 0.05}
OUT = os.path.join(ROOT, "results")


MAP = os.environ.get("QW2_MAP", "rf")          # "probe": the formal DOFA arm (paper/DOFA_FORMAL_DESIGN.md)


class Probe:
    """Linear probe on frozen foundation-model features (QW2_MAP=probe): ridge on the standardised embedding columns
    Xrf[:, 2:] (lon/lat are not used), penalty chosen from 10^-3..10^3 by efficient leave-one-out (GCV) on the data it is
    fitted to, which is always a training fold or part of one."""
    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        Z = X[:, 2:]; self.m, self.s = Z.mean(0), Z.std(0) + 1e-12
        self.r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((Z - self.m) / self.s, y)
        return self

    def predict(self, X):
        return self.r.predict((X[:, 2:] - self.m) / self.s)


def rf(X, y, seed, n=100):
    if MAP == "probe":
        return Probe().fit(X, y)
    return RandomForestRegressor(n_estimators=n, min_samples_leaf=3, max_features=0.6, n_jobs=1,
                                 random_state=seed).fit(X, y)


def fit_mu_sigma(X, y, Dtr, rng):
    """excp benchmark convention (design_based_spatial_cp/sim/bench/run_bench.py)."""
    mu = rf(X[Dtr], y[Dtr], int(rng.integers(2 ** 31))).predict(X)
    perm = rng.permutation(Dtr); a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    oof = np.empty(len(perm))
    oof[: len(a)] = np.abs(y[a] - rf(X[b], y[b], 1, 60).predict(X[a]))
    oof[len(a):] = np.abs(y[b] - rf(X[a], y[a], 2, 60).predict(X[b]))
    sig = RandomForestRegressor(n_estimators=60, min_samples_leaf=10, max_features=0.6, n_jobs=1,
                                random_state=3).fit(X[np.r_[a, b]], oof).predict(X)
    return mu + 1e-9 * rng.standard_normal(len(y)), np.maximum(sig, 0.1 * oof.mean())


def iscore(R, q):
    q = np.where(np.isfinite(q), q, R.max())
    return np.mean(2 * q + (2 / ALPHA) * np.maximum(R - q, 0))


def tune(X, y, s, Dtr, rng):
    """Training fold only: GeoCP v2 bandwidth by interval score; RLCP+ kappa by interval score
    among kappas with +inf share <= 0.5% (else None = global)."""
    perm = rng.permutation(Dtr); k = int(0.6 * len(perm)); D1, D2 = perm[:k], perm[k:]
    R2 = np.abs(y[D2] - rf(X[D1], y[D1], 4, 60).predict(X[D2]))
    base = float(np.median(cKDTree(s[Dtr]).query(s[Dtr], k=min(11, len(Dtr)))[0][:, -1]))
    halves = [np.random.default_rng(r).permutation(len(D2)) for r in range(2)]

    def ev(meth):
        sc, inf = [], []
        for p in halves:
            a, b = p[: len(p) // 2], p[len(p) // 2:]
            c = dict(S_cal=R2[a], w_cal=np.ones(len(a)), w_test=np.ones(len(b)), s_cal=s[D2][a],
                     s_test=s[D2][b], s_train=s[D1], w_train=np.ones(len(D1)), rng=np.random.default_rng(7))
            q = meth.thresholds(c, ALPHA); sc.append(iscore(R2[b], q)); inf.append(np.mean(~np.isfinite(q)))
        return np.mean(sc), np.mean(inf)
    hs = base * np.array([1, 2, 4, 8, 16, 32])
    h2 = hs[int(np.argmin([ev(F.geocp_v2(h))[0] for h in hs]))]
    kaps = [kk for kk in [20, 40, 80, 160] if kk < len(D1)]
    res = [ev(F.excp_default(kk)) for kk in kaps]
    ok = [i for i in range(len(kaps)) if res[i][1] <= 0.005]
    kappa = kaps[min(ok, key=lambda i: res[i][0])] if ok else None
    # JOMI-ExCP: the reference set holds ~frac of the calibration units, so RLCP+ needs a wider
    # neighbourhood. Per selection fraction, the most local kappa whose EXTRA +inf share over
    # plain JOMI (pseudo top- and bottom-K on the training fold) is <= 0.5%; else None.
    mu2 = rf(X[D1], y[D1], 4, 60).predict(X[D2]) + 1e-9 * rng.standard_normal(len(D2))
    scale = len(Dtr) / len(D1)
    ksel = {}
    for f in sorted(set(SEL_FRACS.values())):
        if f >= 0.5:
            ksel[f] = kappa; continue
        grid = [kk for kk in (kappa or 20) * 2 ** np.arange(0, 9) if kk < len(D1)]
        ksel[f] = None
        for kk in grid:
            extra, base_inf = [], []
            for p in halves:
                a, b = p[: len(p) // 2], p[len(p) // 2:]
                c = dict(S_cal=R2[a], w_cal=np.ones(len(a)), w_test=np.ones(len(b)), s_cal=s[D2][a],
                         s_test=s[D2][b], s_train=s[D1], w_train=np.ones(len(D1)), rng=np.random.default_rng(7))
                for largest in (True, False):
                    sel = refsets.topk(mu2[a], mu2[b], max(1, int(round(f * len(b)))), largest)
                    q0 = Mt.jomi(sel).thresholds(c, ALPHA); q1 = Mt.jomi_excp(sel, kk).thresholds(c, ALPHA)
                    extra.append(np.mean(~np.isfinite(q1)) - np.mean(~np.isfinite(q0)))
                    base_inf.append(np.mean(~np.isfinite(q0)))
            if np.mean(base_inf) > 0.005:       # reference set too small to judge: no localization
                break
            if np.mean(extra) <= 0.005:
                ksel[f] = int(round(kk * scale)); break
    return dict(h2=h2, base=base, kappa=kappa, ksel=ksel)


def quotas(state, m, frac=0.05):
    u, cnt = np.unique(state, return_counts=True)
    return {int(a): (max(1, int(round(frac * m * c / len(state)))) if c >= 20 else 0) for a, c in zip(u, cnt)}


def one(args):
    name, rep = args
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    m = len(T)
    t0 = time.time()
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    bw = tune(X, y, s, Dtr, rng)
    Ra, Ss = np.abs(y - mu), np.abs(y - mu) / sig
    ctx = dict(S_cal=Ss[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T],
               s_train=s[Dtr], w_train=np.ones(len(Dtr)), g_cal=st[C], g_test=st[T], rng=rng)
    ctx_a = dict(ctx, S_cal=Ra[C])
    # rules
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
    if DS.POLICY[name] == "bottom" and name != "era5":    # aid-type rules fund the predicted-poorest
        sels["quota_bottom"] = refsets.quota(mu[C], mu[T], st[C], st[T], Kg, largest=False)
        need = mu[Dtr].max() - mu + 0.05 * mu[Dtr].std()
        sels["knapsack_need"] = refsets.knapsack_greedy(need[C], cost[C], need[T], cost[T], B)
    sels["mask_h2"] = refsets.kernel_mask(s[C], s[T], bw["h2"], ALPHA)
    sels["mask_base"] = refsets.kernel_mask(s[C], s[T], bw["base"], ALPHA)
    rows, srows, units = [], [], {}
    base_row = dict(dataset=name, rep=rep, n_cal=len(C), m=m, h2=bw["h2"], base=bw["base"],
                    kappa=bw["kappa"] if bw["kappa"] else 0)
    for rn, sel in sels.items():
        if len(sel) == 0:
            continue
        J = T[sel.selected]
        M = {
            "SCP |res|": (Mt.on_selected(F.split_cp(), sel), ctx_a, Ra, 1.0),
            "SCP stud": (Mt.on_selected(F.split_cp(), sel), ctx, Ss, sig),
            "GeoCP v2": (Mt.on_selected(F.geocp_v2(bw["h2"]), sel), ctx_a, Ra, 1.0),
            "ExCP-default": (Mt.on_selected(F.excp_default(bw["kappa"]), sel), ctx, Ss, sig),
            "BY stud": (Mt.by(sel, m), ctx, Ss, sig),
            "JOMI |res|": (Mt.jomi(sel), ctx_a, Ra, 1.0),
            "JOMI stud": (Mt.jomi(sel), ctx, Ss, sig),
            "JOMI-ExCP": (Mt.jomi_excp(sel, bw["ksel"][SEL_FRACS[rn]]), ctx, Ss, sig),
        }
        if rn.startswith("quota"):
            M["JOMI|state stud"] = (Mt.jomi(sel, by_group=True), ctx, Ss, sig)
        Rs = sel.sizes
        for mn, (meth, c, S, scale) in M.items():
            q = meth.thresholds(c, ALPHA)
            cov = S[J] <= q
            hw = q * (scale[J] if np.ndim(scale) else scale)
            fin = np.isfinite(hw)
            U, L = mu[J] + hw, mu[J] - hw
            r = dict(base_row, rule=rn, method=mn, n_sel=len(J), n_cov=int(cov.sum()), n_inf=int((~fin).sum()),
                     hw_mean=float(hw[fin].mean()) if fin.any() else np.nan,
                     hw_med=float(np.median(hw)), sumY=float(y[J].sum()), sumMu=float(mu[J].sum()),
                     sumU=float(U.sum()), sumL=float(np.where(fin, L, -np.inf).sum()),
                     up_breach=int((y[J] > U).sum()), lo_breach=int((y[J] < L).sum()),
                     sd_y=float(y.std()))
            if mn == "JOMI-ExCP":
                r.update(kappa_sel=bw["ksel"][SEL_FRACS[rn]] or 0)
            if mn.startswith("JOMI"):
                r.update(R_mean=float(Rs.mean()), R_min=int(Rs.min()),
                         ub_sum=float(Mt.upper_bound(sel, ALPHA).sum()))
            rows.append(r)
            if rn.startswith("quota"):
                for u in np.unique(st[J]):
                    k = st[J] == u
                    srows.append(dict(dataset=name, rep=rep, rule=rn, method=mn, state=int(u), n_sel=int(k.sum()),
                                      n_cov=int(cov[k].sum()), n_inf=int((~fin[k]).sum())))
            if rep == 0:
                units[f"{rn}|{mn}|cov"] = cov; units[f"{rn}|{mn}|hw"] = hw
        if rep == 0:
            units[f"{rn}|idx"] = J
    if rep == 0:
        np.savez_compressed(os.path.join(OUT, f"units_{name}_rep0.npz"), coords=s, y=y, mu=mu, sig=sig,
                            state=st, C=C, T=T, **units)
    return rows, srows, time.time() - t0


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[3:] or [n for n in DS.ALL if n not in DS.DOFA]
    for nm in names:
        DS.load(nm)
    for nm in names:
        t0 = time.time()
        jobs = [(nm, r) for r in range(R)]
        with Pool(W) as p:
            res = p.map(one, jobs, chunksize=1)
        rows = [x for a, _, _ in res for x in a]; srows = [x for _, b, _ in res for x in b]
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"main_{nm}.csv"), index=False)
        pd.DataFrame(srows).to_csv(os.path.join(OUT, f"state_{nm}.csv"), index=False)
        df = pd.DataFrame(rows)
        g = df[df.rule.isin(["top5", "bottom5"])].groupby(["rule", "method"])
        cov = (g.n_cov.sum() / g.n_sel.sum()).unstack(0).round(3)
        print(f"== {nm}: R={R}, {time.time()-t0:.0f}s (mean {np.mean([t for *_, t in res]):.1f}s/rep)\n{cov}", flush=True)


if __name__ == "__main__":
    main()
