"""WP2: decision consequences of map uncertainty at the selected units.

Two selection rules per domain, both covariate-only (so JOMI reference sets apply; the key is
fixed given the training / structure fold):
  point     rank by mu                    (carbon/floods: top-K, aid: bottom-K)
  cautious  rank by a bound built from the training-fold scale model sigma (kappa = 1.645):
            carbon mu - kappa sigma (high AND sure), aid mu + kappa sigma (surely poor),
            floods mu + kappa sigma (possibly severe)
K = 5% of the test units; aid also uses country quotas on the same keys.
Decisions on the selected units, for every interval method [L, U] (point = no interval):
  carbon  credit = max(L, 0) (point: mu); over-credit = credit > Y; amounts in biomass units
  aid     eligible = Y <= c_country (training-fold country 25th percentile, as Pettersson 2026);
          fund if U <= c, drop if L > c, otherwise survey (point: fund iff mu <= c)
  floods  supplies = min(U, 1) (point: mu); shortage = (Y - supplies)_+ , over-supply = (supplies - Y)_+
Data: RF datasets (fia = carbon; poverty_lab, poverty = aid) on run_main.py splits, and GROUND
GeoFM combos (m3leo = carbon, poverty = aid, floods) on run_geofm.py splits.
Methods: point, Normalized CP, GeoCP (published), QRF, bootstrap ensemble (RF data only), JOMI,
SC-GeoCCP (l1 = 0.95).
    ~/miniforge3/envs/favela/bin/python -u experiments/run_decisions.py <R> <workers> [targets...]
writes results/dec_<target>.csv
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
from run_baselines import tune_local, qrf_bounds, bootstrap_bounds      # noqa: E402
import run_geofm as G                                                   # noqa: E402
from excp_select import refsets, basis, methods as Mt                   # noqa: E402
from excp import family as F                                            # noqa: E402

ALPHA, L1, KAPPA = 0.1, 0.95, 1.6448536269514722
OUT = os.path.join(ROOT, "results")
RF_TARGETS = {DS.FIA: "carbon", "poverty_lab": "aid", "poverty": "aid"}
FM_TARGETS = {"m3leo_resnet": "carbon", "m3leo_dofa": "carbon", "poverty_resnet": "aid", "poverty_dofa": "aid",
              "floods_resnet": "floods", "floods_dofa": "floods"}


def keys(domain, mu, sig):
    if domain == "carbon":
        return {"point": (mu, True), "cautious": (mu - KAPPA * sig, True)}
    if domain == "aid":
        return {"point": (mu, False), "cautious": (mu + KAPPA * sig, False)}
    return {"point": (mu, True), "cautious": (mu + KAPPA * sig, True)}


def consequences(domain, y, mu, L, U, c=None):
    """Per selected unit decision outcomes, summed. L/U may be None for the point rule."""
    out = {}
    if domain == "carbon":
        credit = mu if L is None else np.maximum(L, 0)
        credit = np.where(np.isfinite(credit), credit, 0.0)        # +inf interval -> no credit
        over = credit > y
        out.update(n_over=int(over.sum()), sum_credit=float(credit.sum()), sum_over=float(np.maximum(credit - y, 0).sum()),
                   sum_credited_true=float(np.minimum(credit, y).sum()), sumY=float(y.sum()),
                   total_overclaim=int(credit.sum() > y.sum()))
    elif domain == "aid":
        elig = y <= c
        if L is None:
            fund, drop = mu <= c, mu > c
        else:
            fund, drop = U <= c, L > c
        survey = ~fund & ~drop
        out.update(n_elig=int(elig.sum()), n_fund=int(fund.sum()), n_fund_inelig=int((fund & ~elig).sum()),
                   n_drop=int(drop.sum()), n_drop_elig=int((drop & elig).sum()), n_survey=int(survey.sum()),
                   n_fund_elig=int((fund & elig).sum()))
    else:
        sup = np.clip(mu if U is None else U, 0, 1)
        sup = np.where(np.isfinite(sup), sup, 1.0)
        out.update(n_short=int((y > sup).sum()), sum_short=float(np.maximum(y - sup, 0).sum()),
                   sum_oversupply=float(np.maximum(sup - y, 0).sum()), sum_supply=float(sup.sum()), sumY=float(y.sum()))
    return out


def evaluate(target, domain, rep, y, mu, sig, Ra, Ss, C, T, st, sels, ctx, ctx_a, geo_h, extra_bounds, c_of):
    rows = []
    for rn, sel in sels.items():
        if len(sel) == 0:
            continue
        k = sel.selected; J = T[k]
        c = c_of[J] if c_of is not None else None
        meths = {"point": (None, None)}
        q = F.split_cp().thresholds(dict(ctx, w_test=np.ones(len(k))), ALPHA)[0]
        meths["Normalized CP"] = (mu[J] - q * sig[J], mu[J] + q * sig[J])
        qg = Mt.on_selected(F.geocp_v1(geo_h), sel).thresholds(ctx_a, ALPHA)
        meths["GeoCP"] = (mu[J] - qg, mu[J] + qg)
        for nm, (lo, hi) in extra_bounds.items():
            meths[nm] = (lo[k], hi[k])
        qj = Mt.jomi(sel).thresholds(ctx, ALPHA)
        meths["JOMI"] = (mu[J] - qj * sig[J], mu[J] + qj * sig[J])
        qs = Mt.sc_geoccp(sel, L1, ctx["pen"]).thresholds(ctx, ALPHA)
        meths["SC-GeoCCP"] = (mu[J] - qs * sig[J], mu[J] + qs * sig[J])
        for mn, (L, U) in meths.items():
            r = dict(target=target, domain=domain, rep=rep, rule=rn, method=mn, n_sel=len(J))
            if L is not None:
                r["n_cov"] = int(((y[J] >= L) & (y[J] <= U)).sum())
            r.update(consequences(domain, y[J], mu[J], L, U, c))
            rows.append(r)
    return rows


def build_sels(domain, mu, sig, C, T, st, m):
    sels = {}
    K = max(1, int(round(0.05 * m)))
    for nm, (key, largest) in keys(domain, mu, sig).items():
        sels[f"{nm} 5%"] = refsets.topk(key[C], key[T], K, largest=largest)
        if domain == "aid":
            sels[f"{nm} quota"] = refsets.quota(key[C], key[T], st[C], st[T], quotas(st, m), largest=largest)
    return sels


def one_rf(args):
    name, rep = args
    domain = RF_TARGETS[name]
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # == run_main.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    m = len(T)
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    rng2 = np.random.default_rng([11, rep, DS.ALL.index(name)])
    bw = tune_local(X, y, s, Dtr, rng2)                                   # == run_baselines.py
    Ra, Ss = np.abs(y - mu), np.abs(y - mu) / sig
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st)
    ctx = dict(S_cal=Ss[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T], g_cal=st[C],
               g_test=st[T], Phi_cal=Phi[C], Phi_test=Phi[T], pen=B.penalize, rng=rng2)
    ctx_a = dict(ctx, S_cal=Ra[C])
    seed = int(rng2.integers(2 ** 31))
    qlo, qhi = qrf_bounds(X[Dtr], y[Dtr], X[T], seed)
    (blo, bhi), _ = bootstrap_bounds(X, y, Dtr, T, seed)
    c_of = None
    if domain == "aid":
        cc = {u: np.quantile(y[Dtr][st[Dtr] == u], 0.25) if (st[Dtr] == u).sum() >= 5 else np.quantile(y[Dtr], 0.25)
              for u in np.unique(st)}
        c_of = np.array([cc[u] for u in st])
    sels = build_sels(domain, mu, sig, C, T, st, m)
    return evaluate(name, domain, rep, y, mu, sig, Ra, Ss, C, T, st, sels, ctx, ctx_a, bw["h1"],
                    {"QRF": (qlo, qhi), "Bootstrap": (blo, bhi)}, c_of)


def one_fm(args):
    combo, rep = args
    domain = FM_TARGETS[combo]
    d = G.load(combo); y, s, g = d["y"], d["s"], d["group"]; N = len(y)
    rng = np.random.default_rng([606, rep, list(G.COMBOS).index(combo)])   # == run_geofm.py
    perm = rng.permutation(N); a, b = int(0.25 * N), int(0.625 * N)
    Ds, C, T = perm[:a], perm[a:b], perm[b:]
    m = len(T)
    mu = d["mu"] + 1e-9 * rng.standard_normal(N)
    Ra = np.abs(y - mu)
    feat = np.c_[d["emb"], s / 1000.0]
    sm = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.6, n_jobs=1,
                               random_state=int(rng.integers(2 ** 31))).fit(feat[Ds], Ra[Ds])
    sig = np.maximum(sm.predict(feat), 0.1 * Ra[Ds].mean() + 1e-9)
    Ss = Ra / sig
    h1 = G.tune_geocp(Ra[Ds], s[Ds], rng)
    B = basis.Multiscale(s[Ds]).with_units(g); Phi = B(s, g)
    ctx = dict(S_cal=Ss[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T], g_cal=g[C],
               g_test=g[T], Phi_cal=Phi[C], Phi_test=Phi[T], pen=B.penalize, rng=rng)
    ctx_a = dict(ctx, S_cal=Ra[C])
    c_of = None
    if domain == "aid":
        cc = {u: np.quantile(y[Ds][g[Ds] == u], 0.25) if (g[Ds] == u).sum() >= 5 else np.quantile(y[Ds], 0.25)
              for u in np.unique(g)}
        c_of = np.array([cc[u] for u in g])
    sels = build_sels(domain, mu, sig, C, T, g, m)
    return evaluate(combo, domain, rep, y, mu, sig, Ra, Ss, C, T, g, sels, ctx, ctx_a, h1, {}, c_of)


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    targets = sys.argv[3:] or list(RF_TARGETS) + list(FM_TARGETS)
    for tg in targets:
        t0 = time.time()
        fn = one_rf if tg in RF_TARGETS else one_fm
        with Pool(W) as p:
            res = p.map(fn, [(tg, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a in res for x in a])
        df.to_csv(os.path.join(OUT, f"dec_{tg}.csv"), index=False)
        print(f"== {tg}: R={R}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
