"""Experiment D of paper/MONDRIAN_ADVANTAGE_DESIGN.md (written before the run): funding with FDR control inside the
selected villages, reference-set cfBH against Mondrian conformal p-values.

Setting, score, eligibility, rules and seeds are those of experiments/run_cfbh.py (bottom 5% and country quota, DHS
random-forest targets poverty / poverty_lab and the deep-backbone targets poverty_resnet / poverty_dofa, 100 splits,
q = 0.05, 0.10). For a selected village j the conformal p-value is
    p_j = (1 + #{i in C_j : Y_i >= c_i, g_i < g_j}) / (|C_j| + 1),   g = (mu - c) / sigma,
followed by BH at level q over the selected villages, where the calibration set C_j is
  reference set   R_j of the selection rule (cfBH as in run_cfbh.py; reproduced exactly before anything is written)
  full            all calibration units (the p-values of Jin & Candes 2023, ignoring the first-stage selection; BH still
                  runs over the selected villages)
  Mondrian 10/20  the calibration units in j's bin of mu (ten or twenty bins cut at quantiles of mu on the fold that
                  fitted sigma: the training fold, or the structure fold Ds for the deep backbones)
  Mondrian country (quota only) the calibration units of j's country
  Mondrian within-country 10/20 (quota only; added post hoc after the adversarial check of 2026-10-30, see
                  paper/MONDRIAN_ADVANTAGE_DESIGN.md) bins matched to the quota: each country's mu cut at the quantiles
                  of that country's units in the same fold (of the whole fold if it holds fewer than G of the country's
                  units: with 20 bins, 1 ResNet and 3 DOFA splits of 100), pooled across countries by bin index
and the Mondrian three-way rule funds j iff the upper end of its Mondrian CP interval (normalized score, ten bins) is
below c_j (as 'triage JOMI (U < c)' in run_cfbh.py).
    ~/miniforge3/envs/favela/bin/python -u experiments/run_cfbh_mondrian.py <R> <workers> [targets...]
writes results/cfbh_md_<target>.csv
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
from run_cfbh import bh, cf_pvalues, QS, RF_T, FM_T                     # noqa: E402
from run_mondrian import mondrian                                       # noqa: E402
from excp_select import refsets                                         # noqa: E402

OUT = os.path.join(ROOT, "results")
_REF = {}


def ref(target):
    if target not in _REF:
        _REF[target] = pd.read_csv(os.path.join(OUT, f"cfbh_{target}.csv")).set_index(["rep", "rule", "method"])
    return _REF[target]


def bins(mu, train, G):
    edges = np.quantile(mu[train], np.linspace(0, 1, G + 1)[1:-1])
    return np.searchsorted(edges, mu, side="right")


def country_bins(mu, train, st, G):
    """Bin index within each country (quantiles of that country's fold units; a country with fewer than G fold units uses
    the whole fold's quantiles), pooled across countries by index."""
    b = np.zeros(len(mu), int)
    for u in np.unique(st):
        tr = train[st[train] == u]; ref = mu[tr] if len(tr) >= G else mu[train]
        edges = np.quantile(ref, np.linspace(0, 1, G + 1)[1:-1]); k = st == u
        b[k] = np.searchsorted(edges, mu[k], side="right")
    return b


def run(target, rep, y, mu, sig, C, T, st, c_of, train):
    g = (mu - c_of) / sig
    m = len(T); K = max(1, int(round(0.05 * m)))
    sels = {"bottom 5%": refsets.topk(mu[C], mu[T], K, largest=False),
            "quota": refsets.quota(mu[C], mu[T], st[C], st[T], quotas(st, m), largest=False)}
    b10, b20 = bins(mu, train, 10), bins(mu, train, 20)
    S = np.abs(y - mu) / sig
    q10, _ = mondrian(S, mu, train, C, T, 10)                 # Mondrian CP thresholds for every test unit
    R = ref(target); rows = []
    for rn, sel in sels.items():
        k = sel.selected; J = T[k]; elig = y[J] < c_of[J]
        cal = {"cfBH": sel.member,
               "cfBH, full calibration": np.ones((len(J), len(C)), bool),
               "Mondrian cfBH, 10 bins": b10[C][None, :] == b10[J][:, None],
               "Mondrian cfBH, 20 bins": b20[C][None, :] == b20[J][:, None]}
        if rn == "quota":
            cal["Mondrian cfBH, country"] = st[C][None, :] == st[J][:, None]
            for G_ in (10, 20):                                                   # post hoc: bins matched to the quota
                bq = country_bins(mu, train, st, G_)
                cal[f"Mondrian cfBH, within-country {G_} bins"] = bq[C][None, :] == bq[J][:, None]
            cal["cfBH (group-wise)"] = sel.member & (st[C][None, :] == st[J][:, None])
        dec, ncal = {}, {}
        for nm_, rows_ in cal.items():
            p = cf_pvalues(g, y, c_of, rows_, J, C); ncal[nm_] = float(rows_.sum(1).mean())
            for q in QS:
                dec[f"{nm_} q={q:.2f}"] = (bh(p, q), ncal[nm_])
        dec["Mondrian triage (U < c)"] = (mu[J] + q10[k] * sig[J] < c_of[J], np.nan)
        for mn, (fund, nc) in dec.items():
            nf = int(fund.sum()); fi = int((fund & ~elig).sum())
            if mn.startswith("cfBH q=") or mn.startswith("cfBH (group-wise)"):       # reproduce run_cfbh.py
                q = mn[-4:]
                old = (f"cfBH q={q}" if rn == "bottom 5%" else f"cfBH q={q} (heuristic)") if mn.startswith("cfBH q=") \
                    else f"cfBH q={q} (group-wise)"
                assert (nf, fi) == tuple(R.loc[(rep, rn, old), ["n_fund", "n_fund_inelig"]]), (target, rep, rn, mn)
            rows.append(dict(target=target, rep=rep, rule=rn, method=mn, n_sel=len(J), n_elig=int(elig.sum()),
                             n_fund=nf, n_fund_inelig=fi, fdp=fi / max(nf, 1), n_fund_elig=int((fund & elig).sum()),
                             cal_size=nc, ref_size=float(sel.member.sum(1).mean())))
    return rows


def one_rf(args):
    name, rep = args
    d = DS.load(name); N = d["N"]; X, y, st = d["Xrf"], d["y"], d["state"]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # == run_main.py / run_cfbh.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    cc = {u: np.quantile(y[Dtr][st[Dtr] == u], 0.25) if (st[Dtr] == u).sum() >= 5 else np.quantile(y[Dtr], 0.25)
          for u in np.unique(st)}
    c_of = np.array([cc[u] for u in st])
    return run(name, rep, y, mu, sig, C, T, st, c_of, Dtr)


def one_fm(args):
    combo, rep = args
    d = G.load(combo); y, s, g = d["y"], d["s"], d["group"]; N = len(y)
    rng = np.random.default_rng([606, rep, list(G.COMBOS).index(combo)])   # == run_geofm.py / run_cfbh.py
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
    return run(combo, rep, y, mu, sig, C, T, g, c_of, Ds)


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    targets = sys.argv[3:] or RF_T + FM_T
    for tg in targets:
        t0 = time.time()
        with Pool(W) as p:
            res = p.map(one_rf if tg in RF_T else one_fm, [(tg, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a in res for x in a]); df.to_csv(os.path.join(OUT, f"cfbh_md_{tg}.csv"), index=False)
        s = df.groupby(["rule", "method"]).agg(FDR=("fdp", "mean"), fund=("n_fund", "mean"), cal=("cal_size", "mean"),
                                               ref=("ref_size", "mean"), reached=("n_fund_elig", "sum"), elig=("n_elig", "sum"))
        s["eligible reached"] = s.reached / s.elig
        print(f"== {tg}: R={R}, {time.time()-t0:.0f}s (cfBH reproduced)\n{s[['FDR', 'fund', 'cal', 'ref', 'eligible reached']].round(3)}",
              flush=True)


if __name__ == "__main__":
    main()
