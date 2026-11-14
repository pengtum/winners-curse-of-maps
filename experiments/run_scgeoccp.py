"""SC-GeoCCP on all datasets: selection-conditional calibration on a multiscale spatial basis.

Same seeds as run_main.py, so mu, sigma and every selection are identical to the main run.
Per dataset and repetition (train 40 / calibration 30 / test 30):
  basis   excp_select.basis.Multiscale: intercept (unpenalized) + Gaussian bumps at 3 scales
          (k-means centres of the TRAINING coordinates) + state/country indicators (penalized)
  l1      chosen on the training fold only (inner 60/40 split, two pseudo cal/test halvings,
          pseudo state quota, macro-over-states interval score), grid L1_GRID (> tau)
  rules   top/bottom 1% and 5%, state quota, knapsack (+ quota_bottom, knapsack_need for the
          PovertyMap variants), as in run_main.py
  methods SCP stud, ExCP-default (kappa from the main run), JOMI stud, JOMI|state (quota rules),
          SC-GeoCCP
Outputs results/scg_main_<ds>.csv (rep x rule x method) and results/scg_groups_<ds>.csv
(pooled counts per state and per spatial cell).

    ~/miniforge3/envs/favela/bin/python -u experiments/run_scgeoccp.py <R> <workers> [datasets...]
Env L1_FIXED=<value>: skip tuning, use that l1 ("smallest l1 > tau" rule), compute only SC-GeoCCP,
write results/scg_main_<ds>_l1fix.csv / scg_groups_<ds>_l1fix.csv.
"""
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma, quotas, rf                         # noqa: E402
from excp_select import refsets, basis, methods as Mt                 # noqa: E402
from excp import family as F                                          # noqa: E402

ALPHA = 0.1
L1_GRID = [0.95, 1.0, 1.5, 2.5, 4.0]     # all >= tau = 0.9: +inf only where JOMI is +inf (Prop. 2)
L1_FIXED = float(os.environ["L1_FIXED"]) if os.environ.get("L1_FIXED") else None
SUFFIX = "_l1fix" if L1_FIXED else ""
OUT = os.path.join(ROOT, "results")
ORDER = ["camels", "seattle", "voting", "health_stroke", "health_diabetes", "health_bphigh",
         "health_obesity", "fia", "era5", "poverty_lab", "poverty"]
_CELLS, _KAPPA = {}, {}


def cells(name, d):
    if name not in _CELLS:
        from sklearn.cluster import KMeans
        k = int(min(20, max(4, d["N"] // 150)))
        _CELLS[name] = KMeans(k, n_init=4, random_state=0).fit_predict(d["coords"])
    return _CELLS[name]


def kappa_of(name, rep):
    if name not in _KAPPA:
        m = pd.read_csv(os.path.join(OUT, f"main_{name}.csv"), usecols=["rep", "kappa"]).drop_duplicates("rep")
        _KAPPA[name] = dict(zip(m.rep, m.kappa))
    k = _KAPPA[name].get(rep, 0)
    return int(k) if k and k > 0 else None


def macro_iscore(S, q, st):
    q = np.where(np.isfinite(q), q, S.max() if len(S) else 0)
    isc = 2 * q + (2 / ALPHA) * np.maximum(S - q, 0)
    return np.mean([isc[st == u].mean() for u in np.unique(st)]) if len(S) else np.inf


def tune_l1(name, d, Dtr, mu_quota_largest, sig, B, rep):
    X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([7, rep, DS.ALL.index(name)])
    perm = rng.permutation(Dtr); k = int(0.6 * len(perm)); D1, D2 = perm[:k], perm[k:]
    mu2 = rf(X[D1], y[D1], 4, 60).predict(X[D2]) + 1e-9 * rng.standard_normal(len(D2))
    S2 = np.abs(y[D2] - mu2) / sig[D2]
    P2 = B(s[D2], st[D2]); pen = B.penalize
    score = {l1: [] for l1 in L1_GRID}
    for h in range(2):
        p = np.random.default_rng([8, rep, h]).permutation(len(D2)); a, b = p[: len(p) // 2], p[len(p) // 2:]
        sel = refsets.quota(mu2[a], mu2[b], st[D2][a], st[D2][b], quotas(st, len(b)), largest=mu_quota_largest)
        if len(sel) == 0:
            continue
        ctx = dict(S_cal=S2[a], Phi_cal=P2[a], Phi_test=P2[b])
        J = b[sel.selected]
        for l1 in L1_GRID:
            q = Mt.sc_geoccp(sel, l1, pen).thresholds(ctx, ALPHA)
            score[l1].append(macro_iscore(S2[J], q, st[D2][J]))
    return min(L1_GRID, key=lambda l: np.mean(score[l]) if score[l] else np.inf)


def one(args):
    name, rep = args
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    cl = cells(name, d)
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])      # == run_main.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    m = len(T); t0 = time.time()
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    S = np.abs(y - mu) / sig
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st); pen = B.penalize
    aid = DS.POLICY[name] == "bottom" and name != "era5"
    l1 = L1_FIXED if L1_FIXED else tune_l1(name, d, Dtr, not aid, sig, B, rep)
    # rules (identical construction to run_main.py)
    cost = 1 + np.linalg.norm(s - s[Dtr].mean(0), axis=1) / np.median(np.linalg.norm(s[Dtr] - s[Dtr].mean(0), axis=1))
    val = mu - mu[Dtr].min() + 0.05 * mu[Dtr].std()
    B_budget = 0.05 * m * cost[Dtr].mean()
    sels = {}
    for f in (0.01, 0.05):
        K = max(1, int(round(f * m)))
        sels[f"top{int(f*100)}"] = refsets.topk(mu[C], mu[T], K)
        sels[f"bottom{int(f*100)}"] = refsets.topk(mu[C], mu[T], K, largest=False)
    Kg = quotas(st, m)
    sels["quota"] = refsets.quota(mu[C], mu[T], st[C], st[T], Kg)
    sels["knapsack"] = refsets.knapsack_greedy(val[C], cost[C], val[T], cost[T], B_budget)
    if aid:
        sels["quota_bottom"] = refsets.quota(mu[C], mu[T], st[C], st[T], Kg, largest=False)
        need = mu[Dtr].max() - mu + 0.05 * mu[Dtr].std()
        sels["knapsack_need"] = refsets.knapsack_greedy(need[C], cost[C], need[T], cost[T], B_budget)
    ctx = dict(S_cal=S[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T],
               s_train=s[Dtr], w_train=np.ones(len(Dtr)), g_cal=st[C], g_test=st[T],
               Phi_cal=Phi[C], Phi_test=Phi[T], rng=np.random.default_rng([3, rep]))
    rows, groups = [], defaultdict(lambda: np.zeros(3))
    for rn, sel in sels.items():
        if len(sel) == 0:
            continue
        J = T[sel.selected]
        M = {"SCP stud": Mt.on_selected(F.split_cp(), sel),
             "ExCP-default": Mt.on_selected(F.excp_default(kappa_of(name, rep)), sel),
             "JOMI stud": Mt.jomi(sel),
             "SC-GeoCCP": Mt.sc_geoccp(sel, l1, pen)}
        if rn.startswith("quota"):
            M["JOMI|state stud"] = Mt.jomi(sel, by_group=True)
        if L1_FIXED:
            M = {"SC-GeoCCP": M["SC-GeoCCP"]}
        for mn, meth in M.items():
            q = meth.thresholds(ctx, ALPHA)
            cov = S[J] <= q; fin = np.isfinite(q); hw = q * sig[J]
            rows.append(dict(dataset=name, rep=rep, rule=rn, method=mn, l1=l1, n_sel=len(J),
                             n_cov=int(cov.sum()), n_inf=int((~fin).sum()),
                             hw_med=float(np.median(hw)), hw_mean_fin=float(hw[fin].mean()) if fin.any() else np.nan,
                             R_mean=float(sel.sizes.mean())))
            for gt, lab in (("state", st[J]), ("cell", cl[J])):
                for u in np.unique(lab):
                    k = lab == u
                    groups[(rn, mn, gt, int(u))] += (k.sum(), cov[k].sum(), (~fin[k]).sum())
    return rows, dict(groups), time.time() - t0


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[3:] or ORDER
    for nm in names:
        t0 = time.time()
        with Pool(W) as p:
            res = p.map(one, [(nm, r) for r in range(R)], chunksize=1)
        pd.DataFrame([x for a, _, _ in res for x in a]).to_csv(os.path.join(OUT, f"scg_main_{nm}{SUFFIX}.csv"), index=False)
        tot = defaultdict(lambda: np.zeros(3))
        for _, g, _ in res:
            for k, v in g.items():
                tot[k] += v
        pd.DataFrame([dict(rule=k[0], method=k[1], gtype=k[2], group=k[3], n=int(v[0]), c=int(v[1]), inf=int(v[2]))
                      for k, v in tot.items()]).to_csv(os.path.join(OUT, f"scg_groups_{nm}{SUFFIX}.csv"), index=False)
        df = pd.DataFrame([x for a, _, _ in res for x in a])
        g = df.groupby(["rule", "method"]); cov = (g.n_cov.sum() / g.n_sel.sum()).unstack(1).round(3)
        print(f"== {nm}: R={R}, {time.time()-t0:.0f}s (mean {np.mean([t for *_, t in res]):.1f}s/rep), "
              f"l1 chosen: {df.drop_duplicates('rep').l1.value_counts().to_dict()}\n{cov}", flush=True)


if __name__ == "__main__":
    main()
