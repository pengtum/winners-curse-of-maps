"""Mondrian CP baseline and per-split group counts (paper/MONDRIAN_NOISE_DESIGN.md, written before the run).

Same seeds as run_main.py / run_baselines.py: mu, sigma, Dtr / C / T and every selected set are identical (checked: split CP
with the normalized score is recomputed and must match the 'SCP stud' counts of results/main_<ds>.csv; JOMI's per-split state
counts under the quota rules must match results/state_<ds>.csv).
Mondrian CP (Vovk et al.): bins are quantiles of the map's fitted values on its training fold (10 bins main, 20 sensitivity);
a test unit in bin b gets the conformal quantile of the calibration scores in bin b (+inf if the bin is too small). Scores:
normalized |y - mu| / sigma (main) and additive |y - mu| (10 bins). The guarantee is per bin, not per selected unit.
Per-split group counts: for the quota rules, JOMI and normalized CP coverage counts per split and per state / spatial cell
(k-means cells of run_scgeoccp.cells), for the split-level bootstrap of the sampling-noise limit.
    ~/miniforge3/envs/favela/bin/python -u experiments/run_mondrian.py <R> <workers> [datasets...]
writes results/md_main_<ds>.csv, results/md_groups_<ds>.csv (pooled per state / cell) and results/md_split_groups_<ds>.csv
"""
import os, sys, time, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from excp_select import refsets, methods as Mt                        # noqa: E402
from run_main import fit_mu_sigma, quotas, FRACS                      # noqa: E402
from run_baselines import bw_of, q_conf, ALPHA                        # noqa: E402
from run_scgeoccp import cells                                        # noqa: E402

OUT = os.path.join(ROOT, "results")
NAMES = ["camels", "seattle", "voting", "health_stroke", "health_diabetes", "health_bphigh", "health_obesity", "fia", "era5",
         "poverty_lab", "poverty"]
_MAIN, _STATE = {}, {}


def ref(name):
    """Stored per-split counts used for the reproduction checks."""
    if name not in _MAIN:
        m = pd.read_csv(os.path.join(OUT, f"main_{name}.csv"), usecols=["rep", "rule", "method", "n_sel", "n_cov"])
        _MAIN[name] = m[m.method == "SCP stud"].set_index(["rep", "rule"])
        s = pd.read_csv(os.path.join(OUT, f"state_{name}.csv"))
        if "rule" not in s.columns:                       # files from before the bottom quota existed: quota only
            s["rule"] = "quota"
        _STATE[name] = s[s.method == "JOMI stud"].set_index(["rep", "rule", "state"])
    return _MAIN[name], _STATE[name]


def mondrian(S, mu, Dtr, C, T, G):
    """Per test unit: conformal quantile of the calibration scores in its training-fold quantile bin."""
    edges = np.quantile(mu[Dtr], np.linspace(0, 1, G + 1)[1:-1])
    bC, bT = np.searchsorted(edges, mu[C], side="right"), np.searchsorted(edges, mu[T], side="right")
    q = np.array([q_conf(S[C][bC == b]) for b in range(G)])
    return q[bT], np.bincount(bC, minlength=G)


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
    Ra, Ss = np.abs(y - mu), np.abs(y - mu) / sig
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
    # half-widths on all test units
    q10, n10 = mondrian(Ss, mu, Dtr, C, T, 10); q20, n20 = mondrian(Ss, mu, Dtr, C, T, 20); qa10, _ = mondrian(Ra, mu, Dtr, C, T, 10)
    ncal = {"Mondrian CP": n10, "Mondrian CP, 20 bins": n20, "Mondrian CP |res|": n10}         # calibration units per bin
    hw = {"Mondrian CP": q10 * sig[T], "Mondrian CP, 20 bins": q20 * sig[T], "Mondrian CP |res|": qa10 * np.ones(m)}
    hw_check = q_conf(Ss[C]) * sig[T]                                       # split CP, normalized score
    ctx = dict(S_cal=Ss[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T], s_train=s[Dtr],
               w_train=np.ones(len(Dtr)), g_cal=st[C], g_test=st[T], rng=rng)
    M, S_ref = ref(name)
    rows, groups, split_rows = [], defaultdict(lambda: np.zeros(3)), []
    for rn, sel in sels.items():
        if len(sel) == 0:
            continue
        k = sel.selected; J = T[k]; e = np.abs(y[J] - mu[J])
        chk = int((e <= hw_check[k]).sum())
        assert (len(J), chk) == tuple(M.loc[(rep, rn), ["n_sel", "n_cov"]]), (name, rep, rn, len(J), chk)
        for mn, h in hw.items():
            hh = h[k]; cov = e <= hh; fin = np.isfinite(hh)
            rows.append(dict(dataset=name, rep=rep, rule=rn, method=mn, n_sel=len(J), n_cov=int(cov.sum()),
                             n_inf=int((~fin).sum()), hw_med=float(np.median(hh)),
                             hw_mean_fin=float(hh[fin].mean()) if fin.any() else np.nan, sumY=float(y[J].sum()),
                             min_bin_cal=int(ncal[mn].min())))
            for gt, lab in (("state", st[J]), ("cell", cl[J])):
                for u in np.unique(lab):
                    kk = lab == u
                    groups[(rn, mn, gt, int(u))] += (kk.sum(), cov[kk].sum(), (~fin[kk]).sum())
        if rn.startswith("quota"):
            for mn, meth, scale in (("JOMI stud", Mt.jomi(sel), sig), ("SCP stud", None, sig)):
                if meth is None:
                    hh = hw_check[k]
                else:
                    hh = meth.thresholds(ctx, ALPHA) * scale[J]
                cov = e <= hh; fin = np.isfinite(hh)
                for gt, lab in (("state", st[J]), ("cell", cl[J])):
                    for u in np.unique(lab):
                        kk = lab == u
                        split_rows.append(dict(dataset=name, rep=rep, rule=rn, method=mn, gtype=gt, group=int(u),
                                               n=int(kk.sum()), c=int(cov[kk].sum()), inf=int((~fin[kk]).sum())))
                        if mn == "JOMI stud" and gt == "state":
                            r = S_ref.loc[(rep, rn, int(u))]
                            assert (int(kk.sum()), int(cov[kk].sum())) == (int(r.n_sel), int(r.n_cov)), (name, rep, rn, u)
    return rows, dict(groups), split_rows, time.time() - t0


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[3:] or NAMES
    for nm in names:
        DS.load(nm)
    for nm in names:
        t0 = time.time()
        with Pool(W) as p:
            res = p.map(one, [(nm, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a, *_ in res for x in a])
        df.to_csv(os.path.join(OUT, f"md_main_{nm}.csv"), index=False)
        tot = defaultdict(lambda: np.zeros(3))
        for _, g, _, _ in res:
            for k, v in g.items():
                tot[k] += v
        pd.DataFrame([dict(rule=k[0], method=k[1], gtype=k[2], group=k[3], n=int(v[0]), c=int(v[1]), inf=int(v[2]))
                      for k, v in tot.items()]).to_csv(os.path.join(OUT, f"md_groups_{nm}.csv"), index=False)
        pd.DataFrame([x for _, _, b, _ in res for x in b]).to_csv(os.path.join(OUT, f"md_split_groups_{nm}.csv"), index=False)
        g = df.groupby(["rule", "method"]); cov = (g.n_cov.sum() / g.n_sel.sum()).unstack(1).round(3)
        print(f"== {nm}: R={R}, {time.time() - t0:.0f}s (checks passed)\n"
              f"{cov.loc[[r for r in ['top1', 'top5', 'bottom5', 'quota', 'quota_bottom'] if r in cov.index]]}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
