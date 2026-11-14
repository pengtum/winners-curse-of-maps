"""WP9: backbone invariance -- the same selection audit on ResNet and frozen-DOFA (GeoFM) outputs.

Data: GROUND's per-sample score files (read-only), 0-Xiayin_research/15_RobustGeoCP/
real_world_study/results_8ds/<ds>[_dofa]_scores.npz: y, y_pred (backbone trained on the fixed
`in_train` fold), lon/lat, region, 32-d embedding. Only files loadable WITHOUT pickle are used.
Per combination and repetition, the non-training pool is split at random into structure 25% /
calibration 37.5% / test 37.5%:
  sigma   RF on the structure fold: |y - y_pred| ~ (embedding, coordinates)  (fixed given it)
  GeoCP   published version, |res| score, bandwidth tuned on the structure fold
  basis   SC-GeoCCP multiscale bumps from structure coordinates + region indicators, l1 = 0.95
Coordinates are 3-D Earth-centred (km) so kernels and k-means are valid at global scale.
Rules: top/bottom 1/5/20% and a region quota in the policy direction (aid = bottom wealth,
carbon = top biomass, rescue = top flood fraction, tree planting = bottom canopy).
Methods: SCP |res|, SCP stud, GeoCP, BY stud, JOMI stud, JOMI|region (quota), SC-GeoCCP (not on 20% rules).

    ~/miniforge3/envs/favela/bin/python -u experiments/run_geofm.py <R> <workers> [combos...]
writes results/fm_main_<combo>.csv and results/fm_groups_<combo>.csv
"""
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
from run_main import quotas, iscore                                   # noqa: E402
from excp_select import refsets, basis, methods as Mt                 # noqa: E402
from excp import family as F                                          # noqa: E402

GROUND = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "0-Xiayin_research", "15_RobustGeoCP",
                      "real_world_study", "results_8ds")      # found from the project location
ALPHA = 0.1
OUT = os.path.join(ROOT, "results")
COMBOS = {  # combo: (file, backbone, dataset, policy direction)
    "poverty_resnet": ("poverty_scores.npz", "ResNet", "poverty", "bottom"),
    "poverty_dofa": ("poverty_dofa_scores.npz", "DOFA", "poverty", "bottom"),
    "m3leo_resnet": ("m3leo_scores.npz", "ResNet", "m3leo", "top"),
    "m3leo_dofa": ("m3leo_dofa_scores.npz", "DOFA", "m3leo", "top"),
    "floods_resnet": ("urbansarfloods_scores.npz", "ResNet", "floods", "top"),
    "floods_dofa": ("urbansarfloods_dofa_scores.npz", "DOFA", "floods", "top"),
    "usavars_dofa": ("usavars_dofa_scores.npz", "DOFA", "usavars", "bottom"),
}
LABEL = {"poverty": "DHS wealth (PovertyMap)", "m3leo": "Biomass (M3LEO)", "floods": "Flood fraction (UrbanSARFloods)",
         "usavars": "Tree cover (USAVars)"}
L1 = 0.95
_D = {}


def ecef_km(lat, lon, R=6371.0):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.c_[R * np.cos(la) * np.cos(lo), R * np.cos(la) * np.sin(lo), R * np.sin(la)]


def load(combo):
    if combo not in _D:
        f, bb, ds, pol = COMBOS[combo]
        z = np.load(os.path.join(GROUND, f), allow_pickle=False)
        pool = ~z["in_train"]
        d = dict(y=z["y"][pool].astype(float), mu=z["y_pred"][pool].astype(float), emb=z["embed"][pool],
                 s=ecef_km(z["lat"][pool], z["lon"][pool]), region=z["region"][pool])
        if ds == "poverty":            # country labels live in the ResNet file (same order, verified)
            zr = np.load(os.path.join(GROUND, "poverty_scores.npz"), allow_pickle=False)
            d["group"] = zr["region_country"][~zr["in_train"]]
        else:
            d["group"] = d["region"]
        from sklearn.cluster import KMeans
        d["cell"] = KMeans(int(min(20, max(4, pool.sum() // 150))), n_init=4, random_state=0).fit_predict(d["s"])
        _D[combo] = d
    return _D[combo]


def tune_geocp(R, s, rng):
    base = float(np.median(cKDTree(s).query(s, k=min(11, len(s)))[0][:, -1]))
    hs = base * np.array([1, 2, 4, 8, 16, 32]); sc = []
    halves = [rng.permutation(len(R)) for _ in range(2)]
    for h in hs:
        v = []
        for p in halves:
            a, b = p[: len(p) // 2], p[len(p) // 2:]
            q = F.geocp_v1(h).thresholds(dict(S_cal=R[a], w_cal=np.ones(len(a)), w_test=np.ones(len(b)),
                                              s_cal=s[a], s_test=s[b]), ALPHA)
            v.append(iscore(R[b], q))
        sc.append(np.mean(v))
    return hs[int(np.argmin(sc))]


def one(args):
    combo, rep = args
    d = load(combo); _, bb, ds, pol = COMBOS[combo]
    y, s, g = d["y"], d["s"], d["group"]; N = len(y)
    rng = np.random.default_rng([606, rep, list(COMBOS).index(combo)])
    perm = rng.permutation(N); a, b = int(0.25 * N), int(0.625 * N)
    Ds, C, T = perm[:a], perm[a:b], perm[b:]
    m = len(T); t0 = time.time()
    mu = d["mu"] + 1e-9 * rng.standard_normal(N)
    Ra = np.abs(y - mu)
    feat = np.c_[d["emb"], s / 1000.0]
    sm = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_features=0.6, n_jobs=1,
                               random_state=int(rng.integers(2 ** 31))).fit(feat[Ds], Ra[Ds])
    sig = np.maximum(sm.predict(feat), 0.1 * Ra[Ds].mean() + 1e-9)
    Ss = Ra / sig
    h1 = tune_geocp(Ra[Ds], s[Ds], rng)
    B = basis.Multiscale(s[Ds]).with_units(g); Phi = B(s, g); pen = B.penalize
    ctx = dict(S_cal=Ss[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T],
               g_cal=g[C], g_test=g[T], Phi_cal=Phi[C], Phi_test=Phi[T], rng=rng)
    ctx_a = dict(ctx, S_cal=Ra[C])
    sels = {}
    for f in (0.01, 0.05, 0.20):
        K = max(1, int(round(f * m)))
        sels[f"top{int(f*100)}"] = refsets.topk(mu[C], mu[T], K)
        sels[f"bottom{int(f*100)}"] = refsets.topk(mu[C], mu[T], K, largest=False)
    sels["quota"] = refsets.quota(mu[C], mu[T], g[C], g[T], quotas(g, m), largest=(pol == "top"))
    rows, groups = [], defaultdict(lambda: np.zeros(3))
    for rn, sel in sels.items():
        if len(sel) == 0:
            continue
        J = T[sel.selected]
        M = {"SCP |res|": (Mt.on_selected(F.split_cp(), sel), ctx_a, Ra, 1.0),
             "SCP stud": (Mt.on_selected(F.split_cp(), sel), ctx, Ss, sig),
             "GeoCP": (Mt.on_selected(F.geocp_v1(h1), sel), ctx_a, Ra, 1.0),
             "BY stud": (Mt.by(sel, m), ctx, Ss, sig),
             "JOMI stud": (Mt.jomi(sel), ctx, Ss, sig)}
        if rn not in ("top20", "bottom20"):          # LP cost; the 20% rules barely distort
            M["SC-GeoCCP"] = (Mt.sc_geoccp(sel, L1, pen), ctx, Ss, sig)
        if rn == "quota":
            M["JOMI|region stud"] = (Mt.jomi(sel, by_group=True), ctx, Ss, sig)
        for mn, (meth, c, S, scale) in M.items():
            q = meth.thresholds(c, ALPHA)
            cov = S[J] <= q; fin = np.isfinite(q); hw = q * (scale[J] if np.ndim(scale) else scale)
            rows.append(dict(combo=combo, dataset=ds, backbone=bb, policy=pol, rep=rep, rule=rn, method=mn,
                             n_sel=len(J), n_cov=int(cov.sum()), n_inf=int((~fin).sum()),
                             hw_med=float(np.median(hw)), hw_mean_fin=float(hw[fin].mean()) if fin.any() else np.nan,
                             R_mean=float(sel.sizes.mean()), h1=h1))
            for gt, lab in (("region", g[J]), ("cell", d["cell"][J])):
                for u in np.unique(lab):
                    k = lab == u
                    groups[(rn, mn, gt, int(u))] += (k.sum(), cov[k].sum(), (~fin[k]).sum())
    return rows, dict(groups), time.time() - t0


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    combos = sys.argv[3:] or list(COMBOS)
    for cb in combos:
        t0 = time.time()
        with Pool(W) as p:
            res = p.map(one, [(cb, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a, _, _ in res for x in a])
        df.to_csv(os.path.join(OUT, f"fm_main_{cb}.csv"), index=False)
        tot = defaultdict(lambda: np.zeros(3))
        for _, gg, _ in res:
            for k, v in gg.items():
                tot[k] += v
        pd.DataFrame([dict(rule=k[0], method=k[1], gtype=k[2], group=k[3], n=int(v[0]), c=int(v[1]), inf=int(v[2]))
                      for k, v in tot.items()]).to_csv(os.path.join(OUT, f"fm_groups_{cb}.csv"), index=False)
        gb = df.groupby(["rule", "method"]); cov = (gb.n_cov.sum() / gb.n_sel.sum()).unstack(1).round(3)
        print(f"== {cb}: R={R}, {time.time()-t0:.0f}s (mean {np.mean([t for *_, t in res]):.1f}s/rep)\n"
              f"{cov.loc[['top5', 'bottom5', 'quota']]}", flush=True)


if __name__ == "__main__":
    main()
