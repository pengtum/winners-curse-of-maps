"""Carbon crediting with an operational biomass map (paper/BIOMASS_PRODUCT_DESIGN.md, frozen 2026-10-31).

The map is ESA CCI Biomass v7.0 for 2020 at the FIA plots (data/fia_cci_v7_2020.csv, experiments/build_fia_cci.py):
mu = product AGB, error layer s = product AGB_SD; 'win' = means over a 1 km window around the public coordinates (main),
'ctr' = the centre pixel (sensitivity). Nothing is trained. Truth: FIA above-ground live-tree biomass Y (Mg/ha).
Design (section 4): 20 random subsets of 6,000 plots (the 20 'maps' of Fig. 7) x 100 random audits x audit share
0.2 / 0.5 / 0.8 x the highest 0.5-10% of candidates by mu. Estimates and protocol formulas are those of
experiments/run_carbon.py (error layer independent and fully correlated, survey-calibrated, selection-aware with and
without the VCS waiver, exact pool-lemma bound) and the ten-stratum domain estimate of experiments/run_carbon_domain.py,
with strata cut at deciles of mu over the plots outside the subset. CARBON_SELECT=random: the same K candidates
chosen at random (the random-selection control, all 20 subsets). Ties in mu (frequent for the integer centre pixel)
are broken by a fixed random jitter of 1e-6 Mg/ha.
    ~/miniforge3/envs/favela/bin/python -u experiments/run_carbon_product.py <subsets> <audits> <workers> [win ctr]
writes results/fia_db/product/carbon_product_<win|ctr>[_random].csv.gz (one row per selection); parts are cached in
results/cache/carbon_product_parts/ so a rerun skips finished subsets.
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
import run_carbon as RC                                               # noqa: E402
import run_carbon_domain as RD                                        # noqa: E402
from excp_select import refsets, totals as TOT                        # noqa: E402

OUT = os.path.join(DS.FIA_RES, "product")
PARTS = os.path.join(DS.CACHE, "carbon_product_parts")
RANDOM = os.environ.get("CARBON_SELECT", "map") == "random"
TAG = "_random" if RANDOM else ""
NSUB = 6000
COLS = {"win": ("agb", "sd"), "ctr": ("agb_ctr", "sd_ctr")}


def load(v):
    z = pd.read_csv(os.path.join(ROOT, "data", "fia_cci_v7_2020.csv"))
    a, s = COLS[v]
    mu = z[a].to_numpy(float) + 1e-6 * np.random.default_rng(6153).random(len(z))
    return z.Y.to_numpy(float), mu, z[s].to_numpy(float)


def credits(mu, e, s, J, R, C):
    K, nR = len(J), len(R); L = nR + K; est = mu[J].sum()
    se, se_corr = np.sqrt((s[J] ** 2).sum()), s[J].sum()
    ma, se_ma = est + K * e[C].mean(), e[C].std(ddof=1) * np.sqrt(K + K ** 2 / len(C))
    sa, se_sa = (est + K * e[R].mean(), K * e[R].std(ddof=1) * np.sqrt(L / (nR * K))) if nR >= 2 else (0.0, np.inf)
    out = {"point": est,
           "VCS map": RC.vcs(est, se), "ART map": RC.art(est, se),
           "VCS map (corr.)": RC.vcs(est, se_corr), "ART map (corr.)": RC.art(est, se_corr),
           "VCS model-assisted": RC.vcs(ma, se_ma), "ART model-assisted": RC.art(ma, se_ma),
           "ours: VCS formula, sel.-aware": sa - RC.Z_VCS * se_sa, "ours: ART formula, sel.-aware": sa - RC.Z_ART * se_sa,
           "ours: VCS formula + waiver": RC.vcs(sa, se_sa) if nR >= 2 else 0.0}
    for p, d in RC.DELTA.items():
        out[f"ours: exact, risk {p}"] = est + TOT.lower_total(e[R], K, -mu[J].max(), RC.env(L, K, d))
    aux = dict(se_map=se, se_map_corr=se_corr, ma=ma, se_ma=se_ma, sa=sa, se_sa=se_sa)
    return {k: max(0.0, float(v)) for k, v in out.items()}, {k: float(v) for k, v in aux.items()}


def task(args):
    v, f, ns = args
    part = os.path.join(PARTS, f"{v}{TAG}_f{f:02d}_s{ns}.csv.gz")
    if os.path.exists(part):
        return part
    t0 = time.time()
    y, mu, s = load(v); N = len(y); e = y - mu
    perm = np.random.default_rng([6150, f]).permutation(N)
    sub, out_ = perm[:NSUB], perm[NSUB:]
    h = RD.strata(mu, out_, 10)                                      # deciles of mu outside the subset
    r2 = 1 - (e[sub] ** 2).sum() / ((y[sub] - y[sub].mean()) ** 2).sum()
    rows = []
    for k in range(ns):
        p = np.random.default_rng([6152, f, k]).permutation(sub)
        u = np.random.default_rng([6161, f, k]).random(N) if RANDOM else mu
        for a in RC.AUDIT:
            na = int(a * len(p)); C, T = p[:na], p[na:]
            grp = RD.merged(h, C, 10)
            for frac in RC.FRACS:
                K = max(1, int(round(frac * len(T))))
                sel = refsets.topk(u[C], u[T], K)
                J = T[sel.selected]; R = C[sel.member[0]]
                cr, aux = credits(mu, e, s, J, R, C)
                est, se, nu = RD.domain(mu, e, h, grp, J, C)
                rows.append(dict(map=v, fold=f, split=k, audit=a, frac=frac, K=K, nR=len(R), r2=r2,
                                 total=float(y[J].sum()), **cr, **aux, dom10_est=est, dom10_se=se, dom10_n=nu,
                                 **{"VCS domain10": max(0.0, RC.vcs(est, se)),
                                    "VCS domain10, formula": max(0.0, est - RC.Z_VCS * se),
                                    "ART domain10": max(0.0, RC.art(est, se))}))
    os.makedirs(PARTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False, float_format="%.6g")
    print(f"  {v}{TAG} subset {f}: R2={r2:.3f}, {time.time() - t0:.0f}s", flush=True)
    return part


def main():
    nf, ns, W = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    vs = sys.argv[4:] or ["win", "ctr"]
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    with Pool(W) as pool:
        parts = pool.map(task, [(v, f, ns) for v in vs for f in range(nf)], chunksize=1)
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    for v, g in df.groupby("map", sort=False):
        g.to_csv(os.path.join(OUT, f"carbon_product_{v}{TAG}.csv.gz"), index=False, float_format="%.6g")
    print(f"wrote {len(df)} rows for {vs}{TAG} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
