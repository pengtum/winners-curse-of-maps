"""Plot-level crediting on the WP3 carbon selections (Fig. 7e: the old plot-level credit trade-off,
recomputed on the protocol experiment so that both views share maps, audits and selections).

Same maps, seeds, audit / candidate splits and top-K selections as experiments/run_carbon.py (audit share
0.5; top 1% and 5% of candidates). Each selected plot is credited at the lower end of its 90% interval,
floored at 0 (+inf interval -> no credit), as in run_decisions.py; the project's credit is the sum.
Interval methods (as in WP2): Normalized CP, GeoCP (published; bandwidth tuned on the training fold with
run_baselines.tune_local), QRF (Meinshausen 2006), bootstrap ensemble + residual (Heskes 1997), JOMI,
SC-GeoCCP (l1 = 0.95; top 1% only, for the SI), plus the point map (credit = mu).
Per selection and method: plots over-credited, project credit, true total, credited carbon that is real
(sum of min(credit, Y)).
    ~/miniforge3/envs/favela/bin/python -u experiments/run_carbon_units.py <folds> <splits> <workers> [maps...]
writes results/carbon_units_<map>.csv.gz; parts in results/cache/carbon_units_parts/
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma                                     # noqa: E402
from run_baselines import qrf_bounds, bootstrap_bounds, tune_local    # noqa: E402
from run_carbon import DROP, FILE                                     # noqa: E402
from excp_select import refsets, basis, methods as Mt                 # noqa: E402
from excp import family as F                                          # noqa: E402

ALPHA, L1 = 0.1, 0.95
FRACS = (0.01, 0.05)
OUT = DS.FIA_RES
PARTS = os.path.join(DS.CACHE, f"carbon_units_parts{DS.FIA_TAG}")


def task(args):
    mp, f, ns = args
    part = os.path.join(PARTS, f"{FILE[mp]}_f{f:02d}_s{ns}.csv.gz")
    if os.path.exists(part):
        return part
    t0 = time.time()
    d = DS.load(DS.FIA); y, s, st, N = d["y"], d["coords"], d["state"], d["N"]
    X = d["Xrf"][:, [c for c in range(d["Xrf"].shape[1]) if c not in DROP[mp]]]
    perm = np.random.default_rng([5150, f]).permutation(N); ntr = int(0.4 * N)     # == run_carbon.py
    Dtr, rest = perm[:ntr], perm[ntr:]
    mu, sig = fit_mu_sigma(X, y, Dtr, np.random.default_rng([5151, f]))
    lo, _ = qrf_bounds(X[Dtr], y[Dtr], X[rest], int(np.random.default_rng([5153, f]).integers(2 ** 31)))
    rng2 = np.random.default_rng([5154, f])
    h1 = tune_local(X, y, s, Dtr, rng2)["h1"]
    (blo, _), _ = bootstrap_bounds(X, y, Dtr, rest, int(rng2.integers(2 ** 31)))
    QLO, BLO = np.full(N, np.nan), np.full(N, np.nan); QLO[rest], BLO[rest] = lo, blo
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st)
    Ra = np.abs(y - mu); S = Ra / sig
    rows = []
    for sp in range(ns):
        p = np.random.default_rng([5152, f, sp]).permutation(rest)                  # == run_carbon.py
        na = int(0.5 * len(p)); C, T = p[:na], p[na:]
        ctx = dict(S_cal=S[C], w_cal=np.ones(len(C)), w_test=np.ones(len(T)), s_cal=s[C], s_test=s[T], g_cal=st[C],
                   g_test=st[T], Phi_cal=Phi[C], Phi_test=Phi[T], pen=B.penalize, rng=np.random.default_rng([5155, f, sp]))
        ctx_a = dict(ctx, S_cal=Ra[C])
        q = F.split_cp().thresholds(dict(ctx, w_test=np.ones(1)), ALPHA)[0]
        for frac in FRACS:
            K = max(1, int(round(frac * len(T))))
            sel = refsets.topk(mu[C], mu[T], K); J = T[sel.selected]
            L = {"point": mu[J], "Normalized CP": mu[J] - q * sig[J],
                 "GeoCP": mu[J] - Mt.on_selected(F.geocp_v1(h1), sel).thresholds(ctx_a, ALPHA),
                 "QRF": QLO[J], "Bootstrap": BLO[J],
                 "JOMI": mu[J] - Mt.jomi(sel).thresholds(ctx, ALPHA) * sig[J]}
            if frac == 0.01:
                L["SC-GeoCCP"] = mu[J] - Mt.sc_geoccp(sel, L1, B.penalize).thresholds(ctx, ALPHA) * sig[J]
            for m, l in L.items():
                cr = np.where(np.isfinite(l), np.maximum(l, 0.0), 0.0)
                rows.append(dict(map=mp, fold=f, split=sp, frac=frac, K=K, method=m, n_over=int((cr > y[J]).sum()),
                                 credit=float(cr.sum()), total=float(y[J].sum()),
                                 credited_true=float(np.minimum(cr, y[J]).sum())))
    pd.DataFrame(rows).to_csv(part, index=False, float_format="%.6g")
    print(f"  {mp} fold {f}: {time.time()-t0:.0f}s", flush=True)
    return part


def main():
    nf, ns, W = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    maps = sys.argv[4:] or list(DROP)
    t0 = time.time(); os.makedirs(PARTS, exist_ok=True)
    with Pool(W) as pool:
        parts = pool.map(task, [(mp, f, ns) for mp in maps for f in range(nf)], chunksize=1)
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    for mp, g in df.groupby("map", sort=False):
        g.to_csv(os.path.join(OUT, f"carbon_units_{FILE[mp]}.csv.gz"), index=False, float_format="%.6g")
    g = df[df.frac == 0.01].groupby(["map", "method"])
    s = pd.DataFrame({"plots over-credited": g.n_over.sum() / g.K.sum(), "credit / true": g.credit.sum() / g.total.sum(),
                      "project risk": g.apply(lambda x: np.mean(x.credit > x.total))})
    pd.set_option("display.width", 200)
    print("\ntop 1%, audit share 0.5\n" + s.round(3).to_string() + f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
