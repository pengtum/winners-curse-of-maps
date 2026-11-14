"""Held-out R^2 of every map in the paper (Methods, "Maps"; results/maps_r2_*.csv).

Random-forest settings (Fig. 4, experiments/data.py ALL): re-creates each split exactly as
experiments/run_main.py:one() does (rng = default_rng([2027, rep, DS.ALL.index(name)]), 40/30/30
permutation) and calls the project's own run_main.fit_mu_sigma (imported, not reimplemented).
For rep 0 the refitted mu is checked against the stored results/units_<name>_rep0.npz mu.
R^2 = 1 - SSE/SST on the 30% test units T (SST around mean of y[T]); also on C u T (60%).
Extra (not a paper setting): FIA site / site+height feature sets on the SAME main splits, using
the column drops of experiments/run_carbon.py:52-53.

Deep-backbone settings (Supplementary Fig. 9a, experiments/run_geofm.py COMBOS): y vs y_pred of the GROUND score
files (allow_pickle=False), on the whole non-training pool and on run_geofm's test units
(rng = default_rng([606, rep, combo index]), test = perm[int(0.625 N):]).

The excp / excp_select packages (which live under design_based_spatial_cp) are stubbed so that
nothing under that directory is read; fit_mu_sigma does not use them.

    ~/miniforge3/envs/favela/bin/python -I -B experiments/heldout_r2.py 10 4
writes results/maps_r2_rf_per_rep.csv, results/maps_r2_rf_summary.csv, results/maps_r2_deep_backbone.csv
"""
import os, sys, time, types
sys.dont_write_bytecode = True

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(PROJ, "experiments")
GROUND = os.path.join(os.path.dirname(os.path.dirname(PROJ)), "0-Xiayin_research", "15_RobustGeoCP",
                      "real_world_study", "results_8ds")                # read-only score files of the GROUND study

# stub the packages run_main imports at module level but fit_mu_sigma never uses
for nm in ("excp", "excp.family", "excp_select", "excp_select.refsets", "excp_select.methods"):
    sys.modules.setdefault(nm, types.ModuleType(nm))
sys.modules["excp"].family = sys.modules["excp.family"]
sys.modules["excp_select"].refsets = sys.modules["excp_select.refsets"]
sys.modules["excp_select"].methods = sys.modules["excp_select.methods"]
if EXP not in sys.path:
    sys.path.insert(0, EXP)

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from multiprocessing import Pool                                      # noqa: E402
import run_main as RM                                                 # noqa: E402  (project code)
import data as DS                                                     # noqa: E402  (project code)

# experiments/run_carbon.py:52-53 (Xrf columns dropped per carbon map type)
DROP = {"full": [], "site": [6, 7, 8, 9, 11], "site+height": [6, 7, 8, 11]}

COMBOS = {  # experiments/run_geofm.py:38-46 (order defines the seed index)
    "poverty_resnet": "poverty_scores.npz", "poverty_dofa": "poverty_dofa_scores.npz",
    "m3leo_resnet": "m3leo_scores.npz", "m3leo_dofa": "m3leo_dofa_scores.npz",
    "floods_resnet": "urbansarfloods_scores.npz", "floods_dofa": "urbansarfloods_dofa_scores.npz",
    "usavars_dofa": "usavars_dofa_scores.npz"}


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def one(args):
    name, rep, feats = args
    d = DS.load(name); N = d["N"]; X, y = d["Xrf"], d["y"]
    if feats != "full":
        X = X[:, [c for c in range(X.shape[1]) if c not in DROP[feats]]]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # run_main.py:127
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)        # run_main.py:128
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    t0 = time.time()
    mu, sig = RM.fit_mu_sigma(X, y, Dtr, rng)                                # run_main.py:132
    out = dict(setting=name if feats == "full" else f"{name}[{feats}]", rep=rep, N=N, p=X.shape[1],
               n_train=len(Dtr), n_test=len(T), r2_test=r2(y[T], mu[T]),
               r2_cal_test=r2(y[np.r_[C, T]], mu[np.r_[C, T]]), r2_train=r2(y[Dtr], mu[Dtr]),
               sec=time.time() - t0)
    if rep == 0 and feats == "full":
        z = np.load(os.path.join(PROJ, "results", f"units_{name}_rep0.npz"), allow_pickle=False)
        out.update(stored_T_match=bool(np.array_equal(np.sort(z["T"]), np.sort(T))),
                   stored_mu_maxdiff=float(np.max(np.abs(z["mu"] - mu))),
                   stored_r2_test=r2(z["y"][z["T"]], z["mu"][z["T"]]))
    return out


def ground(reps):
    rows = []
    for k, (combo, f) in enumerate(COMBOS.items()):
        try:
            z = np.load(os.path.join(GROUND, f), allow_pickle=False)
            pool = ~z["in_train"].astype(bool)
            y, p = z["y"][pool].astype(float), z["y_pred"][pool].astype(float)
        except Exception as e:                                               # noqa: BLE001
            rows.append(dict(setting=combo, error=repr(e))); continue
        N = len(y); te = []
        for rep in range(reps):
            rng = np.random.default_rng([606, rep, k])                       # run_geofm.py:95-97
            T = rng.permutation(N)[int(0.625 * N):]
            te.append(r2(y[T], p[T]))
        rows.append(dict(setting=combo, n_total=int(len(z["y"])), n_pool=N,
                         r2_pool=r2(y, p), r2_test_split0=te[0], r2_test_mean=float(np.mean(te)),
                         r2_test_min=float(np.min(te)), r2_test_max=float(np.max(te)),
                         r2_train_fold=r2(z["y"][~pool].astype(float), z["y_pred"][~pool].astype(float))))
    return pd.DataFrame(rows)


def main():
    reps, W = int(sys.argv[1]), int(sys.argv[2])
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(PROJ, "results")
    t0 = time.time()
    jobs = [(nm, r, "full") for nm in DS.ALL if nm not in DS.DOFA for r in range(reps)]   # DOFA arm: its own design
    jobs += [(DS.FIA, r, f) for f in ("site", "site+height") for r in range(reps)]
    with Pool(W) as pool:
        res = pool.map(one, jobs, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(out_dir, "maps_r2_rf_per_rep.csv"), index=False)
    g = df.groupby("setting", sort=False)
    summ = pd.DataFrame({"N": g.N.first(), "p_features": g.p.first(), "n_test": g.n_test.first(),
                         "r2_test_split0": g.apply(lambda t: t.loc[t.rep == 0, "r2_test"].iloc[0]),
                         "r2_test_mean": g.r2_test.mean(), "r2_test_min": g.r2_test.min(),
                         "r2_test_max": g.r2_test.max(), "r2_cal_test_mean": g.r2_cal_test.mean(),
                         "r2_train_mean": g.r2_train.mean(), "sec_per_fit": g.sec.mean()})
    chk = df[df.rep.eq(0) & df.stored_r2_test.notna()].set_index("setting")[
        ["stored_T_match", "stored_mu_maxdiff", "stored_r2_test"]]
    summ = summ.join(chk)
    summ.to_csv(os.path.join(out_dir, "maps_r2_rf_summary.csv"))
    gd = ground(reps)
    gd.to_csv(os.path.join(out_dir, "maps_r2_deep_backbone.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"reps={reps} workers={W} total {time.time() - t0:.0f}s\n")
    print(summ.round(4).to_string()); print()
    print(gd.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
