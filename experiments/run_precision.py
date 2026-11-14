"""Ranking precision and coverage at the selected units (paper/PRECISION_COVERAGE_DESIGN.md, frozen 2026-11-09).

For the 11 random-forest settings of Fig. 2a and the 5% policy-direction rule, re-creates each split exactly as
experiments/run_main.py:one() does (experiments/heldout_r2.py's stubs; the project's own run_main.fit_mu_sigma), selects
the K = round(0.05 m) test units with the most extreme predictions in the policy direction, and records per split
  n_in5   selected units whose true value is among the K most extreme true values of the test units (ties inclusive),
  n_in20  ... among the round(0.2 m) most extreme,
  n_cov   selected units covered by split CP (score |y - mu|, alpha = 0.1), split by n_in5 membership.
Checks (assert): rep 0 selection = results/units_<name>_rep0.npz '<rule>|idx' and mu = stored mu; every split's
split CP n_sel and n_cov = the stored 'SCP |res|' row of results/main_<name>.csv.

    ~/miniforge3/envs/favela/bin/python -I -B experiments/run_precision.py 100 4
writes results/precision_coverage_per_rep.csv and results/precision_coverage_summary.csv
"""
import os
import sys

sys.dont_write_bytecode = True
EXP = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(EXP)
if EXP not in sys.path:
    sys.path.insert(0, EXP)

import heldout_r2 as H          # noqa: E402  stubs excp/excp_select (nothing under design_based_spatial_cp is read)
import numpy as np              # noqa: E402
import pandas as pd             # noqa: E402
from multiprocessing import Pool  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

RM, DS = H.RM, H.DS
SETTINGS = list(DS.DISPLAY)     # the 11 random-forest settings of Fig. 2a
ALPHA = 0.1
RES = os.path.join(PROJ, "results")


def rule_of(name):
    return ("top" if DS.POLICY[name] == "top" else "bottom") + "5"


def one(args):
    name, rep = args
    d = DS.load(name); N = d["N"]; X, y = d["Xrf"], d["y"]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # run_main.py one()
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    m = len(T)
    mu, _ = RM.fit_mu_sigma(X, y, Dtr, rng)
    sgn = 1.0 if DS.POLICY[name] == "top" else -1.0
    K, K20 = max(1, int(round(0.05 * m))), max(1, int(round(0.20 * m)))
    J = T[np.argsort(-sgn * mu[T], kind="stable")[:K]]                    # refsets.topk on mu[T]; mu carries a 1e-9 jitter
    v = np.sort(sgn * y[T])[::-1]
    in5, in20 = sgn * y[J] >= v[K - 1], sgn * y[J] >= v[K20 - 1]
    R = np.abs(y - mu); n = len(C); kq = int(np.ceil((1 - ALPHA) * (n + 1)))
    q = np.sort(R[C])[kq - 1] if kq <= n else np.inf
    cov = R[J] <= q
    out = dict(setting=name, rep=rep, m=m, K=K, n_in5=int(in5.sum()), n_in20=int(in20.sum()), n_cov=int(cov.sum()),
               n_cov_in5=int((cov & in5).sum()), n_cov_out5=int((cov & ~in5).sum()))
    if rep == 0:
        z = np.load(os.path.join(RES, f"units_{name}_rep0.npz"), allow_pickle=False)
        out.update(rep0_sel_match=bool(set(z[f"{rule_of(name)}|idx"].tolist()) == set(J.tolist())),
                   rep0_mu_maxdiff=float(np.max(np.abs(z["mu"] - mu))))
    return out


def main():
    reps, W = int(sys.argv[1]), int(sys.argv[2])
    jobs = [(nm, r) for nm in SETTINGS for r in range(reps)]
    with Pool(W) as pool:
        df = pd.DataFrame(pool.map(one, jobs, chunksize=1))
    df.to_csv(os.path.join(RES, "precision_coverage_per_rep.csv"), index=False)
    r0 = df[df.rep.eq(0)]
    assert r0.rep0_sel_match.all() and (r0.rep0_mu_maxdiff == 0).all(), r0[["setting", "rep0_sel_match", "rep0_mu_maxdiff"]]
    r2 = pd.read_csv(os.path.join(RES, "maps_r2_rf_summary.csv"), index_col=0)
    rows, bad = [], []
    for nm in SETTINGS:
        t = pd.read_csv(os.path.join(RES, f"main_{nm}.csv"))
        t = t[t.rule.eq(rule_of(nm))]
        st = t[t.method.eq("SCP |res|")].set_index("rep")
        mine = df[df.setting.eq(nm)].set_index("rep")
        bad += [(nm, r) for r in mine.index if (st.loc[r, "n_sel"], st.loc[r, "n_cov"]) != (mine.loc[r, "K"], mine.loc[r, "n_cov"])]
        g = t.groupby("method")[["n_sel", "n_cov"]].sum()
        s = mine.sum()
        rows.append(dict(setting=nm, label=DS.LABEL[nm], rule=rule_of(nm), splits=len(mine), selected=int(s.K),
                         r2_test_mean=r2.loc[nm, "r2_test_mean"],
                         P5=s.n_in5 / s.K, P20=s.n_in20 / s.K,
                         cov_split=s.n_cov / s.K,
                         cov_split_true5=s.n_cov_in5 / s.n_in5 if s.n_in5 else np.nan,
                         cov_split_not5=s.n_cov_out5 / (s.K - s.n_in5) if s.K > s.n_in5 else np.nan,
                         cov_normalized=g.loc["SCP stud", "n_cov"] / g.loc["SCP stud", "n_sel"],
                         cov_jomi=g.loc["JOMI stud", "n_cov"] / g.loc["JOMI stud", "n_sel"],
                         P5_split_min=(mine.n_in5 / mine.K).min(), P5_split_max=(mine.n_in5 / mine.K).max()))
    assert not bad, f"split CP counts differ from results/main_*.csv: {bad[:10]}"
    sm = pd.DataFrame(rows)
    sm.to_csv(os.path.join(RES, "precision_coverage_summary.csv"), index=False)
    med = sm.P5.median()
    n_main = int(((sm.P20 >= 0.80) & (sm.cov_split < 0.85)).sum())
    rho = spearmanr(sm.P5, (sm.cov_split - 0.90).abs()).statistic
    rule_a = sm[(sm.P5 >= med) & (sm.cov_split < 0.85)]
    pd.set_option("display.width", 250)
    print(sm.round(3).to_string(index=False))
    print(f"\nmain summary: settings with P20 >= 0.80 and split CP coverage < 0.85: {n_main} of {len(sm)}")
    print(f"descriptive: Spearman rho(P5, |split CP coverage - 0.90|) = {rho:.3f} (n = {len(sm)}, no test)")
    print(f"median P5 = {med:.3f}; rule (A) holds for: {rule_a.setting.tolist()}")


if __name__ == "__main__":
    main()
