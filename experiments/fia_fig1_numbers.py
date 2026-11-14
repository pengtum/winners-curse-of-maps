"""Numbers printed in Fig. 1b: split CP coverage of all FIA test plots and of the map's highest 5%, pooled over 100 splits.

Re-creates each split as experiments/run_main.py:one() does (experiments/heldout_r2.py's stubs; the project's own
run_main.fit_mu_sigma), computes split CP (score |y - mu|, alpha = 0.1) on all test plots and on the 5% with the highest
predictions, and asserts that the selected-plot counts equal the stored 'top5' / 'SCP |res|' rows of results/main_<FIA>.csv.
A descriptive number for the figure, not a new analysis.

    ~/miniforge3/envs/favela/bin/python -I -B experiments/fia_fig1_numbers.py 100 4
writes results/fig1_fia_split_cp.csv (one row per split)
"""
import os
import sys

sys.dont_write_bytecode = True
EXP = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(EXP)
if EXP not in sys.path:
    sys.path.insert(0, EXP)

import heldout_r2 as H          # noqa: E402  stubs excp/excp_select; imports run_main and data
import numpy as np              # noqa: E402
import pandas as pd             # noqa: E402
from multiprocessing import Pool  # noqa: E402

RM, DS = H.RM, H.DS
NAME, ALPHA = DS.FIA, 0.1


def one(rep):
    d = DS.load(NAME); N = d["N"]; X, y = d["Xrf"], d["y"]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(NAME)])           # run_main.py one()
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    mu, _ = RM.fit_mu_sigma(X, y, Dtr, rng)
    R = np.abs(y - mu); n = len(C); kq = int(np.ceil((1 - ALPHA) * (n + 1)))
    q = np.sort(R[C])[kq - 1]
    K = max(1, int(round(0.05 * len(T))))
    J = T[np.argsort(-mu[T], kind="stable")[:K]]
    return dict(rep=rep, m=len(T), n_cov_all=int((R[T] <= q).sum()), K=K, n_cov_top5=int((R[J] <= q).sum()))


def main():
    reps, W = int(sys.argv[1]), int(sys.argv[2])
    with Pool(W) as pool:
        df = pd.DataFrame(pool.map(one, range(reps), chunksize=1))
    t = pd.read_csv(os.path.join(PROJ, "results", f"main_{NAME}.csv"))
    st = t[t.rule.eq("top5") & t.method.eq("SCP |res|")].set_index("rep")
    bad = [r for r in df.rep if (st.loc[r, "n_sel"], st.loc[r, "n_cov"]) != tuple(df.set_index("rep").loc[r, ["K", "n_cov_top5"]])]
    assert not bad, f"selected-plot counts differ from results/main_{NAME}.csv in splits {bad[:10]}"
    df.to_csv(os.path.join(PROJ, "results", "fig1_fia_split_cp.csv"), index=False)
    s = df.sum()
    print(f"{NAME}: split CP coverage, all test plots {s.n_cov_all / s.m:.4f} ({int(s.n_cov_all)}/{int(s.m)}); "
          f"highest 5% {s.n_cov_top5 / s.K:.4f} ({int(s.n_cov_top5)}/{int(s.K)}); {len(df)} splits")


if __name__ == "__main__":
    main()
