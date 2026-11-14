"""Does the package reproduce the code behind the manuscript's results? (paper/PACKAGE_BENCHMARK_DESIGN.md)

Same data, map, split and rules (run_scgeoccp.py, repetition 0), through excp_select and through mapselect: selected units, reference
sets, JOMI thresholds, SC-GeoCCP thresholds in both forms, and the programme's objective. The published runs used the primal
form with dense constraints; the objective, not the vertex, is what must agree (Theorem 1 holds for any minimizer).
    ~/miniforge3/envs/favela/bin/python experiments/check_package_parity.py
writes results/fia_db/package/parity.csv (no unit-level data)
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "package", "src")]
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma, quotas                             # noqa: E402
from excp_select import refsets, basis, methods as Mt                 # noqa: E402
import mapselect as ms                                                # noqa: E402
from mapselect.calibrate import _primal_beta, _dual_beta, objective   # noqa: E402

ALPHA, L1 = 0.1, 0.95


def one(name, nlp):
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([2027, 0, DS.ALL.index(name)])
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    m = len(T); mu, sig = fit_mu_sigma(X, y, Dtr, rng); S = np.abs(y - mu) / sig
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st); pen = B.penalize
    cost = 1 + np.linalg.norm(s - s[Dtr].mean(0), axis=1) / np.median(np.linalg.norm(s[Dtr] - s[Dtr].mean(0), axis=1))
    val = mu - mu[Dtr].min() + 0.05 * mu[Dtr].std(); Bud = 0.05 * m * cost[Dtr].mean(); Kg = quotas(st, m)
    K1, K5 = max(1, round(.01 * m)), max(1, round(.05 * m))
    pairs = {"top1": (refsets.topk(mu[C], mu[T], K1), ms.top_k(mu[C], mu[T], K1)),
             "bottom5": (refsets.topk(mu[C], mu[T], K5, largest=False), ms.top_k(mu[C], mu[T], K5, largest=False)),
             "quota": (refsets.quota(mu[C], mu[T], st[C], st[T], Kg), ms.quota(mu[C], mu[T], st[C], st[T], Kg)),
             "budget": (refsets.knapsack_greedy(val[C], cost[C], val[T], cost[T], Bud), ms.budget(val[C], cost[C], val[T], cost[T], Bud))}
    h = float(np.median(np.linalg.norm(s[Dtr][:500] - s[Dtr].mean(0), axis=1)) / 4)
    pairs["mask"] = (refsets.kernel_mask(s[C], s[T], h, ALPHA), ms.publication_mask(s[C], s[T], h, ALPHA))
    ctx = dict(S_cal=S[C], w_cal=np.ones(len(C)), w_test=np.ones(m), g_cal=st[C], g_test=st[T], Phi_cal=Phi[C], Phi_test=Phi[T],
               s_cal=s[C], s_test=s[T], s_train=s[Dtr], w_train=np.ones(len(Dtr)), rng=np.random.default_rng(0))
    rows = []
    for rule, (old, new) in pairs.items():
        same_sel = bool(np.array_equal(old.selected, new.selected)); same_ref = bool(np.array_equal(old.member, new.dense()))
        q0, q1 = Mt.jomi(old).thresholds(ctx, ALPHA), ms.jomi(S[C], new, ALPHA)
        fin = np.isfinite(q0)
        row = dict(dataset=name, rule=rule, K=len(new), same_selected=same_sel, same_reference=same_ref,
                   jomi_max_diff=float(np.abs(q0[fin] - q1[fin]).max()) if fin.any() else 0.0, jomi_same_inf=bool(np.array_equal(fin, np.isfinite(q1))))
        if rule != "mask":                                             # SC-GeoCCP was not evaluated on masks
            idx = np.random.default_rng(1).permutation(len(new))[:nlp]; gaps, d1, d2 = [], [], []
            for r in idx:
                R = new.reference(r); PR, pj, SR = Phi[C][R], Phi[T][new.selected[r]], S[C][R]
                a, b = _primal_beta(SR, PR, pj, 1 - ALPHA, L1, pen), _dual_beta(SR, PR, pj, 1 - ALPHA, L1, pen)
                if a is None or b is None:
                    assert a is None and b is None
                    continue
                q_old = Mt.sc_geoccp(type(old)(old.selected[[r]], old.member[[r]], "one"), L1, pen).thresholds(ctx, ALPHA)[0]
                gaps.append(abs(objective(a, SR, PR, pj, 1 - ALPHA, L1, pen) - objective(b, SR, PR, pj, 1 - ALPHA, L1, pen)))
                d1.append(abs(pj @ a - q_old)); d2.append(abs(pj @ b - q_old))
            row.update(lp_units=len(gaps), objective_max_gap=max(gaps, default=0.0), primal_q_max_diff=max(d1, default=0.0), dual_q_max_diff=max(d2, default=0.0),
                       dual_q_median_diff=float(np.median(d2)) if d2 else 0.0)
        rows.append(row); print(row, flush=True)
    return rows


def main():
    t0 = time.time(); out = os.path.join(DS.FIA_RES, "package"); os.makedirs(out, exist_ok=True)
    T = pd.DataFrame([r for name in (DS.FIA, "poverty", "seattle", "camels") for r in one(name, 40)])
    T.to_csv(os.path.join(out, "parity.csv"), index=False, float_format="%.4g")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
