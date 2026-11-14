"""GeoCERT quick pilots (paper/innovation_GeoCERT.md).

Design: a few FIXED training folds (mu, sigma fitted once each), then many random
calibration/test splits of the remaining units -- exactly the design-based setting of the
guarantees (validity is over the split, given the training fold).

  l2  SC-GeoCCP: per selected unit j, Gibbs-Cherian-Candes augmented quantile regression on the
      JOMI reference set R_j (+ test point imputed at +inf), basis = intercept + coarse regions
      (unpenalized) [+ state/country indicators with L1 penalty]. Rule: state quota (FIA, top) /
      country quota (PovertyMap, bottom = aid to the predicted-poorest).
  l3  SC-Total: exact finite-sample lower bound for the TOTAL of Y over the top-K selected units
      (carbon certification). Under top-K, given the pool P = S u R (R = JOMI reference set),
      S is a uniform K-subset of P, so the bucket counts of S among the sorted calibration
      residuals have a value-free law; a simultaneous rank envelope (as in Jin's P12) gives
      sum_S e >= sum_b c_b r_(b) with prob >= 1 - delta. Since 2026-10-28 the envelope is the exact, integer-certified
      excp_select.totals.envelope_exact; results/pilot_l3_<ds>.csv of 22 October used the retired Monte Carlo
      envelope (kept for comparison in tests/test_pool_lemma.py), pilot_l3x_<ds>.csv the exact one.

    ~/miniforge3/envs/favela/bin/python -u experiments/pilot_geocert.py l2 fia 3 150 3
    ~/miniforge3/envs/favela/bin/python -u experiments/pilot_geocert.py l3 fia 3 300 3
args: part dataset n_folds n_splits workers. Writes results/pilot_<part>_<dataset>.csv
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                    # noqa: E402
from run_main import fit_mu_sigma, quotas                             # noqa: E402
from excp_select import refsets, methods as Mt                       # noqa: E402
from excp import engines as E, family as F                           # noqa: E402

ALPHA, DELTA = 0.1, 0.05
OUT = os.path.join(ROOT, "results")
L1S = [1.0, 1.5]          # must exceed tau = 1 - alpha, else the LP is unbounded (+inf) for states absent from R_j
N_REGIONS = {"fia": 3, "poverty": 5, "poverty_lab": 5}


def fold(name, f):
    d = DS.load(name); N = d["N"]
    rng = np.random.default_rng([4242, f, DS.ALL.index(name)])
    perm = rng.permutation(N); ntr = int(0.4 * N)
    mu, sig = fit_mu_sigma(d["Xrf"], d["y"], perm[:ntr], rng)
    return dict(Dtr=perm[:ntr], rest=perm[ntr:], mu=mu, sig=sig)


def regions(d, k):
    """Coarse regions = k-means of state/country centroids (label-free, fixed)."""
    from sklearn.cluster import KMeans
    st = d["state"]; u = np.unique(st)
    cent = np.array([d["coords"][st == a].mean(0) for a in u])
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(cent)
    return dict(zip(u, lab))


def basis(st, reg_of, states, n_reg, with_states):
    r = np.array([reg_of[a] for a in st])
    cols = [np.ones(len(st))] + [(r == k).astype(float) for k in range(1, n_reg)]
    pen = [False] * len(cols)
    if with_states:
        cols += [(st == a).astype(float) for a in states]; pen += [True] * len(states)
    return np.column_stack(cols), np.array(pen)


# ------------------------------------------------------------------ layer 2
def task_l2(args):
    name, f, fd, splits, reg_of = args[:5]
    d = DS.load(name); y, st = d["y"], d["state"]; mu, sig = fd["mu"], fd["sig"]
    S_all = np.abs(y - mu) / sig
    largest = DS.POLICY[name] != "bottom"
    states = np.unique(st); n_reg = len(set(reg_of.values()))
    rows = []
    for s in splits:
        rng = np.random.default_rng([99, f, s])
        p = rng.permutation(fd["rest"]); C, T = p[: len(p) // 2], p[len(p) // 2:]
        Kg = quotas(st, len(T))
        sel = refsets.quota(mu[C], mu[T], st[C], st[T], Kg, largest=largest)
        J = T[sel.selected]
        ctx = dict(S_cal=S_all[C], w_cal=np.ones(len(C)), w_test=np.ones(len(T)), g_cal=st[C], g_test=st[T])
        qs = {"SCP stud": np.full(len(J), F.split_cp().thresholds(ctx, ALPHA)[0]),
              "JOMI": Mt.jomi(sel).thresholds(ctx, ALPHA),
              "JOMI|state": Mt.jomi(sel, by_group=True).thresholds(ctx, ALPHA)}
        for tag, ws, l1 in [("SC-GeoCCP coarse", False, 0.0)] + [(f"SC-GeoCCP l1={l}", True, l) for l in L1S]:
            Pc, pen = basis(st[C], reg_of, states, n_reg, ws); Pt, _ = basis(st[J], reg_of, states, n_reg, ws)
            q = np.empty(len(J))
            for r, j in enumerate(J):
                m = sel.member[r]
                q[r] = E.fclass(S_all[C][m], np.ones(m.sum()), Pc[m], np.ones(1), Pt[r:r + 1], ALPHA,
                                l1=l1, penalize=pen)[0]
            qs[tag] = q
        for mth, q in qs.items():
            hw = q * sig[J]
            for r, j in enumerate(J):
                rows.append((name, f, s, mth, int(st[j]), bool(S_all[j] <= q[r]), bool(np.isfinite(q[r])),
                             float(hw[r]), int(sel.member[r].sum())))
    return rows


# ------------------------------------------------------------------ layer 3
def envelope(L, K, delta, rng=None, n_mc=None):
    """Simultaneous upper band C_b (b = 0..n_R) on N_b = #S-units below the (b+1)-th smallest
    calibration value, for S a uniform K-subset of L slots, with P(any N_b > C_b) <= delta exactly.
    Delegates to excp_select.totals.envelope_exact (2026-10-28). The Monte Carlo version used here before fitted its
    shape and lambda on the same 10,000 draws, so its true violation could exceed delta (0.0508 at L = 120, K = 30;
    tests/test_pool_lemma.py prints it as the retired MC envelope); rng and n_mc are kept only for call compatibility."""
    from excp_select import totals as TOT
    return TOT.envelope_exact(L, K, delta, iters=16, fast=True)


def lower_total(r_obs, K, floor, C, weights=None):
    """sum_S v >= sum_b c_b r_(b), c_b = C_b - C_{b-1}, r_(0) = floor (values in bucket 0 only
    known to exceed floor). With weights (sigma_j), pair largest weight with smallest bucket
    value (rearrangement) for sum_S sigma_j v_j."""
    r_obs = np.sort(r_obs)
    if len(r_obs):
        floor = min(floor, r_obs[0])       # a bucket-0 value is >= floor; keep r nondecreasing
    r = np.r_[floor, r_obs]
    c = np.diff(np.r_[0, C]).astype(int)
    vals = np.repeat(r, c)[:K]
    if weights is None:
        return vals.sum()
    return (np.sort(vals) * np.sort(weights)[::-1]).sum()


from functools import lru_cache
from excp_select import totals as TOT                                # noqa: E402


@lru_cache(maxsize=None)
def _env_exact(L, K):
    return tuple(int(c) for c in TOT.envelope_exact(L, K, DELTA))


def task_l3(args):
    name, f, fd, splits, _ = args[:5]
    exact = len(args) > 5 and args[5]          # part "l3x": exact DP envelope (theory/POOL_LEMMA.md T1(b))
    d = DS.load(name); y = d["y"]; mu, sig = fd["mu"], fd["sig"]
    e = y - mu; z = 1.6448536269514722
    # support floor for Y (needed for any distribution-free bound on a sum): biomass >= 0;
    # otherwise the training-fold minimum (an explicit support assumption)
    ymin = 0.0 if name == "fia" else float(y[fd["Dtr"]].min())
    largest = DS.POLICY[name] != "bottom"     # FIA: top-K carbon; PovertyMap: bottom-K aid
    rows = []
    for s in splits:
        rng = np.random.default_rng([77, f, s])
        p = rng.permutation(fd["rest"]); C, T = p[: len(p) // 2], p[len(p) // 2:]
        for frac in (0.01, 0.05):
            K = max(1, int(round(frac * len(T))))
            sel = refsets.topk(mu[C], mu[T], K, largest=largest)
            J = T[sel.selected]; Rm = sel.member[0]; R = C[Rm]; nR = len(R); L = nR + K
            tot, smu = y[J].sum(), mu[J].sum()
            Cb = np.array(_env_exact(L, K)) if exact else envelope(L, K, DELTA, rng)
            lt = TOT.lower_total if exact else lower_total
            b = {}
            b["SC-Total exact (raw)"] = smu + lt(e[R], K, ymin - mu[J].max(), Cb)
            v = e / sig
            b["SC-Total exact (stud)"] = smu + lt(v[R], K, ((ymin - mu[J]) / sig[J]).min(), Cb, sig[J])
            sR = e[R].std(ddof=1) if nR > 1 else np.inf
            b["Ref-set SRSWOR CLT"] = smu + K * e[R].mean() - z * K * sR * np.sqrt(L / (max(nR, 1) * K))
            b["Cal iid CLT (no selection)"] = smu + K * e[C].mean() - z * e[C].std(ddof=1) * np.sqrt(K)
            b["Model iid CLT"] = smu - z * np.sqrt(((np.sqrt(np.pi / 2) * sig[J]) ** 2).sum())
            b["Point sum mu"] = smu
            qs = F.split_cp().thresholds(dict(S_cal=np.abs(e[C]) / sig[C], w_cal=np.ones(len(C)),
                                              w_test=np.ones(1)), ALPHA)[0]
            b["Sum unit lower (SCP stud)"] = (mu[J] - qs * sig[J]).sum()
            qj = Mt.jomi(sel).thresholds(dict(S_cal=np.abs(e[C]) / sig[C], w_cal=np.ones(len(C)),
                                              w_test=np.ones(len(T))), ALPHA)
            b["Sum unit lower (JOMI stud)"] = (mu[J] - qj * sig[J]).sum()
            for mth, lb in b.items():
                rows.append((name, f, s, frac, K, nR, mth, float(lb), float(tot), bool(lb <= tot)))
    return rows


def main():
    part, name = sys.argv[1], sys.argv[2]
    nf, ns, W = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    t0 = time.time()
    d = DS.load(name)
    reg_of = regions(d, N_REGIONS.get(name, 3))
    folds = [fold(name, f) for f in range(nf)]
    print(f"fitted {nf} folds in {time.time()-t0:.0f}s", flush=True)
    chunk = max(1, ns // (2 * W))
    jobs = [(name, f, folds[f], list(range(a, min(ns, a + chunk))), reg_of, part == "l3x")
            for f in range(nf) for a in range(0, ns, chunk)]
    fn = task_l2 if part == "l2" else task_l3
    with Pool(W) as pool:
        res = pool.map(fn, jobs, chunksize=1)
    rows = [r for x in res for r in x]
    cols = (["dataset", "fold", "split", "method", "state", "cov", "finite", "hw", "R"] if part == "l2" else
            ["dataset", "fold", "split", "frac", "K", "nR", "method", "lb", "total", "covered"])
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(os.path.join(OUT, f"pilot_{part}_{name}.csv"), index=False)   # l3x -> pilot_l3x_<ds>.csv
    print(f"done {part} {name}: {len(df)} rows, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
