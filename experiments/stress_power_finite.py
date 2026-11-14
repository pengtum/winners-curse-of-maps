"""Finite-sample Gaussian replica of the stress-test decision rules (paper/STRESS_TEST_DESIGN.md), written by the
pre-run design review on 2026-10-30 and copied here unchanged (its own note follows). It reads no project data: N
and the held-out R2 of each setting are typed in from results/maps_r2_rf_summary.csv.
    ~/miniforge3/envs/favela/bin/python experiments/stress_power_finite.py 300 4 results/stress_power_finite_300.json
(300 simulated studies of 100 splits each); summary: experiments/stress_power_finite_summary.py.

Original note:
Finite-sample Gaussian replica of the STRESS_TEST_DESIGN decision rules (no project data).

Per setting: N and held-out R2 copied from the reported results/maps_r2_rf_summary.csv (typed in below).
Population: mu ~ N(0, R2), r ~ N(0, 1-R2) (homoscedastic; mu = the true conditional mean, so studentized = additive).
Each "study" = 100 random 40/30/30 splits of one fixed population, pooled coverage (+inf counted as covered),
exactly the estimand of the design. M studies per setting give the sampling distribution of the pooled statistic.
Arm B rules (5%): S_tau = (tau^2 mu + y + tau s eps)/(tau^2+1), tau in 0.5,1,2; control (2mu + s eps)/2; policy mu.
Arm A (FIA N=10000): mu_s, mu_h, mu_f nested (variances .39,.28,.25,.08); intervals from mu_s; top 5% and 1%.
Methods: naive Mondrian (10/20 bins of the interval map's mu, edges = training-fold quantiles),
informed Mondrian (10/20 bins of the rule score, training-fold edges), JOMI (reference set {i in C: S_i > theta}).
"""
import sys, json
import numpy as np
from multiprocessing import Pool

ALPHA = 0.1
SETTINGS = {  # name: (N, R2 test mean) from results/maps_r2_rf_summary.csv
    "camels": (670, 0.9144), "seattle": (3000, 0.8457), "voting": (3108, 0.8049),
    "health_stroke": (2350, 0.8938), "health_diabetes": (2350, 0.9288), "health_bphigh": (2350, 0.9106),
    "health_obesity": (2350, 0.7495), "fia": (10000, 0.9152), "era5": (10000, 0.9512),
    "poverty_lab": (10000, 0.7768), "poverty": (10000, 0.7633)}
NAMES = list(SETTINGS)


def qconf(a):
    n = len(a); k = int(np.ceil((1 - ALPHA) * (n + 1)))
    return np.inf if k > n else np.partition(a, k - 1)[k - 1]


def mond(a, key, D, C, G):
    edges = np.quantile(key[D], np.linspace(0, 1, G + 1)[1:-1])
    bC = np.searchsorted(edges, key[C], side="right")
    q = np.array([qconf(a[C][bC == b]) for b in range(G)])
    return edges, q


def cov_bins(a, key, J, edges, q):
    qb = q[np.searchsorted(edges, key[J], side="right")]
    return int((a[J] <= qb).sum()), int((~np.isfinite(qb)).sum())


def study_B(args):
    name, k = args
    N, R2 = SETTINGS[name]
    rng = np.random.default_rng([99, NAMES.index(name), k])
    s = np.sqrt(1 - R2)
    mu = rng.normal(0, np.sqrt(R2), N) + 1e-9 * rng.standard_normal(N); r = rng.normal(0, s, N); y = mu + r
    a = np.abs(r)
    ntr, nca = int(0.4 * N), int(0.3 * N)
    rules = ["tau0.5", "tau1", "tau2", "control", "policy"]
    meths = ["naive10", "naive20", "inf10", "inf20", "jomi"]
    tot = {(ru, me): np.zeros(3) for ru in rules for me in meths}   # n_sel, n_cov, n_inf
    cor = {ru: [] for ru in rules[:3]}
    for rep in range(100):
        perm = rng.permutation(N); D, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
        m = len(T); K = max(1, int(round(0.05 * m)))
        eps = rng.standard_normal(N)
        nb = {G: mond(a, mu, D, C, G) for G in (10, 20)}
        scores = {}
        for tau, ru in zip((0.5, 1.0, 2.0), rules[:3]):
            Z = y + tau * s * eps; scores[ru] = (tau ** 2 * mu + Z) / (tau ** 2 + 1)
            cor[ru].append(np.corrcoef(Z[T] - mu[T], r[T])[0, 1])
        scores["control"] = (mu + mu + s * eps) / 2
        scores["policy"] = mu
        for ru, S in scores.items():
            sT = S[T]; o = np.argsort(-sT, kind="stable")[:K + 1]
            J = T[o[:K]]; theta = sT[o[K]]
            res = {}
            for G in (10, 20):
                e, q = nb[G]; res[f"naive{G}"] = cov_bins(a, mu, J, e, q)
                e2, q2 = mond(a, S, D, C, G); res[f"inf{G}"] = cov_bins(a, S, J, e2, q2)
            R = C[S[C] > theta]; qj = qconf(a[R])
            res["jomi"] = (int((a[J] <= qj).sum()), K if not np.isfinite(qj) else 0)
            for me, (nc, ni) in res.items():
                tot[(ru, me)] += (K, nc, ni)
    out = {f"{ru}|{me}": [float(v[1] / v[0]), float(v[2] / v[0])] for (ru, me), v in tot.items()}
    out["cor"] = {ru: float(np.mean(v)) for ru, v in cor.items()}
    return name, k, out


def study_A(k):
    N = 10000
    rng = np.random.default_rng([98, k])
    u = rng.normal(0, 1, (4, N))
    mus = np.sqrt(0.39) * u[0]; muh = mus + np.sqrt(0.28) * u[1]; muf = muh + np.sqrt(0.25) * u[2]
    y = muf + np.sqrt(0.08) * u[3]
    jit = 1e-9 * rng.standard_normal((3, N)); mus = mus + jit[0]; muh = muh + jit[1]; muf = muf + jit[2]
    a = np.abs(y - mus)
    ntr, nca = int(0.4 * N), int(0.3 * N)
    sels = {"site": mus, "siteheight": muh, "full": muf}
    tot = {}
    for rep in range(100):
        perm = rng.permutation(N); D, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
        m = len(T)
        nb = {G: mond(a, mus, D, C, G) for G in (10, 20)}
        for sn, S in sels.items():
            inf_b = {G: mond(a, S, D, C, G) for G in (10, 20)}
            for f in (0.05, 0.01):
                K = max(1, int(round(f * m)))
                sT = S[T]; o = np.argsort(-sT, kind="stable")[:K + 1]
                J = T[o[:K]]; theta = sT[o[K]]
                res = {}
                for G in (10, 20):
                    e, q = nb[G]; res[f"naive{G}"] = cov_bins(a, mus, J, e, q)
                    e2, q2 = inf_b[G]; res[f"inf{G}"] = cov_bins(a, S, J, e2, q2)
                R = C[S[C] > theta]; qj = qconf(a[R])
                res["jomi"] = (int((a[J] <= qj).sum()), K if not np.isfinite(qj) else 0)
                for me, (nc, ni) in res.items():
                    key = f"{sn}|{f}|{me}"
                    tot.setdefault(key, np.zeros(3)); tot[key] += (K, nc, ni)
    return k, {kk: [float(v[1] / v[0]), float(v[2] / v[0])] for kk, v in tot.items()}


if __name__ == "__main__":
    M = int(sys.argv[1]); W = int(sys.argv[2]); outp = sys.argv[3]
    with Pool(W) as p:
        resA = p.map(study_A, range(M), chunksize=1)
        resB = p.map(study_B, [(nm, k) for nm in NAMES for k in range(M)], chunksize=1)
    json.dump({"A": resA, "B": resB}, open(outp, "w"))
    print("done")
