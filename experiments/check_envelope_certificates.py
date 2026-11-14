"""Did the integer certificate of the rank envelope ever fail on the rebuilt FIA table? (SI note on Theorem 2(a), 2026-11-10)

excp_select.totals.envelope_exact bisects lambda with the floating-point DP and then certifies the returned envelope in
integer arithmetic; if the certificate fails it falls back to the trivial envelope C = K. The carbon, product and planner
runs on the rebuilt table (results/fia_db/) call it through run_carbon.env(L, K, delta) = envelope_exact(L, K, delta,
iters=16, fast=True), with delta = 1/3 (VCS) and 0.30 (ART). This script collects every pool size (L = nR + K, K) those
runs recorded, repeats the same bisection, and records whether the certificate of the bisected envelope fails.
    ~/miniforge3/envs/favela/bin/python -I -B experiments/check_envelope_certificates.py 4
writes results/fia_db/envelope_certificates.csv and prints the number of failures (0 means the fall-back never occurred)
"""
import glob
import os
import sys
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import excp_select.totals as TOT                                       # noqa: E402

DELTAS = (1 / 3, 0.30)                                                  # run_carbon.DELTA


def pairs():
    out = set()
    for f in sorted(glob.glob(os.path.join(PROJ, "results", "fia_db", "**", "*.csv*"), recursive=True)):
        try:
            cols = pd.read_csv(f, nrows=0).columns
        except Exception:                                               # noqa: BLE001
            continue
        if {"K", "nR"} <= set(cols):
            d = pd.read_csv(f, usecols=["K", "nR"])
            out |= set(zip((d.nR + d.K).astype(int), d.K.astype(int)))
    return sorted(out)


def one(args):
    ell, K, delta = args
    nR = ell - K
    if nR == 0:
        return dict(L=ell, K=K, delta=delta, certificate_failed=False, trivial_pool=True)
    mu, sd = TOT._moments(ell, K)

    def C_of(lam):
        C = np.maximum.accumulate(np.minimum(K, np.floor(mu + lam * sd))).astype(int)
        return list(C) + [K]
    lo, hi = 0.0, 12.0
    for _ in range(16):                                                 # iters=16, fast=True, as run_carbon.env
        lam = (lo + hi) / 2
        if 1 - TOT.valid_prob_fast(ell, K, C_of(lam)) > delta:
            lo = lam
        else:
            hi = lam
    C = C_of(hi)
    failed = TOT.violation_exact(ell, K, C) > Fraction(repr(float(delta)))
    return dict(L=ell, K=K, delta=delta, certificate_failed=bool(failed), trivial_pool=False)


def main():
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [(ell, K, d) for ell, K in pairs() for d in DELTAS]
    with Pool(W) as pool:
        df = pd.DataFrame(pool.map(one, jobs, chunksize=8))
    df.to_csv(os.path.join(PROJ, "results", "fia_db", "envelope_certificates.csv"), index=False)
    print(f"{len(df)} envelopes ({df[['L', 'K']].drop_duplicates().shape[0]} pool sizes x {len(DELTAS)} risks); "
          f"L up to {df.L.max()}; certificate failures: {int(df.certificate_failed.sum())}")


if __name__ == "__main__":
    main()
