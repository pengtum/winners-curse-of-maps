"""Where can SC-GeoCCP's guarantee be afforded? (Figs 8b, 6c; theory/SC_GeoCCP.md Theorem 1, corollary 2-3)

For every penalized basis column f (Gaussian bump or state indicator) Theorem 1 gives
    f-weighted coverage | selected  >=  1 - alpha - l1 E[1/(|R_j|+1) | selected] / E[f(X_j) | selected],
so the guarantee at a place is set by how much selected mass the rule puts there. This script fixes the
training fold, mu, sigma and basis of run_scgeoccp.py repetition 0 (same seeds), redraws the random
calibration / test split <splits> times, applies the dataset's quota rule (state / country quota; for the aid
data the poorest villages of each country, quota_bottom), and estimates both expectations.
    ~/miniforge3/envs/favela/bin/python -u experiments/run_scg_slack.py [splits] [dataset]   (default 200 fia)
writes results/scg_slack_<dataset>.csv: one row per basis column (centre, bandwidth, unit, E[f | selected],
slack, guaranteed floor = 1 - alpha - slack; floor <= 0 means the guarantee is vacuous there)
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma, quotas                             # noqa: E402
from excp_select import refsets, basis                                # noqa: E402

ALPHA, L1, REP = 0.1, 0.95, 0


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    NAME = sys.argv[2] if len(sys.argv) > 2 else "fia"
    aid = DS.POLICY[NAME] == "bottom" and NAME != "era5"                # == run_scgeoccp.py
    rule = "quota_bottom" if aid else "quota"
    t0 = time.time()
    d = DS.load(NAME); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([2027, REP, DS.ALL.index(NAME)])      # == run_scgeoccp.py / run_main.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, rest = perm[:ntr], perm[ntr:]
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st)
    sf, si, ns = np.zeros(Phi.shape[1]), 0.0, 0
    for k in range(M):
        p = np.random.default_rng([6101, k]).permutation(rest)
        C, T = p[:nca], p[nca:]
        sel = refsets.quota(mu[C], mu[T], st[C], st[T], quotas(st, len(T)), largest=not aid)
        J = T[sel.selected]
        sf += Phi[J].sum(0); si += (1.0 / (sel.sizes + 1)).sum(); ns += len(J)
    Ef, Einv = sf / ns, si / ns
    with np.errstate(divide="ignore"):
        slack = L1 * Einv / Ef                                          # inf: a unit the rule never selects
    nb = len(B.h)
    kind = ["intercept"] + ["bump"] * nb + ["unit"] * len(B.units)
    level = [-1] + list(B.level) + [-1] * len(B.units)
    cx = [np.nan] + list(B.centres[:, 0]) + [np.nan] * len(B.units)
    cy = [np.nan] + list(B.centres[:, 1]) + [np.nan] * len(B.units)
    h = [np.nan] + list(B.h) + [np.nan] * len(B.units)
    unit = [-1] * (1 + nb) + [int(u) for u in B.units]
    out = pd.DataFrame(dict(rule=rule, col=np.arange(Phi.shape[1]), kind=kind, level=level, cx=cx, cy=cy, h=h, unit=unit,
                            Ef=Ef, slack=slack, floor=1 - ALPHA - slack))
    out.loc[0, ["slack", "floor"]] = [0.0, 1 - ALPHA]                   # intercept is unpenalized: exact
    out["nbar"], out["l1"], out["splits"], out["selected_per_split"] = 1 / Einv, L1, M, ns / M
    f = os.path.join(ROOT, "results", f"scg_slack_{NAME}.csv")
    out.to_csv(f, index=False, float_format="%.6g")
    print(f"n_bar = {1 / Einv:.1f}, selected per split = {ns / M:.1f}, {time.time() - t0:.0f}s -> {f}")
    print(out.groupby(["kind", "level"]).floor.describe().round(3).to_string())


if __name__ == "__main__":
    main()
