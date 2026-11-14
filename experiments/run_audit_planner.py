"""Audit planner, outcome test (paper/AUDIT_PLANNER_DESIGN.md; planner: excp_select/planner.py).

carbon  A project of K = 30 plots from a population of 6,000 non-training plots, on the 20 site-attribute maps of
        run_carbon.py and the 20 ESA CCI subsets of run_carbon_product.py (same seeds, so the same maps and subsets),
        100 audits each. Arms: n audited at random from the whole population, or from the stratum of its highest-
        predicted 5% or 10% (the stratum is fixed by the map before any audit); the rest of the population are the
        candidates and the rule credits their top 30. Estimates and VCS formulas are those of run_carbon.py.
floors  Fig. 6c and Fig. 8b's setting: the training fold, map and basis of run_scg_slack.py, its 200 redraws of the
        split (same seeds), with the first n of its 3,000 calibration units as the audit, n = 300, 1,000, 3,000, and the
        same quota rule on the same m = 3,000 candidates. JOMI and SC-GeoCCP (l1 = 0.95), coverage per basis column.

    ~/miniforge3/envs/favela/bin/python -u experiments/run_audit_planner.py carbon <maps> <audits> <workers>
    ~/miniforge3/envs/favela/bin/python -u experiments/run_audit_planner.py floors <redraws> <workers> <dataset> [ns...]
writes results/fia_db/planner/carbon_<site|cci>.csv.gz and floors_<dataset>.csv
(DHS: per-country and per-bump aggregates only, as for every other output of this project); parts in results/cache/planner_parts/
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
import run_carbon_product as RP                                       # noqa: E402
from run_main import fit_mu_sigma, quotas                             # noqa: E402
from excp_select import refsets, basis, methods as Mt, totals as TOT  # noqa: E402

OUT = os.path.join(DS.FIA_RES, "planner")
PARTS = os.path.join(DS.CACHE, "planner_parts")
K, POP = 30, 6000
# (share of the population in the stratum, audited plots): 92 and 186 are the planner's audits for Pr(finite) >= 0.9 in
# the 5% and 10% strata, 1,869 for the whole population; 150 and 300 audit half the stratum
ARMS = [(1.0, 92), (1.0, 186), (1.0, 1869), (1.0, 3000), (0.05, 92), (0.05, 150), (0.10, 186), (0.10, 300)]
ALPHA, TAU, L1 = 0.1, 0.9, 0.95


def population(kind, f):
    if kind == "site":
        d = DS.load(DS.FIA); y = d["y"]; N = d["N"]
        X = d["Xrf"][:, [c for c in range(d["Xrf"].shape[1]) if c not in RC.DROP["site"]]]
        perm = np.random.default_rng([5150, f]).permutation(N); ntr = int(0.4 * N)
        Dtr, pop = perm[:ntr], perm[ntr:]
        mu, sig = fit_mu_sigma(X, y, Dtr, np.random.default_rng([5151, f]))
        return y, mu, sig, np.sqrt(np.pi / 2) * sig, pop
    y, mu, s = RP.load("win")
    perm = np.random.default_rng([6150, f]).permutation(len(y))
    pop, out_ = perm[:RP.NSUB], perm[RP.NSUB:]
    return y, mu, np.maximum(s, 0.1 * s[out_].mean()), s, pop           # score scale floored as for the forest's sigma (SD can be 0)


def carbon_task(args):
    kind, f, ns = args
    part = os.path.join(PARTS, f"carbon_{kind}_f{f:02d}_s{ns}.csv.gz")
    if os.path.exists(part):
        return part
    t0 = time.time()
    y, mu, scale, layer, pop = population(kind, f)
    e = y - mu; S = np.abs(e) / scale
    assert len(pop) == POP
    order = pop[np.argsort(-mu[pop], kind="stable")]
    rows = []
    for k in range(ns):
        perms = {}
        for q, n in ARMS:
            if q not in perms:
                M = int(round(q * POP))
                perms[q] = np.random.default_rng([6201, 0 if kind == "site" else 1, f, k, M]).permutation(order[:M])
            C = perms[q][:n]
            T = np.setdiff1d(pop, C, assume_unique=False)
            sel = refsets.topk(mu[C], mu[T], K)
            J = T[sel.selected]; R = C[sel.member[0]]; nR = len(R); L = nR + K
            est = mu[J].sum()
            ma, se_ma = est + K * e[C].mean(), e[C].std(ddof=1) * np.sqrt(K + K ** 2 / n)
            sa, se_sa = (est + K * e[R].mean(), K * e[R].std(ddof=1) * np.sqrt(L / (nR * K))) if nR >= 2 else (0.0, np.inf)
            exact = est + TOT.lower_total(e[R], K, -mu[J].max(), RC.env(L, K, 1 / 3))
            r = int(np.ceil(TAU * (nR + 1)))
            q_j = np.sort(S[R])[r - 1] if nR >= 9 else np.inf
            q_s = np.sort(S[C])[int(np.ceil(TAU * (n + 1))) - 1]
            rows.append(dict(map=kind, fold=f, split=k, q=q, n=n, m=len(T), K=K, nR=nR, total=float(y[J].sum()), point=float(est),
                             se_layer=float(np.sqrt((layer[J] ** 2).sum())), layer_stratum=float(np.sqrt((layer[order[:max(int(q * POP), 1)]] ** 2).mean())),
                             ma=float(ma), se_ma=float(se_ma), sa=float(sa), se_sa=float(se_sa),
                             vcs_ma_rule=max(0.0, RC.vcs(ma, se_ma)), vcs_ma_always=max(0.0, ma - RC.Z_VCS * se_ma),
                             vcs_sa_always=max(0.0, sa - RC.Z_VCS * se_sa), vcs_sa_rule=max(0.0, RC.vcs(sa, se_sa)) if nR >= 2 else 0.0,
                             exact=max(0.0, float(exact)), finite=float(nR >= 9), cov_jomi=float((S[J] <= q_j).mean()),
                             hw_jomi=float((q_j * scale[J]).mean()) if nR >= 9 else np.nan,
                             cov_split=float((S[J] <= q_s).mean()), hw_split=float((q_s * scale[J]).mean())))
    os.makedirs(PARTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False, float_format="%.6g")
    print(f"  {kind} {f}: {time.time() - t0:.0f}s", flush=True)
    return part


def floors_setup(name):
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([2027, 0, DS.ALL.index(name)])          # == run_scg_slack.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, rest = perm[:ntr], perm[ntr:]
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st)
    aid = DS.POLICY[name] == "bottom" and name != "era5"
    return dict(y=y, st=st, rest=rest, nca=nca, mu=mu, sig=sig, B=B, Phi=Phi, aid=aid, S=np.abs(y - mu) / sig)


_SETUP = {}


def floors_task(args):
    name, k, ns = args
    if name not in _SETUP:
        _SETUP[name] = floors_setup(name)
    z = _SETUP[name]; mu, st, S, Phi, B = z["mu"], z["st"], z["S"], z["Phi"], z["B"]
    p = np.random.default_rng([6101, k]).permutation(z["rest"])
    C0, T = p[:z["nca"]], p[z["nca"]:]
    out = []
    for n in ns:
        C = C0[:n]
        sel = refsets.quota(mu[C], mu[T], st[C], st[T], quotas(st, len(T)), largest=not z["aid"])
        J = T[sel.selected]
        ctx = dict(S_cal=S[C], w_cal=np.ones(len(C)), w_test=np.ones(len(T)), Phi_cal=Phi[C], Phi_test=Phi[T],
                   g_cal=st[C], g_test=st[T], s_cal=np.zeros((len(C), 2)), s_test=np.zeros((len(T), 2)),
                   s_train=np.zeros((1, 2)), w_train=np.ones(1), rng=np.random.default_rng(0))
        qj = Mt.jomi(sel).thresholds(ctx, ALPHA); qs = Mt.sc_geoccp(sel, L1, B.penalize).thresholds(ctx, ALPHA)
        cj, cs = S[J] <= qj, S[J] <= qs
        f = Phi[J]
        out.append(dict(n=n, sf=f.sum(0), scj=(f * cj[:, None]).sum(0), scs=(f * cs[:, None]).sum(0),
                        inv=(1.0 / (sel.sizes + 1)).sum(), ns=len(J), ninf=int((~np.isfinite(qj)).sum()),
                        ninf_s=int((~np.isfinite(qs)).sum())))
    return out


def run_floors(M, W, name, ns):
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    if W > 1:
        with Pool(W) as pool:
            res = pool.map(floors_task, [(name, k, ns) for k in range(M)], chunksize=2)
    else:
        res = [floors_task((name, k, ns)) for k in range(M)]
    z = floors_setup(name); B = z["B"]; nb = len(B.h)
    kind = ["intercept"] + ["bump"] * nb + ["unit"] * len(B.units)
    level = [-1] + list(B.level) + [-1] * len(B.units)
    unit = [-1] * (1 + nb) + [int(u) for u in B.units]
    rows = []
    for i, n in enumerate(ns):
        sf = sum(r[i]["sf"] for r in res); scj = sum(r[i]["scj"] for r in res); scs = sum(r[i]["scs"] for r in res)
        tot = sum(r[i]["ns"] for r in res); inv = sum(r[i]["inv"] for r in res) / tot
        ninf = sum(r[i]["ninf"] for r in res); ninf_s = sum(r[i]["ninf_s"] for r in res)
        Ef = sf / tot
        with np.errstate(divide="ignore", invalid="ignore"):
            fl = 1 - ALPHA - L1 * inv / Ef
            rows.append(pd.DataFrame(dict(dataset=name, n=n, redraws=M, col=np.arange(len(sf)), kind=kind, level=level, unit=unit,
                                          Ef=Ef, M_u=sf / M, Einv=inv, K=tot / M, floor=fl, cov_jomi=scj / sf, cov_scg=scs / sf,
                                          inf_share=ninf / tot, inf_share_scg=ninf_s / tot)))
    pd.concat(rows).to_csv(os.path.join(OUT, f"floors_{name}.csv"), index=False, float_format="%.6g")
    print(f"floors {name} done, {time.time() - t0:.0f}s", flush=True)


def main():
    mode = sys.argv[1]
    if mode == "carbon":
        nf, ns, W = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        os.makedirs(OUT, exist_ok=True); t0 = time.time()
        for kind in ("site", "cci"):
            with Pool(W) as pool:
                parts = pool.map(carbon_task, [(kind, f, ns) for f in range(nf)], chunksize=1)
            pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(os.path.join(OUT, f"carbon_{kind}.csv.gz"), index=False, float_format="%.6g")
        print(f"carbon done, {time.time() - t0:.0f}s")
    else:
        M, W, name = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        run_floors(M, W, name, [int(a) for a in sys.argv[5:]] or [300, 1000, 3000])


if __name__ == "__main__":
    main()
