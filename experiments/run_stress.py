"""Stress test of paper/STRESS_TEST_DESIGN.md (written and frozen before the run): selection rules that use information
beyond the map that supplies the intervals.

Arm A ('twomap', FIA; fia_post2000 as a sensitivity run): three random-forest maps on the same training fold, with the
feature tiers of run_carbon.py (DROP): site (site attributes only), site+height (+ mean tree height, a stand-in for
lidar canopy height) and full (the main FIA map, identical to run_main.py). The site and site+height maps are fitted as
experiments/heldout_r2.py fits them (a fresh default_rng([2027, rep, dataset]) per map: permutation, then fit_mu_sigma),
so they are the maps of results/maps_r2_rf_per_rep.csv. Intervals always come from the site map; the rule selects the
highest 5% or 1% of the test units by the site, the site+height or the full map's prediction.
Arm B ('signal', 11 settings): a simulated field signal Z_tau = y + tau sqrt(pi/2) sigma eps (sigma models the mean
absolute residual, so tau is nominally the ratio of the field error to the map error); the rule selects 5% in the policy
direction d by S_tau = d (tau^2 mu + Z_tau) / (tau^2 + 1), tau = 0.5, 1, 2; control S_0 = d (mu + Z_0) / 2 with
Z_0 = mu + sqrt(pi/2) sigma eps (noise only); policy d mu (reproduction check).
JOMI's reference set for top-K of a score is {i in C : score_i > theta}, theta the (K+1)-th largest test score
(refsets.topk). Methods: split CP (|y - mu|), normalized CP (|y - mu| / sigma), Mondrian CP with ten and twenty bins of
the interval map's prediction ('naive'), Mondrian CP with ten and twenty bins of the rule's own score ('informed'), JOMI
with the normalized and the additive score. Bin edges use the training fold only. Coverage counts +inf as covering.
Recorded per (rule, method): selected, covered, infinite, misses above (y - mu > half-width) and below, finite half-width,
mean residual of the selected units (policy direction; also in units of sqrt(pi/2) sigma), overlap with the interval
map's own selection; for bins, the calibration units in the extreme bin (policy direction) and the fewest in any bin that
holds a selected unit; for JOMI, the mean and smallest reference-set size; for arm B, corr(Z - mu, y - mu) and
sd(Z - y) / sd(y - mu) on the test units; for arm A, the maps' held-out R2.
Seeds: folds, the main map and its sigma as run_main.py ([2027, rep, dataset]); field noise [2035, rep, dataset], one eps
shared by all tau. Reproduction checks (stop on failure):
  arm A  full map under its own 5% rule with its own scores: JOMI stud and SCP stud (and naive Mondrian, if
         results/md_main_<ds>.csv exists) equal results/main_<ds>.csv; reps 0-9 of fia: held-out R2 of the site and
         site+height maps equal results/maps_r2_rf_per_rep.csv; under the site map's own rule the informed bins equal
         the naive bins unit by unit
  arm B  policy rule: JOMI stud, SCP stud and naive Mondrian equal results/main_<ds>.csv and md_main_<ds>.csv; informed
         bins equal naive bins unit by unit
    ~/miniforge3/envs/favela/bin/python -u experiments/run_stress.py twomap <R> <workers> [fia|fia_post2000]
    ~/miniforge3/envs/favela/bin/python -u experiments/run_stress.py signal <R> <workers> [datasets...]
writes results/st_twomap_<ds>.csv and results/st_signal_<ds>.csv (or to $STRESS_OUT). With R < 10 (code check) the
script requires $STRESS_OUT and prints only the checks, row counts and times.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from excp_select import refsets, methods as Mt                        # noqa: E402
from run_main import fit_mu_sigma                                     # noqa: E402
from run_baselines import q_conf, ALPHA                               # noqa: E402
from run_mondrian import mondrian, NAMES                              # noqa: E402
from run_carbon import DROP                                           # noqa: E402

OUT = os.path.join(ROOT, "results")                                    # stored results (reproduction checks)
OUT_W = os.environ.get("STRESS_OUT", OUT)                             # where outputs go (scratch for the code check)
TAUS = (0.5, 1.0, 2.0)
CSD = np.sqrt(np.pi / 2)
_REF, _R2 = {}, None


def ref(name):
    """Stored per-split counts for the reproduction checks (md_main only where it exists)."""
    if name not in _REF:
        parts = [pd.read_csv(os.path.join(OUT, f"main_{name}.csv"), usecols=["rep", "rule", "method", "n_sel", "n_cov"])]
        f = os.path.join(OUT, f"md_main_{name}.csv")
        if os.path.exists(f):
            parts.append(pd.read_csv(f, usecols=["rep", "rule", "method", "n_sel", "n_cov"]))
        _REF[name] = pd.concat(parts).set_index(["rep", "rule", "method"])
    return _REF[name]


def r2_stored():
    global _R2
    if _R2 is None:
        _R2 = pd.read_csv(os.path.join(OUT, "maps_r2_rf_per_rep.csv")).set_index(["setting", "rep"]).r2_test
    return _R2


def r2(yt, pt):
    return float(1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())


def split(name, rep):
    d = DS.load(name); N = d["N"]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # == run_main.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    return d, rng, perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]


def ctx_of(S, s, st, Dtr, C, T, rng):
    return dict(S_cal=S[C], w_cal=np.ones(len(C)), w_test=np.ones(len(T)), s_cal=s[C], s_test=s[T], s_train=s[Dtr],
                w_train=np.ones(len(Dtr)), g_cal=st[C], g_test=st[T], rng=rng)


def bins_info(key, Dtr, C, J, G, counts, extreme_top):
    edges = np.quantile(key[Dtr], np.linspace(0, 1, G + 1)[1:-1])
    bJ = np.searchsorted(edges, key[J], side="right")
    return dict(extreme_bin_cal=int(counts[-1] if extreme_top else counts[0]), min_bin_cal_sel=int(counts[bJ].min()),
                extreme_bin_share=float((counts[-1] if extreme_top else counts[0]) / len(C)))


def evaluate(y, mu, sig, Dtr, C, T, scores, K, ctx, ctx_a, dg, own, extra):
    """Intervals from (mu, sig) at the top K of each score; one row per (rule, method)."""
    m = len(T); Ra, Ss = np.abs(y - mu), np.abs(y - mu) / sig
    naive = {G: mondrian(Ss, mu, Dtr, C, T, G) for G in (10, 20)}
    hw_all = {"SCP |res|": q_conf(Ra[C]) * np.ones(m), "SCP stud": q_conf(Ss[C]) * sig[T],
              "Mondrian CP": naive[10][0] * sig[T], "Mondrian CP, 20 bins": naive[20][0] * sig[T]}
    J0 = set(T[refsets.topk(own[C], own[T], K).selected])
    rows, sels, hws = [], {}, {}
    for rn, sc in scores.items():
        sel = refsets.topk(sc[C], sc[T], K); k = sel.selected; J = T[k]; sels[rn] = sel
        hw = {mn: h[k] for mn, h in hw_all.items()}
        info = {"Mondrian CP": bins_info(mu, Dtr, C, J, 10, naive[10][1], dg > 0),
                "Mondrian CP, 20 bins": bins_info(mu, Dtr, C, J, 20, naive[20][1], dg > 0)}
        for G in (10, 20):
            qs, cnt = mondrian(Ss, sc, Dtr, C, T, G)
            hw[f"Mondrian CP, rule bins {G}"] = qs[k] * sig[J]
            info[f"Mondrian CP, rule bins {G}"] = bins_info(sc, Dtr, C, J, G, cnt, True)
        hw["JOMI stud"] = Mt.jomi(sel).thresholds(ctx, ALPHA) * sig[J]
        hw["JOMI |res|"] = Mt.jomi(sel).thresholds(ctx_a, ALPHA)
        sz = sel.sizes
        info["JOMI stud"] = info["JOMI |res|"] = dict(R_mean=float(sz.mean()), R_min=int(sz.min()))
        res = y[J] - mu[J]; ov = len(J0.intersection(J)) / len(J); hws[rn] = hw
        for mn, hh in hw.items():
            cov = np.abs(res) <= hh; fin = np.isfinite(hh)
            rows.append(dict(rule=rn, method=mn, n_sel=len(J), n_cov=int(cov.sum()), n_inf=int((~fin).sum()),
                             up_miss=int((res > hh).sum()), lo_miss=int((res < -hh).sum()),
                             hw_mean_fin=float(hh[fin].mean()) if fin.any() else np.nan, hw_med=float(np.median(hh)),
                             overlap_own=ov, resid_mean=float(np.mean(dg * res)),
                             resid_mean_sd=float(np.mean(dg * res / (CSD * sig[J]))),
                             sig_ratio=float(sig[J].mean() / sig[T].mean()), **info.get(mn, {}), **extra.get(rn, {})))
    return rows, sels, hws


def one_twomap(args):
    name, rep = args
    d, rng, Dtr, C, T = split(name, rep); X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    N, m = len(y), len(T); t0 = time.time()
    mu_f, sig_f = fit_mu_sigma(X, y, Dtr, rng)                          # the main FIA map, identical to run_main.py
    maps = {}
    for k in ("site", "site+height"):                                    # as experiments/heldout_r2.py
        rk = np.random.default_rng([2027, rep, DS.ALL.index(name)]); rk.permutation(N)
        maps[k] = fit_mu_sigma(np.delete(X, DROP[k], axis=1), y, Dtr, rk)
    (mu_s, sig_s), (mu_h, _) = maps["site"], maps["site+height"]
    R2 = {"site": r2(y[T], mu_s[T]), "site+height": r2(y[T], mu_h[T]), "full": r2(y[T], mu_f[T])}
    if name == "fia" and rep < 10:                                       # reproduction check: the stored maps
        for k in ("site", "site+height"):
            assert abs(R2[k] - r2_stored().loc[(f"fia[{k}]", rep)]) < 1e-9, (k, rep, R2[k])
    Ra, Ss = np.abs(y - mu_s), np.abs(y - mu_s) / sig_s
    ctx = ctx_of(Ss, s, st, Dtr, C, T, rng); ctx_a = dict(ctx, S_cal=Ra[C])
    scores = {"site": mu_s, "site+height": mu_h, "full": mu_f}                  # FIA's policy direction is top
    extra = {k: dict(r2_interval_map=R2["site"], r2_selector=R2[k]) for k in scores}
    out = []
    for frac in (0.05, 0.01):
        K = max(1, int(round(frac * m)))
        rows, sels, hws = evaluate(y, mu_s, sig_s, Dtr, C, T, scores, K, ctx, ctx_a, 1.0, mu_s, extra)
        for G in (10, 20):                                                     # own rule: informed bins == naive bins
            a, b = hws["site"][f"Mondrian CP, rule bins {G}"], hws["site"]["Mondrian CP" + ("" if G == 10 else ", 20 bins")]
            assert np.array_equal(a, b), (rep, frac, G)
        if frac == 0.05:                                                       # reproduction check: the full map
            Ssf, Raf = np.abs(y - mu_f) / sig_f, np.abs(y - mu_f)
            sel = sels["full"]; J = T[sel.selected]; Rf = ref(name)
            chk = {"JOMI stud": Mt.jomi(sel).thresholds(ctx_of(Ssf, s, st, Dtr, C, T, rng), ALPHA) * sig_f[J],
                   "SCP stud": q_conf(Ssf[C]) * sig_f[J]}
            if (rep, "top5", "Mondrian CP") in Rf.index:
                chk["Mondrian CP"] = mondrian(Ssf, mu_f, Dtr, C, T, 10)[0][sel.selected] * sig_f[J]
            for mn, hh in chk.items():
                assert (len(J), int((Raf[J] <= hh).sum())) == tuple(Rf.loc[(rep, "top5", mn), ["n_sel", "n_cov"]]), (name, rep, mn)
        out += [dict(arm="twomap", dataset=name, rep=rep, frac=frac, **r) for r in rows]
    return out, time.time() - t0


def one_signal(args):
    name, rep = args
    d, rng, Dtr, C, T = split(name, rep); X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    m = len(T); t0 = time.time()
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    eps = np.random.default_rng([2035, rep, DS.ALL.index(name)]).standard_normal(len(y))
    dg = 1.0 if DS.POLICY[name] == "top" else -1.0
    K = max(1, int(round(0.05 * m)))
    scores = {"policy": dg * mu}
    extra = {"policy": dict(tau=np.nan, cor_signal_resid=np.nan, err_ratio=np.nan)}
    sdr = np.std(y[T] - mu[T])
    for tau in TAUS:
        Zt = y + tau * CSD * sig * eps
        rn = f"signal tau={tau:g}"
        scores[rn] = dg * (tau ** 2 * mu + Zt) / (tau ** 2 + 1)
        extra[rn] = dict(tau=tau, cor_signal_resid=float(np.corrcoef(Zt[T] - mu[T], y[T] - mu[T])[0, 1]),
                         err_ratio=float(np.std(Zt[T] - y[T]) / sdr))
    Z0 = mu + CSD * sig * eps
    scores["control"] = dg * (mu + Z0) / 2
    extra["control"] = dict(tau=np.nan, cor_signal_resid=float(np.corrcoef(Z0[T] - mu[T], y[T] - mu[T])[0, 1]),
                            err_ratio=float(np.std(Z0[T] - y[T]) / sdr))
    Ra, Ss = np.abs(y - mu), np.abs(y - mu) / sig
    ctx = ctx_of(Ss, s, st, Dtr, C, T, rng); ctx_a = dict(ctx, S_cal=Ra[C])
    rows, _, hws = evaluate(y, mu, sig, Dtr, C, T, scores, K, ctx, ctx_a, dg, dg * mu, extra)
    for G in (10, 20):                                                         # policy: informed bins == naive bins
        a, b = hws["policy"][f"Mondrian CP, rule bins {G}"], hws["policy"]["Mondrian CP" + ("" if G == 10 else ", 20 bins")]
        assert np.array_equal(a, b), (name, rep, G)
    R = ref(name); pol = f"{DS.POLICY[name]}5"
    for r in rows:                                                             # reproduction check
        if r["rule"] == "policy" and r["method"] in ("JOMI stud", "SCP stud", "Mondrian CP"):
            assert (r["n_sel"], r["n_cov"]) == tuple(R.loc[(rep, pol, r["method"]), ["n_sel", "n_cov"]]), (name, rep, r["method"])
    return [dict(arm="signal", dataset=name, rep=rep, frac=0.05, **r) for r in rows], time.time() - t0


def report(df, keys, R):
    if R < 10:
        print(f"   {len(df)} rows; checks passed (code check: coverage not printed)", flush=True); return
    g = df.groupby(keys + ["method"]); cov = (g.n_cov.sum() / g.n_sel.sum()).unstack(-1).round(3)
    print(cov.to_string(), flush=True)


def main():
    mode, R, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if R < 10 and "STRESS_OUT" not in os.environ:
        raise SystemExit("code check (R < 10) needs STRESS_OUT pointing to a scratch folder")
    if mode == "twomap":
        name = sys.argv[4] if len(sys.argv) > 4 else "fia"
        DS.load(name); t0 = time.time()
        with Pool(W) as p:
            res = p.map(one_twomap, [(name, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a, _ in res for x in a])
        df.to_csv(os.path.join(OUT_W, f"st_twomap_{name}.csv"), index=False)
        print(f"== twomap {name}: R={R}, {time.time() - t0:.0f}s (checks passed)", flush=True)
        report(df, ["frac", "rule"], R)
    elif mode == "signal":
        names = sys.argv[4:] or NAMES
        for nm in names:
            DS.load(nm)
        for nm in names:
            t0 = time.time()
            with Pool(W) as p:
                res = p.map(one_signal, [(nm, r) for r in range(R)], chunksize=1)
            df = pd.DataFrame([x for a, _ in res for x in a])
            df.to_csv(os.path.join(OUT_W, f"st_signal_{nm}.csv"), index=False)
            print(f"== signal {nm}: R={R}, {time.time() - t0:.0f}s (checks passed)", flush=True)
            report(df, ["rule"], R)
    else:
        raise SystemExit("mode: twomap | signal")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
