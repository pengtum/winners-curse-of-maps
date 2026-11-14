"""Experiment B of paper/MONDRIAN_ADVANTAGE_DESIGN.md (written before the run): selection rules that do not rank the
map's prediction alone, where bins of the predicted value need not approximate the reference set.

Rules (5% of the test units, policy direction d = +1 top / -1 bottom; ranked by a score s known for every unit):
  policy      s = d mu                     (the main 5% rule; reproduction check)
  upper       s = d mu + 1.645 sigma        ("could reach the target": precautionary screening)
  lower       s = d mu - 1.645 sigma        ("surely reaches the target": certification-style selection)
  second map  s = d mu2, mu2 a random forest on the coordinates only (same training fold and settings), while the
              intervals still come from the main map mu
JOMI's reference set for top-K of s is {i in C : s_i > theta}, theta the (K+1)-th largest test score (refsets.topk).
Methods: split CP (|y - mu|), normalized CP (|y - mu| / sigma), Mondrian CP with ten bins of mu ("naive", the main
version of [S23]), Mondrian CP with ten and twenty bins of the rule's own score s ("informed"), JOMI with the
normalized and the additive score. Bin edges use the training fold only.
Same seeds as run_main.py (mu, sigma, folds and the policy selection are identical; mu2 uses its own stream
[2029, rep, dataset], with the same 1e-9 jitter as mu so that the closed-form reference set has no ties).
Checked before writing: per-split counts of JOMI (normalized) and normalized CP under the policy
rule equal results/main_<ds>.csv, and naive Mondrian CP equals results/md_main_<ds>.csv.
    ~/miniforge3/envs/favela/bin/python -u experiments/run_mondrian_rules.py <R> <workers> [datasets...]
writes results/mr_main_<ds>.csv
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
from run_main import fit_mu_sigma, rf                                 # noqa: E402
from run_baselines import q_conf, ALPHA                               # noqa: E402
from run_mondrian import mondrian, NAMES                              # noqa: E402

OUT = os.path.join(ROOT, "results")
Z, FRAC = 1.645, 0.05
_REF = {}


def ref(name):
    """Stored per-split counts for the reproduction checks."""
    if name not in _REF:
        m = pd.read_csv(os.path.join(OUT, f"main_{name}.csv"), usecols=["rep", "rule", "method", "n_sel", "n_cov"])
        md = pd.read_csv(os.path.join(OUT, f"md_main_{name}.csv"), usecols=["rep", "rule", "method", "n_sel", "n_cov"])
        _REF[name] = pd.concat([m, md]).set_index(["rep", "rule", "method"])
    return _REF[name]


def one(args):
    name, rep = args
    d = DS.load(name); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([2027, rep, DS.ALL.index(name)])          # == run_main.py
    perm = rng.permutation(N); ntr, nca = int(0.4 * N), int(0.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    m = len(T); t0 = time.time()
    mu, sig = fit_mu_sigma(X, y, Dtr, rng)
    r2 = np.random.default_rng([2029, rep, DS.ALL.index(name)])
    mu2 = rf(s[Dtr], y[Dtr], int(r2.integers(2 ** 31))).predict(s) + 1e-9 * r2.standard_normal(N)   # break exact ties, as for mu
    dg = 1.0 if DS.POLICY[name] == "top" else -1.0
    K = max(1, int(round(FRAC * m)))
    Ra, Ss = np.abs(y - mu), np.abs(y - mu) / sig
    scores = {"policy": dg * mu, "upper": dg * mu + Z * sig, "lower": dg * mu - Z * sig, "second map": dg * mu2}
    ctx = dict(S_cal=Ss[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T], s_train=s[Dtr],
               w_train=np.ones(len(Dtr)), g_cal=st[C], g_test=st[T], rng=rng)
    ctx_a = dict(ctx, S_cal=Ra[C])
    q10, _ = mondrian(Ss, mu, Dtr, C, T, 10)
    hw_all = {"SCP |res|": q_conf(Ra[C]) * np.ones(m), "SCP stud": q_conf(Ss[C]) * sig[T], "Mondrian CP": q10 * sig[T]}
    R = ref(name); pol = f"{DS.POLICY[name]}5"
    J0 = set(T[refsets.topk(scores["policy"][C], scores["policy"][T], K).selected])
    rows = []
    for rn, sc in scores.items():
        sel = refsets.topk(sc[C], sc[T], K)
        k = sel.selected; J = T[k]; e = np.abs(y[J] - mu[J])
        hw = {mn: h[k] for mn, h in hw_all.items()}
        for G in (10, 20):
            qs, _ = mondrian(Ss, sc, Dtr, C, T, G)
            hw[f"Mondrian CP, rule bins {G}"] = qs[k] * sig[J]
        hw["JOMI stud"] = Mt.jomi(sel).thresholds(ctx, ALPHA) * sig[J]
        hw["JOMI |res|"] = Mt.jomi(sel).thresholds(ctx_a, ALPHA)
        ov = len(J0.intersection(J)) / len(J)
        for mn, hh in hw.items():
            cov = e <= hh; fin = np.isfinite(hh)
            if rn == "policy" and mn in ("JOMI stud", "SCP stud", "Mondrian CP"):
                assert (len(J), int(cov.sum())) == tuple(R.loc[(rep, pol, mn), ["n_sel", "n_cov"]]), (name, rep, mn)
            rows.append(dict(dataset=name, rep=rep, rule=rn, method=mn, n_sel=len(J), n_cov=int(cov.sum()),
                             n_inf=int((~fin).sum()), hw_mean_fin=float(hh[fin].mean()) if fin.any() else np.nan,
                             hw_med=float(np.median(hh)), overlap_policy=ov, sig_ratio=float(sig[J].mean() / sig[T].mean()),
                             resid_mean=float(np.mean(dg * (y[J] - mu[J]))), cor_mu_mu2=float(np.corrcoef(mu[T], mu2[T])[0, 1])))
    return rows, time.time() - t0


def main():
    R, W = int(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[3:] or NAMES
    for nm in names:
        DS.load(nm)
    for nm in names:
        t0 = time.time()
        with Pool(W) as p:
            res = p.map(one, [(nm, r) for r in range(R)], chunksize=1)
        df = pd.DataFrame([x for a, _ in res for x in a])
        df.to_csv(os.path.join(OUT, f"mr_main_{nm}.csv"), index=False)
        g = df.groupby(["rule", "method"]); cov = (g.n_cov.sum() / g.n_sel.sum()).unstack(1).round(3)
        print(f"== {nm}: R={R}, {time.time() - t0:.0f}s (checks passed)\n{cov.to_string()}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
