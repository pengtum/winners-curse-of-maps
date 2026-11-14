"""Out-of-design evaluation (paper/OOD_DESIGN.md, revised after two design reviews on 2026-10-29).

Held-out groups (design "loco": DHS countries, full table of 19,669 clusters; "loso": FIA states, the 23,068
plots inventoried in 2000 or later, states with >= 300 plots). One job = one held-out group g x one map draw;
inside it, 40 calibration / test re-splits of g, so the guarantees are judged conditional on each map.
Per map draw, the other groups are split by units ~70/30 into training groups and calibration groups, and g
into G_tr (30%, used only by the R0 map) and the rest, which every inner split halves into C_g and T_g. If the
calibration groups hold fewer units than |C_g| (FIA state 1, map 1 only), the group split is redrawn.
  R0  map on G_tr + training-group units (4,000 in all); calibrate on C_g       (within the design)
  R1  map on 4,000 training-group units only;          calibrate on C_g       (within: map transfer, Togo case)
  R2  the R1 map; calibrate on |C_g| units drawn from the calibration groups (outside: no exchangeability)
  R1-n  the R1 map; calibrate on n of the C_g units, n in SURVEY (local survey size)
R0 -> R1 isolates training the map elsewhere, R1 -> R2 calibrating elsewhere (same map, test units and
selection; R1-n on the first 20 re-splits). Rules on T_g in the policy direction: 5% and 25.2% (DHS; the Novissi share) or 5% and 20% (FIA),
plus "all" (marginal coverage). Methods: split CP |res|, normalized CP, BY, JOMI |res| and stud, CQR, QRF,
regression kriging, bootstrap ensemble, GeoCP (published), LSCP, RLCP (bandwidths tuned on the map's training
fold, as in the main analysis). SC-GeoCCP is not run here: its group column is constant inside g, and its
bumps, anchored on training coordinates outside g (R1, R2), localize only partly within g; it was not evaluated there. Model-based intervals do not
use the calibration set and are identical in R1 and R2 (asserted). For DHS at 25.2%, a 90% upper bound on the
selected clusters' mean wealth: map + calibration mean residual with margin s_C (K + K^2/n_C)^1/2, and the
reference-set central-limit bound (as experiments/pilot_togo_novissi.py).
Design "block": cluster-randomized splits. DHS: countries as blocks; FIA (post-2000, site+height map):
k-means blocks (k = 25 or 100); "random" assigns single units (the main design, same pipeline). Blocks are shuffled and allotted ~40/30/30 (by units) to training /
calibration / test, and 4,000 / 3,000 / 3,000 units are drawn from them; 100 repetitions; SC-GeoCCP included for rules with K <= 200.
    ~/miniforge3/envs/favela/bin/python -u experiments/run_ood.py <workers> <spec> ...
    spec: loco:poverty  loso:fia_post2000:siteheight  loso:fia_post2000:full  block:poverty:country
          block:fia_post2000:k25  block:fia_post2000:k100
writes results/ood_<spec>.csv (and ood_<spec>_bounds.csv for loco); parts in results/cache/ood_parts/
"""
import os, sys, time, warnings
from math import ceil
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from run_main import fit_mu_sigma                                     # noqa: E402
import run_baselines as RB                                            # noqa: E402
from excp_select import refsets, basis, methods as Mt                 # noqa: E402
from excp import family as F                                          # noqa: E402

ALPHA, L1, DELTA = 0.1, 0.95, 0.1
OUT = os.path.join(ROOT, "results")
PARTS = os.path.join(DS.CACHE, "ood_parts")
NTR, NCA = 4000, 3000
MIN_GROUP = 300
MAPS, INNER = 5, 40
SURVEY_INNER = 20                         # local-survey-size arms on the first 20 inner splits only
SCG_MAX_K = 200                           # SC-GeoCCP (one LP per selected unit) only for rules with K <= 200
SURVEY = (50, 100, 200)
FRACS = {"poverty": (0.05, 0.252), "fia_post2000": (0.05, 0.20), "fia_db": (0.05, 0.20)}
DROP = {"siteheight": [6, 7, 8, 11], "full": [], "": []}     # Xrf columns (run_carbon.DROP): BA, TPH, DIA, stand size
SEED = {"loco": 31, "loso": 32, "block": 33}


def full(name, feats):
    """Full table (not subsampled), cached; Xrf restricted to the feature set."""
    f = os.path.join(DS.CACHE, f"{name}_full.npz")
    if os.path.exists(f):
        d = dict(np.load(f))
    else:
        d = DS._build(name); np.savez_compressed(f, **d)
    keep = [c for c in range(d["Xrf"].shape[1]) if c not in DROP[feats]]
    return dict(d, Xrf=d["Xrf"][:, keep], N=len(d["y"]))


def split_groups(st, groups, rng, frac=0.7):
    """Shuffle groups and give them to the first set until it holds >= frac of their units."""
    o = rng.permutation(groups); cnt = np.array([(st == u).sum() for u in o]); cum = np.cumsum(cnt) / cnt.sum()
    k = min(int(np.searchsorted(cum, frac)) + 1, len(o) - 1)       # both sets non-empty
    return o[:k], o[k:]


class Map:
    """mu, sigma, tuned bandwidths and model-based intervals of one training fold, on the units Q."""
    def __init__(self, d, Dtr, Q, rng):
        X, y, s = d["Xrf"], d["y"], d["coords"]
        self.mu, self.sig = fit_mu_sigma(X, y, Dtr, rng)
        self.bw = RB.tune_local(X, y, s, Dtr, rng)
        seed = int(rng.integers(2 ** 31))
        self.Q = np.asarray(Q); pos = np.full(len(y), -1); pos[self.Q] = np.arange(len(self.Q)); self.pos = pos
        self.model = {"QRF": RB.qrf_bounds(X[Dtr], y[Dtr], X[self.Q], seed),
                      "Regression kriging": RB.kriging_bounds(X, y, s, Dtr, self.Q, self.mu, rng),
                      "Bootstrap ensemble + residual": RB.bootstrap_bounds(X, y, Dtr, self.Q, seed)[0]}
        self.hq = [HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.05,
                                                 min_samples_leaf=20, random_state=seed).fit(X[Dtr], y[Dtr]).predict(X)
                   for q in (ALPHA / 2, 1 - ALPHA / 2)]
        self.Dtr = Dtr


def evaluate(d, M, C, T, fracs, largest, base, rng, extra=None, bounds_rows=None):
    """Rows for one (map, calibration set, test set): every rule x method."""
    y, s, st = d["y"], d["coords"], d["state"]
    mu, sig = M.mu, M.sig; Ra = np.abs(y - mu); Ss = Ra / sig; m = len(T); pT = M.pos[T]
    ctx = dict(S_cal=Ss[C], w_cal=np.ones(len(C)), w_test=np.ones(m), s_cal=s[C], s_test=s[T],
               s_train=s[M.Dtr], w_train=np.ones(len(M.Dtr)), g_cal=st[C], g_test=st[T], rng=rng)
    if extra:
        ctx.update(extra(C, T))
    ctx_a = dict(ctx, S_cal=Ra[C])
    bnd = {mn: (L[pT], U[pT]) for mn, (L, U) in M.model.items()}
    qc = RB.q_conf(np.maximum(M.hq[0][C] - y[C], y[C] - M.hq[1][C]))
    bnd["CQR"] = (M.hq[0][T] - qc, M.hq[1][T] + qc)
    for nm, meth in (("GeoCP", F.geocp_v1(M.bw["h1"])), ("LSCP", F.lscp(min(M.bw["k"], len(C) - 1))),
                     ("RLCP", F.excp_rlcp(M.bw["hR"]))):
        q = meth.thresholds(ctx_a, ALPHA); bnd[nm] = (mu[T] - q, mu[T] + q)
    for nm, c, sc in (("SCP |res|", ctx_a, 1.0), ("SCP stud", ctx, sig[T])):
        hw = F.split_cp().thresholds(c, ALPHA) * sc; bnd[nm] = (mu[T] - hw, mu[T] + hw)
    rows = []
    sels = {"all": None}
    for f in fracs:
        K = max(1, int(ceil(f * m))) if abs(f - 0.252) < 1e-9 else max(1, int(round(f * m)))
        sels[f"{'top' if largest else 'bottom'}{f * 100:g}"] = refsets.topk(mu[C], mu[T], K, largest)
    for rn, sel in sels.items():
        k = np.arange(m) if sel is None else sel.selected; J = T[k]
        res = {mn: (L[k], U[k]) for mn, (L, U) in bnd.items()}
        if sel is not None:
            meths = [("BY stud", Mt.by(sel, m), Ss, sig, ctx), ("JOMI |res|", Mt.jomi(sel), Ra, None, ctx_a),
                     ("JOMI stud", Mt.jomi(sel), Ss, sig, ctx)]
            if "Phi_cal" in ctx and len(sel) <= SCG_MAX_K:
                meths.append(("SC-GeoCCP", Mt.sc_geoccp(sel, L1, ctx["pen"]), Ss, sig, ctx))
            for mn, meth, S, sc, c in meths:
                q = meth.thresholds(c, ALPHA); hw = q * (sc[J] if sc is not None else 1.0)
                res[mn] = (mu[J] - hw, mu[J] + hw)
        for mn, (L, U) in res.items():
            cov = (y[J] >= L) & (y[J] <= U); fin = np.isfinite(U - L); hw = (U - L) / 2
            rows.append(dict(base, rule=rn, method=mn, n_sel=len(J), n_cov=int(cov.sum()), n_inf=int((~fin).sum()),
                             hw_mean_fin=float(hw[fin].mean()) if fin.any() else np.nan,
                             R_med=float(np.median(sel.sizes)) if sel is not None else np.nan))
        if bounds_rows is not None and sel is not None and rn.endswith("25.2"):
            e = y - mu; K = len(J); R = C[sel.member[0]]; nR = len(R); Lp = nR + K; z = norm.ppf(1 - DELTA)
            ub = {"calibration-mean correction": mu[J].sum() + K * e[C].mean()
                  + z * e[C].std(ddof=1) * np.sqrt(K + K ** 2 / len(C)),
                  "reference-set CLT": (mu[J].sum() + K * e[R].mean() + z * K * e[R].std(ddof=1) * np.sqrt(Lp / (nR * K)))
                  if nR >= 2 else np.inf}
            for bn, u in ub.items():
                bounds_rows.append(dict(base, rule=rn, bound=bn, K=K, nR=nR, holds=int(y[J].sum() <= u),
                                        excess=float(u / K - y[J].mean()), bias_sel=float(e[J].mean()),
                                        bias_cal=float(e[C].mean())))
    return rows


def held_out(job):
    design, name, feats, g, mi = job
    part = os.path.join(PARTS, f"{design}_{name}_{feats}_{g}_{mi}.csv")
    if os.path.exists(part):
        return part
    t0 = time.time()
    d = full(name, feats); st = d["state"]
    ss = np.random.SeedSequence([SEED[design], DS.ALL.index(name), int(g), mi])
    r_split, r_map0, r_map1, r_inner, r_eval = [np.random.default_rng(c) for c in ss.spawn(5)]
    other_groups = np.array([u for u in np.unique(st) if u != g])
    ng = int((st == g).sum()); h = (ng - int(0.3 * ng)) // 2
    for _ in range(1000):                          # redraw only if the calibration groups hold < |C_g| units
        tr_groups, ca_groups = split_groups(st, other_groups, r_split)                  # (FIA state 1, map 1)
        pool_tr = np.flatnonzero(np.isin(st, tr_groups)); pool_ca = np.flatnonzero(np.isin(st, ca_groups))
        if len(pool_ca) >= h and len(pool_tr) >= NTR:
            break
    else:
        raise RuntimeError(f"no feasible group split for {job}")
    inside = r_split.permutation(np.flatnonzero(st == g))
    G_tr, rest = inside[: int(0.3 * ng)], inside[int(0.3 * ng):]
    D1 = r_split.choice(pool_tr, NTR, replace=False)
    D0 = np.r_[G_tr, r_split.choice(pool_tr, NTR - len(G_tr), replace=False)]
    inner = [r_inner.permutation(rest) for _ in range(INNER)]
    assert h == len(rest) // 2
    remote = [r_inner.choice(pool_ca, h, replace=False) for _ in range(INNER)]
    survey = {n: [r_inner.permutation(h)[:n] for _ in range(INNER)] for n in SURVEY if n < h}
    largest = DS.POLICY[name] != "bottom"; fracs = FRACS[name]
    rows, brows = [], ([] if name == "poverty" else None)
    M0, M1 = Map(d, D0, rest, r_map0), Map(d, D1, rest, r_map1)
    for i in range(INNER):
        C, T = inner[i][:h], inner[i][h:]
        b = dict(design=design, dataset=name, feats=feats, group=int(g), map=mi, inner=i, n_group=ng, m=len(T))
        rows += evaluate(d, M0, C, T, fracs, largest, dict(b, arm="R0", n_cal=len(C)), r_eval, bounds_rows=brows)
        r1 = evaluate(d, M1, C, T, fracs, largest, dict(b, arm="R1", n_cal=len(C)), r_eval, bounds_rows=brows)
        r2 = evaluate(d, M1, remote[i], T, fracs, largest, dict(b, arm="R2", n_cal=h), r_eval, bounds_rows=brows)
        for x, z in zip(r1, r2):                    # model-based intervals ignore the calibration set
            if x["method"] in M1.model:
                assert (x["n_cov"], x["n_inf"]) == (z["n_cov"], z["n_inf"]), (x, z)
        rows += r1 + r2
        for n, idx in (survey.items() if i < SURVEY_INNER else ()):
            rows += evaluate(d, M1, C[idx[i]], T, fracs, largest, dict(b, arm=f"R1-{n}", n_cal=n), r_eval)
    os.makedirs(PARTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False, float_format="%.6g")
    if brows is not None:
        pd.DataFrame(brows).to_csv(part.replace(".csv", "_bounds.csv"), index=False, float_format="%.6g")
    print(f"  {design} {name} {feats} g={g} map={mi}: n_g={ng}, |T|={len(rest) - h}, {time.time() - t0:.0f}s", flush=True)
    return part


def block_job(job):
    design, name, feats, blk, rep = job
    part = os.path.join(PARTS, f"{design}_{name}_{feats}_{blk}_{rep}.csv")
    if os.path.exists(part):
        return part
    t0 = time.time()
    d = full(name, feats); st, s = d["state"], d["coords"]
    if blk == "country":
        b = st
    elif blk == "random":                         # comparison: every unit its own block (the main design)
        b = None
    else:
        b = np.load(kmeans_blocks(name, feats, blk))
    ss = np.random.SeedSequence([SEED[design], DS.ALL.index(name), int(blk[1:]) if blk[0] == "k" else 0, rep])
    r_split, r_map, r_eval = [np.random.default_rng(c) for c in ss.spawn(3)]
    if b is None:
        p = r_split.permutation(d["N"]); Dtr, C, T = p[:NTR], p[NTR:NTR + NCA], p[NTR + NCA:NTR + 2 * NCA]
        tr = ca = te = ()
    else:
        u = np.unique(b); tr, rest = split_groups(b, u, r_split, 0.4)
        ca, te = split_groups(b, rest, r_split, 0.5)
        pick = lambda groups, n: r_split.choice(np.flatnonzero(np.isin(b, groups)), min(n, int(np.isin(b, groups).sum())),  # noqa: E731
                                                replace=False)
        Dtr, C, T = pick(tr, NTR), pick(ca, NCA), pick(te, NCA)
    M = Map(d, Dtr, T, r_map)
    Bs = basis.Multiscale(s[Dtr]).with_units(st); Phi = Bs(s, st)
    extra = lambda C_, T_: dict(Phi_cal=Phi[C_], Phi_test=Phi[T_], pen=Bs.penalize)                     # noqa: E731
    base = dict(design=design, dataset=name, feats=feats, blocks=blk, rep=rep, arm="block", n_cal=len(C), m=len(T),
                n_train=len(Dtr), blocks_tr=len(tr), blocks_ca=len(ca), blocks_te=len(te))
    rows = evaluate(d, M, C, T, FRACS[name], DS.POLICY[name] != "bottom", base, r_eval, extra=extra)
    os.makedirs(PARTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False, float_format="%.6g")
    print(f"  {design} {name} {feats} {blk} rep={rep}: {time.time() - t0:.0f}s", flush=True)
    return part


def kmeans_blocks(name, feats, blk):
    """k-means blocks on the coordinates, cached; built once in main() before the workers start."""
    f = os.path.join(DS.CACHE, f"{name}_blocks_{blk}.npy")
    if not os.path.exists(f):
        from sklearn.cluster import KMeans
        tmp = f + f".{os.getpid()}.tmp.npy"
        np.save(tmp, KMeans(int(blk[1:]), n_init=4, random_state=0).fit_predict(full(name, feats)["coords"]))
        os.replace(tmp, f)
    return f


def run(job):
    return held_out(job) if job[0] in ("loco", "loso") else block_job(job)


def jobs(spec):
    p = spec.split(":"); design, name = p[0], p[1]
    if design in ("loco", "loso"):
        feats = p[2] if len(p) > 2 else ""
        st = full(name, feats)["state"]; u, cnt = np.unique(st, return_counts=True)
        return [(design, name, feats, int(g), mi) for g in u[cnt >= MIN_GROUP] for mi in range(MAPS)]
    blk = p[2]; feats = "siteheight" if name.startswith("fia") else ""
    return [(design, name, feats, blk, r) for r in range(100)]


def main():
    W = int(sys.argv[1])
    for sp in sys.argv[2:]:
        J = jobs(sp); t0 = time.time()
        if J[0][0] == "block" and J[0][3][0] == "k":
            kmeans_blocks(*J[0][1:4])
        with Pool(W) as p:
            parts = p.map(run, J, chunksize=1)
        df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
        tag = sp.replace(":", "_")
        df.to_csv(os.path.join(OUT, f"ood_{tag}.csv"), index=False, float_format="%.6g")
        bf = [f.replace(".csv", "_bounds.csv") for f in parts if os.path.exists(f.replace(".csv", "_bounds.csv"))]
        if bf:
            pd.concat([pd.read_csv(f) for f in bf], ignore_index=True).to_csv(
                os.path.join(OUT, f"ood_{tag}_bounds.csv"), index=False, float_format="%.6g")
        g = df[df.rule != "all"].groupby(["arm", "rule", "method"])
        print(f"== {sp}: {len(J)} jobs, {time.time() - t0:.0f}s\n"
              + (g.n_cov.sum() / g.n_sel.sum()).unstack([0, 1]).round(3).to_string(), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
