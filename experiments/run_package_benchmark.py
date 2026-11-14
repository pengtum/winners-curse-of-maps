"""Runtime and memory benchmark of mapselect (paper/PACKAGE_BENCHMARK_DESIGN.md). Each cell runs in its own single-threaded process.
    ~/miniforge3/envs/favela/bin/python -I -u experiments/run_package_benchmark.py all
    ~/miniforge3/envs/favela/bin/python -I -u experiments/run_package_benchmark.py cell '<json>'
appends to results/fia_db/package/benchmark.jsonl; a finished cell is skipped on a rerun. No outcomes matter here; scores are synthetic
except in the 'real' and 'demo' cells, which report times only.
"""
import json, os, resource, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "package", "src")]
sys.dont_write_bytecode = True
import numpy as np                                                    # noqa: E402

OUT = os.path.join(ROOT, "results", "fia_db", "package")
JSONL = os.path.join(OUT, "benchmark.jsonl")


def clock(f, repeat=1):
    best, val = np.inf, None
    for _ in range(repeat):
        t = time.perf_counter(); val = f(); best = min(best, time.perf_counter() - t)
    return best, val


def instance(rule, m, n, K, seed=0):
    r = np.random.default_rng([seed, m, n, K])
    d = dict(xa=r.random(n), xt=r.random(m), Sa=np.abs(r.standard_normal(n)))
    if rule == "quota":
        G = 50; d["ga"], d["gt"] = r.integers(0, G, n), r.integers(0, G, m); d["Kg"] = {g: max(1, K // G) for g in range(G)}
    if rule == "budget":
        d["ca"], d["ct"] = 1 + r.random(n), 1 + r.random(m)
        o = np.argsort(-(d["xt"] / d["ct"])); d["B"] = float(np.cumsum(d["ct"][o])[K - 1])
    return d


def select(rule, d, ms, K):
    if rule == "top_k":
        return ms.top_k(d["xa"], d["xt"], K)
    if rule == "quota":
        return ms.quota(d["xa"], d["xt"], d["ga"], d["gt"], d["Kg"])
    return ms.budget(d["xa"], d["ca"], d["xt"], d["ct"], d["B"])


def competent(rule, d, sel, rows):
    """A careful per-unit implementation: thresholds and prefix sums computed once, then O(n) membership and a selection
    of the quantile for each selected unit, without noticing what units share. Time for the given rows."""
    S = d["Sa"]; xa = d["xa"]; xt = d["xt"]; K = len(sel)
    def quantile(mask):
        N = int(mask.sum()); k = int(np.ceil(.9 * (N + 1) - 1e-9 * (N + 1)))
        return np.partition(S[mask], k - 1)[k - 1] if k <= N else np.inf
    if rule == "top_k":
        T = np.partition(xt, len(xt) - K - 1)[len(xt) - K - 1]
        fn = lambda j: quantile(xa > T)
    elif rule == "quota":
        order = np.lexsort((-xt, d["gt"])); gs = d["gt"][order]
        tall, tminus = {}, {}
        for g, k in d["Kg"].items():
            a, b = np.searchsorted(gs, g), np.searchsorted(gs, g, side="right")
            tall[g], tminus[g] = xt[order[a + k - 1]], xt[order[a + k]]
        Tc_other = np.array([tall[g] for g in d["ga"]]) if len(d["ga"]) < 2e5 else np.vectorize(tall.get)(d["ga"])
        Tc_own = np.array([tminus[g] for g in d["ga"]]) if len(d["ga"]) < 2e5 else np.vectorize(tminus.get)(d["ga"])
        fn = lambda j: quantile(xa > np.where(d["ga"] == d["gt"][j], Tc_own, Tc_other))
    else:
        rt, rc = xt / d["ct"], xa / d["ca"]; a = np.argsort(rt); rs = rt[a]; suf = np.r_[np.cumsum(d["ct"][a][::-1])[::-1], 0.0]
        above = suf[np.searchsorted(rs, rc, side="right")]
        fn = lambda j: quantile(above - d["ct"][j] * (rt[j] > rc) + d["ca"] <= d["B"])
    t = time.perf_counter()
    for r in rows:
        fn(sel.selected[r])
    return time.perf_counter() - t


def literal(rule, d, sel, nswap, ru):
    """Literal swaps done in place (kinder to the swap than copying arrays), rerunning the rule once per swap."""
    r = np.random.default_rng(1); n = len(d["xa"]); K = len(sel)
    rl = {"top_k": lambda: ru.topk(K)(None, dict(mu=d["xt"])),
          "quota": lambda: ru.quota(d["Kg"])(None, dict(mu=d["xt"], g=d["gt"])),
          "budget": lambda: ru.knapsack_greedy(d["B"])(None, dict(v=d["xt"], c=d["ct"]))}[rule]
    t = time.perf_counter()
    for _ in range(nswap):
        i, j = r.integers(n), sel.selected[r.integers(K)]
        keep = d["xt"][j]; d["xt"][j] = d["xa"][i]
        if rule == "budget":
            kc = d["ct"][j]; d["ct"][j] = d["ca"][i]
        if rule == "quota":
            kg = d["gt"][j]; d["gt"][j] = d["ga"][i]
        rl()
        d["xt"][j] = keep
        if rule == "budget":
            d["ct"][j] = kc
        if rule == "quota":
            d["gt"][j] = kg
    return (time.perf_counter() - t) / nswap


def cell_refsets(c):
    import mapselect as ms
    from mapselect import rules as ru
    rule, m, n, K = c["rule"], c["m"], c["n"], c["K"]
    d = instance(rule, m, n, K)
    t_sel, sel = clock(lambda: select(rule, d, ms, K), 3 if m <= 10 ** 6 else 1)
    t_q, q = clock(lambda: ms.jomi(d["Sa"], sel, .1), 3 if m <= 10 ** 6 else 1)
    out = dict(selected=len(sel), t_select=t_sel, t_jomi=t_q, finite=float(np.isfinite(q).mean()), R_mean=float(np.mean(sel.sizes[:1000])))
    rss_ours = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    if c.get("compare", True):
        rows = np.random.default_rng(2).permutation(len(sel))[:min(20, len(sel))]
        t = competent(rule, d, sel, rows); out["t_competent_est"] = t / len(rows) * len(sel)
        out["competent_measured_units"] = len(rows)
        if len(sel) * n <= 10 ** 10:                                   # every selected-audited pair, literally: each costs a rule rerun
            ns = 2000 if m <= 10 ** 4 else (200 if m <= 10 ** 6 else 10)
        else:
            ns = 200 if m <= 10 ** 6 else 10
        out["t_swap_each"] = literal(rule, d, sel, ns, ru); out["t_swap_est"] = out["t_swap_each"] * len(sel) * n
    out["rss_gb_ours"] = rss_ours
    return out


def mask_instance(n, seed=0):
    r = np.random.default_rng([seed, n]); return r.random((n, 2)), r.random((n, 2)), float(np.sqrt(30 / (2 * np.pi * n)))


def cell_mask(c):
    import mapselect as ms
    sa, st, h = mask_instance(c["n"])
    t, sel = clock(lambda: ms.publication_mask(sa, st, h, .1, method=c["method"]))
    S = np.abs(np.random.default_rng(3).standard_normal(c["n"]))
    rows = range(0, len(sel), max(1, len(sel) // 200))
    t2 = time.perf_counter(); [sel.reference(r) for r in rows]; per = (time.perf_counter() - t2) / len(rows)
    return dict(selected=len(sel), t_select=t, t_refs_est=per * len(sel), base=len(sel.sets[0]), checksum=int(np.bitwise_xor.reduce(sel.selected)) if len(sel) else 0, rss_gb_ours=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)


def lp_problem(n, p, seed=0):
    r = np.random.default_rng([seed, n, p]); s = r.random((n, 2)); c = r.random((p - 1, 2)); h = float(np.sqrt(1.0 / p))
    Phi = np.c_[np.ones(n), np.exp(-((s[:, None] - c[None]) ** 2).sum(-1) / (2 * h ** 2))]
    S = np.abs(r.standard_normal(n)) * (1 + s[:, 0])
    return S, Phi, c, h, r


def cell_lp(c):
    from mapselect.calibrate import _active_beta, _dual_beta, _primal_beta, objective
    n, p, form, units = c["n"], c["p"], c["form"], c["units"]
    S, Phi, cen, h, r = lp_problem(n, p); pen = np.r_[False, np.ones(p - 1, bool)]; tau = .9
    phis = [np.r_[1, np.exp(-((r.random(2) - cen) ** 2).sum(-1) / (2 * h ** 2))] for _ in range(units)]
    ts, objs, qs = [], [], []
    b0 = np.zeros(p); b0[0] = np.quantile(S, tau)
    for phi in phis:
        t = time.perf_counter()
        if form == "active":
            b, rd, nf = _active_beta(S, Phi, phi, tau, .95, pen, b0); b0 = b
        elif form == "dual":
            b = _dual_beta(S, Phi, phi, tau, .95, pen)
        elif form == "primal":
            b = _primal_beta(S, Phi, phi, tau, .95, pen)
        else:
            import excp_select                                          # puts the verified excp snapshot on the path
            from excp import engines as E
            q = E.fclass(S, np.ones(n), Phi, np.ones(1), phi[None, :], .1, l1=.95, penalize=pen)[0]; b = None
        ts.append(time.perf_counter() - t)
        if b is not None:
            objs.append(objective(b, S, Phi, phi, tau, .95, pen)); qs.append(float(phi @ b))
    return dict(t_first=ts[0], t_rest=float(np.mean(ts[1:])) if len(ts) > 1 else np.nan, units_done=len(ts), obj=json.dumps([round(x, 9) for x in objs]), q=json.dumps([round(x, 9) for x in qs]),
                rss_gb_ours=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)


def cell_lp_full(c):
    """All selected units of a rule, end to end, with the reference sets and the basis held in memory."""
    import mapselect as ms
    m, n, K, p = c["m"], c["n"], c["K"], c["p"]
    r = np.random.default_rng(5); xa, xt = r.random(n), r.random(m); sa = r.random((n, 2)); cen = r.random((p - 1, 2)); h = float(np.sqrt(1.0 / p))
    def phi(s): return np.c_[np.ones(len(s)), np.exp(-((s[:, None] - cen[None]) ** 2).sum(-1) / (2 * h ** 2))]
    S = np.abs(r.standard_normal(n)) * (1 + sa[:, 0])
    sel = ms.top_k(xa, xt, K); st = r.random((len(sel), 2))
    Pa = phi(sa) if n * p < 3e8 else None
    t = time.perf_counter(); q = ms.sc_geoccp(S, sel, Pa, phi(st), .1, form="active"); el = time.perf_counter() - t
    return dict(selected=len(sel), R=len(sel.sets[0]), t_total=el, finite=float(np.isfinite(q).mean()), rss_gb_ours=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)


def cell_real(c):
    """The manuscript's own sizes and bases: one split, all selected units of the quota, old code and package."""
    import data as DS
    from run_main import fit_mu_sigma, quotas
    from excp_select import refsets, basis, methods as Mt
    import mapselect as ms
    d = DS.load(c["dataset"]); N = d["N"]; X, y, s, st = d["Xrf"], d["y"], d["coords"], d["state"]
    rng = np.random.default_rng([2027, 0, DS.ALL.index(c["dataset"])]); perm = rng.permutation(N); ntr, nca = int(.4 * N), int(.3 * N)
    Dtr, C, T = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
    mu, sig = fit_mu_sigma(X, y, Dtr, rng); S = np.abs(y - mu) / sig
    B = basis.Multiscale(s[Dtr]).with_units(st); Phi = B(s, st); pen = B.penalize; Kg = quotas(st, len(T)); aid = DS.POLICY[c["dataset"]] == "bottom"
    old = refsets.quota(mu[C], mu[T], st[C], st[T], Kg, largest=not aid); new = ms.quota(mu[C], mu[T], st[C], st[T], Kg, largest=not aid)
    ctx = dict(S_cal=S[C], Phi_cal=Phi[C], Phi_test=Phi[T], w_cal=np.ones(len(C)), w_test=np.ones(len(T)))
    t0, q0 = clock(lambda: Mt.sc_geoccp(old, .95, pen).thresholds(ctx, .1))
    t1, q1 = clock(lambda: ms.sc_geoccp(S[C], new, Phi[C], Phi[T][new.selected], .1, .95, pen, form="active"))
    fin = np.isfinite(q0)
    return dict(selected=len(new), p=Phi.shape[1], R_mean=float(new.sizes.mean()), t_old=t0, t_new=t1, q_max_diff=float(np.abs(q0[fin] - q1[fin]).max()), q_median_diff=float(np.median(np.abs(q0[fin] - q1[fin]))), same_inf=bool(np.array_equal(fin, np.isfinite(q1))))


def cell_totals(c):
    import mapselect as ms
    from mapselect import _envelope as En
    K, nR = c["K"], c["K"]; r = np.random.default_rng(6)
    t_fast, C = clock(lambda: En.envelope_exact(nR + K, K, 1 / 3, iters=16, fast=True))
    return dict(t_envelope=t_fast, rss_gb_ours=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)


def cell_planner(c):
    import mapselect as ms
    t, v = clock(lambda: ms.plan.n_for_finite(10 ** 8, 10 ** 6))
    t2, v2 = clock(lambda: ms.plan.n_for_floor(10 ** 8, 10 ** 6, 2 * 10 ** 4, .8))
    return dict(t_finite=t, n_finite=v, t_floor=t2, n_floor=v2)


def cell_demo(c):
    """ESA CCI pixels of the downloaded tiles as candidates, FIA plots as the audit. Times only: FIA plots are not a random draw of
    these pixels and their support differs, so no guarantee is claimed and no tonnage is reported."""
    import glob
    import pandas as pd
    import rasterio
    import mapselect as ms
    t0 = time.perf_counter(); keys = []
    fs = sorted(glob.glob(os.path.expanduser("~/data/esacci_biomass_v7/2020/*AGB-MERGED*.tif")))
    for f in fs:
        with rasterio.open(f) as src:
            for _, w in src.block_windows(1):
                a = src.read(1, window=w)
                a = a[a > 0].astype(np.float32)
                if len(a):
                    keys.append(a)
    x = np.concatenate(keys).astype(float); del keys
    x += 1e-6 * np.random.default_rng(1).random(len(x)); t_read = time.perf_counter() - t0
    z = pd.read_csv(os.path.join(ROOT, "data", "fia_cci_v7_2020.csv")); xa = z["agb"].to_numpy(float) + 1e-6 * np.random.default_rng(6153).random(len(z))
    e = z["Y"].to_numpy(float) - xa; sd = z["sd"].to_numpy(float); Sa = np.abs(e) / np.maximum(sd, .1 * sd.mean())
    K = int(.01 * len(x))
    t_plan, nplan = clock(lambda: ms.plan.n_for_finite(len(x), K))
    t_sel, sel = clock(lambda: ms.top_k(xa, x, K)); t_q, q = clock(lambda: ms.jomi(Sa, sel, .1))
    t_tot, v = clock(lambda: ms.selection_aware_total(x[sel.selected], e, sel))
    return dict(tiles=len(fs), candidates=len(x), audit=len(xa), K=K, R=len(sel.sets[0]), planned_audit=nplan, finite=float(np.isfinite(q).mean()), t_read=t_read, t_plan=t_plan, t_select=t_sel, t_jomi=t_q, t_total_est=t_tot,
                rss_gb_ours=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)


CELLS = dict(refsets=cell_refsets, mask=cell_mask, lp=cell_lp, lp_full=cell_lp_full, real=cell_real, totals=cell_totals, planner=cell_planner, demo=cell_demo)


def grid():
    g = []
    for rule in ("top_k", "quota", "budget"):                           # scale: m with n = m / 10 and K = m / 100
        for e in (3, 4, 5, 6, 7, 8):
            m = 10 ** e; g.append(dict(kind="refsets", rule=rule, m=m, n=m // 10, K=m // 100, compare=e <= 7))
    for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):                      # budget at 10^6 candidates: vary n, then K
        g.append(dict(kind="refsets", rule="budget", m=10 ** 6, n=n, K=10 ** 4))
    for K in (10 ** 2, 10 ** 3, 10 ** 5):
        g.append(dict(kind="refsets", rule="budget", m=10 ** 6, n=10 ** 5, K=K))
    for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        g.append(dict(kind="mask", n=n, method="tree"))
        if n <= 10 ** 5:
            g.append(dict(kind="mask", n=n, method="dense"))
    for n in (10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5):
        for p in (20, 60, 120):
            for form in ("active", "dual", "primal", "published"):
                if form == "published" and n > 10 ** 4:
                    continue                                            # dense constraints: 160 GB at 10^5
                g.append(dict(kind="lp", n=n, p=p, form=form, units=20 if form == "active" else 5))
    g.append(dict(kind="lp_full", m=10 ** 6, n=10 ** 5, K=10 ** 4, p=60))
    for ds in ("fia_db", "poverty"):
        g.append(dict(kind="real", dataset=ds))
    for K in (10 ** 2, 10 ** 3, 10 ** 4):
        g.append(dict(kind="totals", K=K))
    g.append(dict(kind="planner")); g.append(dict(kind="demo"))
    return g


def main():
    if sys.argv[1] == "cell":
        c = json.loads(sys.argv[2]); print("RESULT " + json.dumps(CELLS[c["kind"]](c), default=float)); return
    os.makedirs(OUT, exist_ok=True)
    done = {json.loads(l)["cell"] for l in open(JSONL)} if os.path.exists(JSONL) else set()
    for c in grid():
        key = json.dumps(c, sort_keys=True)
        if key in done:
            continue
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-I", "-u", __file__, "cell", key], capture_output=True, text=True, timeout=1800)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            res = json.loads(line[0][7:]) if line else dict(failed=(r.stderr or "no output").strip().splitlines()[-1][:200])
        except subprocess.TimeoutExpired:
            res = dict(failed="timeout 1800 s")
        row = dict(cell=key, wall=time.time() - t0, **c, **res)
        with open(JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(key, round(time.time() - t0, 1), str(res)[:160], flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
