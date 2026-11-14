"""Results shared by the manuscript figures: selected-unit coverage pooled over the 100 random splits,
per-group (state / country / cell) coverage, and the sampling-noise limit for the worst group.

Same conventions as experiments/make_nc.py: coverage = sum covered / sum selected over splits; SC-GeoCCP is
the l1 = 0.95 version (results/scg_*_l1fix.csv); widths are mean finite half-widths; groups count only when
selected >= 50 times over the 100 splits.
"""
import os, sys
import numpy as np
import pandas as pd

from style import RES, ROOT, ALPHA, DATASETS, TRUTH

sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.dont_write_bytecode = True
import data as DS  # noqa: E402

MAIN = ["SCP |res|", "SCP stud", "BY stud", "JOMI stud", "JOMI|state stud"]
COLS = ["rule", "method", "rep", "n_sel", "n_cov", "n_inf", "hw_mean_fin"]
NMIN = 50


def dataset_rows():
    """Row positions of the 11 settings (top = 0), with a header gap before each ground-truth group:
    ({dataset: y}, [(y, header)], total height)."""
    y, out, heads, last = 0.0, {}, [], None
    for nm, _, tr in DATASETS:
        if tr != last:
            heads.append((y + 0.45, TRUTH[tr])); y += 0.9; last = tr
        out[nm] = y + 0.5; y += 1.0
    return out, heads, y


def policy_rule(nm, pct=5):
    return f"{DS.POLICY[nm]}{pct}"


def quota_rule(nm):
    return "quota_bottom" if nm.startswith("poverty") else "quota"


def budget_rule(nm):
    """The greedy travel-cost budget: DHS funds need (lowest predicted wealth per cost), every other setting value
    (highest prediction per cost; for ERA5 this is the opposite of its bottom policy direction), as
    experiments/make_figs.py KNAP_RULE."""
    return "knapsack_need" if nm.startswith("poverty") else "knapsack"


def _csv(name):
    f = os.path.join(RES, name)
    return pd.read_csv(f) if os.path.exists(f) else None


def runs(nm):
    """One row per (split, rule, method): main run, published baselines, SC-GeoCCP, scores as QRF / bootstrap."""
    parts = []
    a = _csv(f"main_{nm}.csv")
    if a is not None:
        a = a.rename(columns={"hw_mean": "hw_mean_fin"}); parts.append(a[a.method.isin(MAIN)])
    for f in (f"bl_main_{nm}.csv", f"scg_main_{nm}_l1fix.csv", f"sf_main_{nm}.csv"):
        x = _csv(f)
        if x is not None:
            parts.append(x)
    return pd.concat([p[COLS] for p in parts], ignore_index=True)


def pooled(df, keys=("rule", "method")):
    """Pooled coverage with a 95% interval from the split-to-split variation (normal approximation)."""
    g = df.groupby(list(keys))
    out = pd.DataFrame({"cov": g.n_cov.sum() / g.n_sel.sum(), "inf": g.n_inf.sum() / g.n_sel.sum(),
                        "width": g.hw_mean_fin.mean(), "K": g.n_sel.mean(), "splits": g.rep.nunique()})
    se = g.apply(lambda x: (x.n_cov / x.n_sel).std(ddof=1) / np.sqrt(len(x)), include_groups=False)
    out["lo"], out["hi"] = out["cov"] - 1.96 * se, out["cov"] + 1.96 * se
    return out


def groups(nm, gtype="state", main=()):
    """Pooled per-group counts: published baselines, SC-GeoCCP, Normalized CP / JOMI / JOMI per state, scores.
    main adds methods that only the main run has per group (results/state_<nm>.csv; groups = states / countries),
    such as Split CP ("SCP |res|")."""
    parts = []
    for f, keep in ((f"bl_groups_{nm}.csv", None), (f"scg_groups_{nm}_l1fix.csv", None),
                    (f"scg_groups_{nm}.csv", ["SCP stud", "JOMI stud", "JOMI|state stud"]),
                    (f"sf_groups_{nm}.csv", None)):
        x = _csv(f)
        if x is not None:
            parts.append(x if keep is None else x[x.method.isin(keep)])
    x = _csv(f"state_{nm}.csv") if main and gtype == "state" else None
    if x is not None and "rule" not in x.columns:          # files from before the bottom quota existed: quota only
        x["rule"] = "quota"
    if x is not None:
        x = x[x.method.isin(main)].groupby(["rule", "method", "state"])[["n_sel", "n_cov", "n_inf"]].sum()
        parts.append(x.reset_index().rename(columns={"state": "group", "n_sel": "n", "n_cov": "c", "n_inf": "inf"})
                     .assign(gtype="state"))
    g = pd.concat(parts, ignore_index=True)
    return g[g.gtype == gtype]


# WILDS country name -> Natural Earth ADM0_A3, display name
DHS = {"angola": ("AGO", "Angola"), "benin": ("BEN", "Benin"), "burkina_faso": ("BFA", "Burkina Faso"),
       "cameroon": ("CMR", "Cameroon"), "cote_d_ivoire": ("CIV", "Côte d'Ivoire"),
       "democratic_republic_of_congo": ("COD", "DR Congo"), "ethiopia": ("ETH", "Ethiopia"), "ghana": ("GHA", "Ghana"),
       "guinea": ("GIN", "Guinea"), "kenya": ("KEN", "Kenya"), "lesotho": ("LSO", "Lesotho"),
       "malawi": ("MWI", "Malawi"), "mali": ("MLI", "Mali"), "mozambique": ("MOZ", "Mozambique"),
       "nigeria": ("NGA", "Nigeria"), "rwanda": ("RWA", "Rwanda"), "senegal": ("SEN", "Senegal"),
       "sierra_leone": ("SLE", "Sierra Leone"), "tanzania": ("TZA", "Tanzania"), "togo": ("TGO", "Togo"),
       "uganda": ("UGA", "Uganda"), "zambia": ("ZMB", "Zambia"), "zimbabwe": ("ZWE", "Zimbabwe")}


# GROUND's per-sample files of the GeoFM settings (read-only, no pickle), == experiments/run_geofm.py
GROUND = os.path.join(os.path.dirname(DS.DS), "real_world_study", "results_8ds")


def ground(name):
    """lon, lat, region codes and region names of one GROUND file (e.g. m3leo_scores.npz)."""
    z = np.load(os.path.join(GROUND, name), allow_pickle=False)
    return {k: z[k] for k in ("lon", "lat", "region", "region_names", "in_train") + (("region_country",)
                                                                                    if "region_country" in z.files else ())}


def fia_states():
    """State of each FIA group code: the pd.factorize order of the FIA table reported (experiments/data.py DS.FIA)."""
    if DS.FIA == "fia":                         # old table: latest record per location, as checked on 2026-10-27
        return {0: "FL", 1: "ME", 2: "MO", 3: "CA", 4: "AZ", 5: "WA", 6: "MT", 7: "VT", 8: "NM", 9: "CO"}
    r = pd.read_csv(os.path.join(ROOT, "data", "fia_fiadb95_plots.csv"), usecols=["region"]).region
    return dict(enumerate(pd.factorize(r)[1]))


def dhs_countries():
    """Country name of each DHS group code: the pd.factorize order of the cluster table (experiments/data.py)."""
    c = pd.read_csv(os.path.join(ROOT, "data", "povertymap_clusters.csv"), usecols=["country"]).country
    return list(pd.factorize(c)[1])


def worst(g, rule, method, nmin=NMIN):
    """(worst-group coverage, +inf share in that group, number of groups, independent-binomial limit null_worst; the
    figures take the limit from noise_limit())."""
    x = g[(g.rule == rule) & (g.method == method) & (g.n >= nmin)]
    if x.empty:
        return np.nan, np.nan, 0, np.nan
    cv = (x.c / x.n).values; k = cv.argmin()
    return cv[k], (x.inf / x.n).values[k], len(x), null_worst(x.n.values)


_LIM = None


def noise_limit(nm, gtype, n=None):
    """Sampling-noise limit of the worst group under the quota rule: the independent-binomial column of
    results/noise_limits.csv (experiments/make_mondrian_noise.py, which also gives the split-level bootstrap check;
    the two differ by at most 0.003, so by the rule in paper/MONDRIAN_NOISE_DESIGN.md the binomial limit is kept),
    or null_worst(n) when that file has no entry. Raises if neither is available."""
    global _LIM
    if _LIM is None:
        f = os.path.join(RES, "noise_limits.csv")
        _LIM = pd.read_csv(f).set_index(["dataset", "gtype"]).binomial if os.path.exists(f) else pd.Series(dtype=float)
    v = _LIM.get((nm, gtype), np.nan)
    if np.isfinite(v):
        return float(v)
    if n is None or not len(n):
        raise ValueError(f"no sampling-noise limit for {nm}/{gtype}: run experiments/make_mondrian_noise.py or pass n")
    return null_worst(n)


def null_worst(n, reps=4000, seed=0):
    """Expected worst-group coverage if every group had coverage exactly 1 - alpha (binomial noise only)."""
    n = np.asarray(n)
    sims = np.random.default_rng(seed).binomial(n[None, :], 1 - ALPHA, (reps, len(n))) / n[None, :]
    return float(sims.min(1).mean())


FM = {"poverty": ("DHS wealth", "bottom"), "m3leo": ("Biomass (ESA-CCI product)", "top"),
      "floods": ("Flooded fraction", "top"), "usavars": ("Tree cover (remote-sensing product)", "bottom")}


def geofm(ds, backbone):
    f = os.path.join(RES, f"fm_main_{ds}_{backbone}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None
