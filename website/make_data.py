"""Write website/data.json: aggregate results, boundaries, and one map's predictions at the FIA plots for the live lab.

Unit-level data: FIA only (plots rebuilt from the public-domain FIADB 9.5, coordinates as perturbed by FIA). DHS appears as
country summaries only, as its terms require. Other sources appear as summaries, because their redistribution terms are
not recorded in this project.
The FIA maps are those of the carbon runs, fold 0 (experiments/run_carbon.py: seeds [5150, 0] and [5151, 0]).
    ~/miniforge3/envs/favela/bin/python website/make_data.py
also prints a Python reference for the lab's defaults, to compare with the page.
"""
import json, os, sys
import numpy as np
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
sys.path[:0] = [os.path.join(ROOT, "experiments"), ROOT]; sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
import run_carbon as RC                                               # noqa: E402
from run_main import fit_mu_sigma                                     # noqa: E402

SD = os.path.join(ROOT, "figs", "source_data"); PK = os.path.join(ROOT, "results", "fia_db")
LAB = {"poverty": "DHS wealth", "poverty_lab": "DHS wealth, no imagery", "fia_db": "FIA biomass", "era5": "ERA5 temperature", "camels": "CAMELS runoff",
       "voting": "County vote share", "seattle": "Seattle house price", "health_stroke": "Stroke*", "health_diabetes": "Diabetes*",
       "health_bphigh": "High blood pressure*", "health_obesity": "Obesity*"}
STATES = ["Arizona", "California", "Colorado", "Florida", "Maine", "Missouri", "Montana", "New Mexico", "Vermont", "Washington"]


def r(x, d=4):
    return float(round(float(x), d))


def ring(coords, d=2):
    out, last = [], None
    for x, y in coords:
        p = (round(x, d), round(y, d))
        if p != last:
            out.append(list(p)); last = p
    return out


def polys(geom):
    g = geom["coordinates"]
    return [[ring(rg) for rg in p] for p in (g if geom["type"] == "MultiPolygon" else [g])]


def boundaries():
    ne = os.path.join(ROOT, "data", "natural_earth_110m_v5.1.2")
    s = json.load(open(os.path.join(ne, "ne_110m_admin_1_states_provinces.geojson")))
    us = [dict(n=f["properties"]["name"], fia=f["properties"]["name"] in STATES, p=polys(f["geometry"])) for f in s["features"]
          if f["properties"].get("iso_a2") == "US" and f["properties"]["name"] not in ("Alaska", "Hawaii")]
    c = json.load(open(os.path.join(ne, "ne_110m_admin_0_countries.geojson")))
    af = [dict(iso=f["properties"]["ISO_A3"] if f["properties"]["ISO_A3"] != "-99" else f["properties"]["ADM0_A3"], n=f["properties"]["NAME"],
               p=polys(f["geometry"])) for f in c["features"] if f["properties"].get("CONTINENT") == "Africa"]
    return us, af


def fia_lab():
    d = DS.load(DS.FIA); y = d["y"]; N = d["N"]; lon, lat = d["Xrf"][:, 0], d["Xrf"][:, 1]
    st = d["state"]
    raw = pd.read_csv(os.path.join(ROOT, "data", "fia_fiadb95_plots.csv"), usecols=["region"])
    perm = np.random.default_rng([5150, 0]).permutation(N); ntr = int(0.4 * N); Dtr, rest = perm[:ntr], perm[ntr:]
    out = dict(lon=[r(v, 3) for v in lon[rest]], lat=[r(v, 3) for v in lat[rest]], y=[r(v, 2) for v in y[rest]], g=[int(v) for v in st[rest]])
    for mp in ("site", "full"):
        X = d["Xrf"][:, [c for c in range(d["Xrf"].shape[1]) if c not in RC.DROP[mp]]]
        mu, sig = fit_mu_sigma(X, y, Dtr, np.random.default_rng([5151, 0]))
        e = y[rest] - mu[rest]
        out[mp] = dict(mu=[r(v, 3) for v in mu[rest]], s=[r(v, 3) for v in sig[rest]],
                       r2=r(1 - (e ** 2).sum() / ((y[rest] - y[rest].mean()) ** 2).sum(), 3))
    names = sorted(set(raw.region))
    out["gname"] = [str(x) for x in pd.factorize(raw.region.values)[1]] if len(names) else []
    return out


def reference(lab, mp="site", share=0.01, audit=0.5, T=400):
    """The lab's computation in Python: top share of candidates by the map, VCS rule on the survey-calibrated estimate, always-deducted
    selection-aware estimate, split CP and JOMI with the normalized score."""
    y = np.array(lab["y"]); mu = np.array(lab[mp]["mu"]); s = np.array(lab[mp]["s"]); M = len(y); e = y - mu; S = np.abs(e) / s
    rng = np.random.default_rng(1); res = dict(cs=0, cj=0, k=0, inf=0, ma=0, sa=0)
    for _ in range(T):
        p = rng.permutation(M); n = int(audit * M); C, Tt = p[:n], p[n:]
        K = max(1, round(share * len(Tt))); o = Tt[np.argsort(-mu[Tt])]; J, th = o[:K], mu[o[K]]; R = C[mu[C] > th]
        qs = np.sort(S[C])[int(np.ceil(.9 * (n + 1))) - 1]; nR = len(R)
        qj = np.sort(S[R])[int(np.ceil(.9 * (nR + 1) - 1e-9 * (nR + 1))) - 1] if nR >= 9 else np.inf
        res["cs"] += (S[J] <= qs).sum(); res["cj"] += (S[J] <= qj).sum(); res["k"] += K; res["inf"] += K * (nR < 9)
        est = mu[J].sum(); tot = y[J].sum()
        ma, se_ma = est + K * e[C].mean(), e[C].std(ddof=1) * np.sqrt(K + K ** 2 / n)
        res["ma"] += RC.vcs(ma, se_ma) > tot
        if nR >= 2:
            sa, se_sa = est + K * e[R].mean(), K * e[R].std(ddof=1) * np.sqrt((nR + K) / (nR * K))
            res["sa"] += max(0.0, sa - RC.Z_VCS * se_sa) > tot
    return {k: (v / res["k"] if k in ("cs", "cj", "inf") else v / T) for k, v in res.items() if k != "k"}


def main():
    D = {}
    a = pd.read_csv(os.path.join(SD, "nc_fig4a.csv"))
    a = a[a.method.isin(["Split CP", "Normalized CP", "GeoCP", "LSCP", "RLCP", "Regression kriging", "QRF", "JOMI", "SC-GeoCCP (ours)"])]
    D["cov"] = [dict(s=LAB[x.dataset], m=x.method.replace(" (ours)", ""), c=r(x.coverage), i=r(x.inf_share, 3)) for x in a.itertuples()]
    b = pd.read_csv(os.path.join(SD, "nc_fig4b.csv"))
    pol = {k: v for k, v in DS.POLICY.items()}
    b = b[[pol.get(ds, "top") == dr for ds, dr in zip(b.dataset, b.direction)]]
    D["share"] = [dict(s=LAB[x.dataset], m=x.method, p=int(x.pct), c=r(x.coverage), i=r(x.inf_share, 3)) for x in b.itertuples()]
    d = pd.read_csv(os.path.join(SD, "si_fig_carbon_protocol_d_f.csv"))
    D["risk"] = {m: {s: [[r(x, 3), r(y)] for x, y in zip(h.delta, h.risk)] for s, h in g.groupby("series")} for m, g in d.groupby("map")}
    f = pd.read_csv(os.path.join(SD, "nc_fig6c_floor.csv"))
    D["floor"] = [dict(n=x.country, iso=x.iso, k=int(x.selected_per_split), f=r(x.floor, 3), c=r(x.coverage, 3)) for x in f.itertuples()]
    g = pd.read_csv(os.path.join(SD, "nc_fig6c.csv"))
    keep = {"SCP |res|": "Split CP", "SCP stud": "Normalized CP", "GeoCP": "GeoCP", "JOMI stud": "JOMI", "SC-GeoCCP": "SC-GeoCCP"}
    g = g[g.key.isin(keep)]
    D["country"] = [dict(iso=x.iso, n=x.country, m=keep[x.key], c=r(x.coverage, 3), sel=int(x.n_selected_pooled)) for x in g.itertuples()]
    b = pd.read_csv(os.path.join(PK, "package", "summary_refsets.csv")); b = b[(b.n == b.m // 10) & (b.K == b.m // 100)].drop_duplicates(["rule", "m"])
    D["bench"] = [dict(rule=x.rule, m=x.m, ours=r(x.ours, 6), comp=None if pd.isna(x.t_competent_est) else r(x.t_competent_est, 5),
                       swap=None if pd.isna(x.t_swap_est) else float("%.4g" % x.t_swap_est), gb=r(x.rss_gb_ours, 2)) for x in b.itertuples()]
    l = pd.read_csv(os.path.join(PK, "package", "summary_lp.csv")); l = l[(l.p == 60) & l.t_first.notna()]
    D["lp"] = [dict(n=int(x.n), f=x.form, t=r(x.t_rest, 5)) for x in l.itertuples()]
    s = pd.read_csv(os.path.join(PK, "planner", "summary_carbon.csv"))
    D["arms"] = [dict(map=x[0], q=x[1], n=int(x[2]), pf=r(x[3], 3), fin=r(x[4], 3), risk=r(x[5], 3), share=r(x[6], 3), survey=r(x[7], 3))
                 for x in s[["map", "q", "n", "pf", "fin", "sa_always", "share_sa", "ma_rule"]].itertuples(index=False)]
    D["us"], D["africa"] = boundaries()
    lab = fia_lab(); D["fia"] = lab
    json.dump(D, open(os.path.join(H, "data.json"), "w"), separators=(",", ":"))
    print("data.json", os.path.getsize(os.path.join(H, "data.json")) // 1024, "KB;", len(lab["y"]), "FIA plots; R2", lab["site"]["r2"], lab["full"]["r2"], "; groups", lab["gname"])
    for mp in ("site", "full"):
        print("python reference", mp, {k: round(v, 3) for k, v in reference(lab, mp).items()})


if __name__ == "__main__":
    main()
