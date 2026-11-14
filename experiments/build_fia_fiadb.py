"""FIA plot table rebuilt from FIADB 9.5 (paper/FIA_REBUILD_DESIGN.md, frozen 2026-10-30; sections 2-3).

Reads the DataMart zips {ST}_PLOT/COND/TREE.zip of the ten states straight from the zip files (nothing is
unzipped) and writes one row per plot: region, fips, lat, lon, Y, X1..X12, time (the column layout of
US_Forest_FIA.csv), plus C (live aboveground carbon) and the FIADB identifiers.

  Y  = 0.0011209 * sum_live DRYBIO_AG * TPA_UNADJ    (Mg/ha, live trees STATUSCD = 1)
  C  = 0.0011209 * sum_live CARBON_AG * TPA_UNADJ    (Mg C/ha)
  X1 elevation class (PLOT.ELEV ft -> m), X2 slope class (COND.SLOPE %), X3 aspect (COND.ASPECT),
  X4 heat load: McCune & Keon (2002) Table 2 Eq. 3, Rad = 0.339 + 0.808 cosL cosS - 0.196 sinL sinS
     - 0.482 cosA sinS, with A = |180 - |aspect - 225|| (their heat-load folding), angles in radians,
  X5 basal area m2/ha, X6 trees/ha, X7 / X8 TPA-weighted mean DIA (in) / HT (ft),
  X9 FORTYPCD, X10 STDSZCD, X11 SITECLCD, X12 PHYSCLCD (condition with the largest CONDPROP_UNADJ).

The rows removed by each inclusion rule (design section 2, applied in order) go to
results/cache/fia_fiadb95_build_log.json. Downloaded files are untrusted input: run with -I.

    ~/miniforge3/envs/favela/bin/python -I -B experiments/build_fia_fiadb.py ~/data/fiadb_9.5/zip
"""
import io, json, os, sys, zipfile
import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "data", "fia_fiadb95_plots.csv")
LOG = os.path.join(PROJ, "results", "cache", "fia_fiadb95_build_log.json")
STATES = ["AZ", "CA", "CO", "FL", "ME", "MO", "MT", "NM", "VT", "WA"]
LBAC = 0.0011209                                   # 1 lb/acre in Mg/ha
PLOT_COLS = ["CN", "STATECD", "UNITCD", "COUNTYCD", "PLOT", "INVYR", "MEASYEAR", "LAT", "LON", "ELEV",
             "PLOT_STATUS_CD", "KINDCD", "DESIGNCD", "QA_STATUS", "INTENSITY"]
COND_COLS = ["PLT_CN", "CONDID", "COND_STATUS_CD", "CONDPROP_UNADJ", "FORTYPCD", "STDSZCD", "SITECLCD",
             "PHYSCLCD", "SLOPE", "ASPECT"]
TREE_COLS = ["PLT_CN", "STATUSCD", "DIA", "HT", "TPA_UNADJ", "DRYBIO_AG", "CARBON_AG"]


def read(zdir, st, table, cols):
    with zipfile.ZipFile(os.path.join(zdir, f"{st}_{table}.zip")) as z:
        names = [n for n in z.namelist() if n.upper().endswith(".CSV")]
        assert len(names) == 1, names
        with z.open(names[0]) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), usecols=cols,
                               dtype={c: str for c in ("CN", "PLT_CN")}, low_memory=False)


def heat_load(lat, slope_pct, aspect):
    L = np.deg2rad(lat)
    S = np.arctan(np.asarray(slope_pct, float) / 100.0)
    A = np.deg2rad(np.abs(180.0 - np.abs(np.asarray(aspect, float) - 225.0)))
    return 0.339 + 0.808 * np.cos(L) * np.cos(S) - 0.196 * np.sin(L) * np.sin(S) - 0.482 * np.cos(A) * np.sin(S)


def elev_class(ft):
    m = np.asarray(ft, float) * 0.3048
    lab = np.array(["0-500m", "500-1000m", "1000-1500m", "1500-2000m", ">2000m"], dtype=object)
    out = lab[np.digitize(m, [500, 1000, 1500, 2000])]
    out[np.isnan(m)] = None
    return out


def slope_class(pct):
    s = np.asarray(pct, float)
    lab = np.array(["Flat", "Gentle", "Moderate", "Steep"], dtype=object)
    out = lab[np.digitize(s, [5, 20, 40])]
    out[np.isnan(s)] = None
    return out


def build_state(zdir, st, log):
    n = {}
    p = read(zdir, st, "PLOT", PLOT_COLS)
    n["0_all_plot_records"] = len(p)
    p = p[p.KINDCD.isin([1, 2, 3]) & (p.MEASYEAR >= 2000)]
    n["2_annual_2000plus"] = len(p)
    p = p[(p.QA_STATUS == 1) & (p.INTENSITY == 1) & (p.PLOT_STATUS_CD == 1)]
    n["3_standard_forested"] = len(p)
    p = p.assign(_cn=p.CN.astype("int64")).sort_values(["MEASYEAR", "INVYR", "_cn"])
    p = p.groupby(["STATECD", "UNITCD", "COUNTYCD", "PLOT"]).tail(1).drop(columns="_cn")
    n["4_latest_per_plot"] = len(p)

    c = read(zdir, st, "COND", COND_COLS)
    c = c[c.PLT_CN.isin(p.CN)]
    other = (c.COND_STATUS_CD != 1) & ~(c.CONDPROP_UNADJ <= 0)   # NaN proportion counts as > 0
    bad = set(c.PLT_CN[other]) | (set(p.CN) - set(c.PLT_CN))
    n["5_cond_prop_nan_records"] = int(c.CONDPROP_UNADJ.isna().sum())
    p = p[~p.CN.isin(bad)]
    n["5_all_forest_accessible"] = len(p)
    c = c[c.PLT_CN.isin(p.CN)].sort_values(["PLT_CN", "CONDPROP_UNADJ", "CONDID"], ascending=[True, False, True])
    c = c.groupby("PLT_CN").head(1).set_index("PLT_CN")

    t = read(zdir, st, "TREE", TREE_COLS)
    t = t[t.PLT_CN.isin(p.CN) & (t.STATUSCD == 1)]
    miss = t.DRYBIO_AG.isna() | t.TPA_UNADJ.isna()
    n["6_live_trees"] = len(t)
    n["6_live_trees_missing_biomass_or_tpa"] = int(miss.sum())
    p = p[~p.CN.isin(set(t.PLT_CN[miss]))]
    t = t[t.PLT_CN.isin(p.CN)]
    n["6_outcome_computable"] = len(p)

    w = t.TPA_UNADJ
    g = t.assign(bio=t.DRYBIO_AG * w, car=t.CARBON_AG * w, ba=0.005454 * t.DIA ** 2 * w,
                 wd=np.where(t.DIA.notna(), w, 0.0), dw=t.DIA.fillna(0) * w,
                 wh=np.where(t.HT.notna(), w, 0.0), hw=t.HT.fillna(0) * w).groupby("PLT_CN")
    s = g[["bio", "car", "ba", "wd", "dw", "wh", "hw"]].sum(min_count=1)
    s["tpa"] = g.TPA_UNADJ.sum()
    s = s.reindex(p.CN).fillna(0.0)                                  # plots without live trees -> 0
    n["6_plots_without_live_trees"] = int((s.tpa == 0).sum())
    cc = c.reindex(p.CN)

    out = pd.DataFrame({
        "region": st,
        "fips": p.STATECD.astype(int).map("{:02d}".format).values + p.COUNTYCD.astype(int).map("{:03d}".format).values,
        "lat": p.LAT.values, "lon": p.LON.values,
        "Y": LBAC * s.bio.values,
        "X1": elev_class(p.ELEV.values), "X2": slope_class(cc.SLOPE.values), "X3": cc.ASPECT.values,
        "X4": heat_load(p.LAT.values, cc.SLOPE.values, cc.ASPECT.values),
        "X5": 0.229568 * s.ba.values, "X6": 2.47105 * s.tpa.values,
        "X7": np.divide(s.dw.values, s.wd.values, out=np.zeros(len(s)), where=s.wd.values > 0),
        "X8": np.divide(s.hw.values, s.wh.values, out=np.zeros(len(s)), where=s.wh.values > 0),
        "X9": cc.FORTYPCD.values, "X10": cc.STDSZCD.values, "X11": cc.SITECLCD.values, "X12": cc.PHYSCLCD.values,
        "time": p.MEASYEAR.values.astype(int),
        "C": LBAC * s.car.values,
        "STATECD": p.STATECD.values, "UNITCD": p.UNITCD.values, "COUNTYCD": p.COUNTYCD.values,
        "PLOT": p.PLOT.values, "INVYR": p.INVYR.values, "DESIGNCD": p.DESIGNCD.values, "PLT_CN": p.CN.values})
    n["missing_covariate_rows"] = int(out[[f"X{i}" for i in range(1, 13)]].isna().any(axis=1).sum())
    n["slope_over_60deg"] = int((cc.SLOPE.values > 173.2).sum())
    n["lat_below_30"] = int((p.LAT.values < 30).sum())
    log[st] = n
    return out


def main():
    zdir = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/data/fiadb_9.5/zip")
    log = {}
    d = pd.concat([build_state(zdir, st, log) for st in STATES], ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False)
    log["total_plots"] = len(d)
    with open(LOG, "w") as f:
        json.dump(log, f, indent=1)
    print(pd.DataFrame({k: v for k, v in log.items() if k in STATES}).T.to_string())
    print(f"wrote {OUT} ({len(d)} plots)")


if __name__ == "__main__":
    main()
