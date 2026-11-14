"""Datasets for QW2 (read-only sources under 0-Xiayin_research; nothing is downloaded).

Each loader returns dict(name, coords [km], Xrf [model features incl. lon/lat], y, state
[quota group], N). FIA is reduced to one record per plot location (latest inventory) -- the
pilot sampled raw rows, so repeat visits of a plot could fall on both sides of a split.
FIA, ERA5 and PovertyMap are subsampled to NMAX units with a fixed seed. Cached under results/cache/.
PovertyMap (WILDS v1.1, Yeh et al. 2020): ~/data/wilds holds dhs_metadata.csv and image summary
features streamed by stream_povertymap.py; the cluster table is copied to data/povertymap_clusters.csv.
DHS-derived labels: research use only, do not redistribute (see STATUS.md).

PLACES health outcomes are MRP model outputs, not measured truth (see CLAUDE.md).
"""
import os
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                   "0-Xiayin_research")                 # Research/0-Xiayin_research, found from the project location
DS = os.path.join(RES, "15_RobustGeoCP", "datasets")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cache")
NMAX = 10000
WILDS = os.path.expanduser("~/data/wilds")
# append-only: run_main.py seeds each dataset by its index in ALL
ALL = ["fia", "era5", "camels", "voting", "seattle",
       "health_stroke", "health_diabetes", "health_bphigh", "health_obesity", "poverty", "poverty_lab",
       "fia_post2000",                                  # sensitivity only; not in DISPLAY or the figures
       "fia_db",                                        # FIADB 9.5 rebuild (paper/FIA_REBUILD_DESIGN.md), seed index 12
       # formal DOFA arm (paper/DOFA_FORMAL_DESIGN.md), seed indices 13-16: frozen DOFA features + linear probe;
       # loaded only with QW2_MAP=probe, which only they use
       "dofa_poverty", "dofa_m3leo", "dofa_floods", "dofa_usavars"]
GROUND_8DS = os.path.join(RES, "15_RobustGeoCP", "real_world_study", "results_8ds")   # GROUND score files, read-only
DOFA = {"dofa_poverty": "poverty_dofa", "dofa_m3leo": "m3leo_dofa", "dofa_floods": "urbansarfloods_dofa",
        "dofa_usavars": "usavars_dofa"}
# the FIA table the paper reports: fia_db (rebuilt, Mg/ha) unless QW2_FIA=fia reproduces the old-table outputs.
# FIA_RES holds the FIA outputs whose file names do not carry the dataset name (carbon, OOD, decisions).
FIA = os.environ.get("QW2_FIA", "fia_db")
FIA_RES = os.path.join(os.path.dirname(CACHE), "" if FIA == "fia" else FIA)
FIA_TAG = "" if FIA == "fia" else f"_{FIA}"             # suffix of the FIA part caches under CACHE
DISPLAY = ["poverty", "poverty_lab", FIA] + ALL[1:9]
LABEL = {"poverty": "DHS wealth (PovertyMap)", "poverty_lab": "DHS wealth, no imagery", "fia": "FIA biomass", "era5": "ERA5 t2m", "camels": "CAMELS runoff", "voting": "Vote share",
         "seattle": "Seattle price", "health_stroke": "Stroke*", "health_diabetes": "Diabetes*",
         "health_bphigh": "High BP*", "health_obesity": "Obesity*", "fia_post2000": "FIA biomass, plots inventoried 2000 or later",
         "fia_db": "FIA biomass", "dofa_poverty": "DHS wealth, 13 countries (DOFA)", "dofa_m3leo": "Biomass† (M3LEO, DOFA)",
         "dofa_floods": "Flood fraction (UrbanSARFloods, DOFA)", "dofa_usavars": "Tree cover† (USAVars, DOFA)"}
# policy-relevant selection direction (top = fund / certify the highest predicted values)
POLICY = {"poverty": "bottom", "poverty_lab": "bottom", "fia": "top", "era5": "bottom", "camels": "top", "voting": "top", "seattle": "top",
          "health_stroke": "top", "health_diabetes": "top", "health_bphigh": "top", "health_obesity": "top",
          "fia_post2000": "top", "fia_db": "top",
          "dofa_poverty": "bottom", "dofa_m3leo": "top", "dofa_floods": "top", "dofa_usavars": "bottom"}


def to_km(lat, lon):
    lat0 = np.deg2rad(np.mean(lat))
    return np.c_[np.asarray(lon) * 111.32 * np.cos(lat0), np.asarray(lat) * 110.57]


def ecef_km(lat, lon, R=6371.0):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.c_[R * np.cos(la) * np.cos(lo), R * np.cos(la) * np.sin(lo), R * np.sin(la)]


def _dofa(name):
    """Frozen DOFA features for the formal DOFA arm (paper/DOFA_FORMAL_DESIGN.md): `embed` is the PCA-32 of the frozen
    backbone's output before GROUND's linear head, computed on all samples without labels; GROUND's head and y_pred are not
    used, and all samples enter (GROUND's in_train included). Xrf = [lon, lat, 32 embedding columns]; the map is a linear
    probe on the embedding (run_main.rf with QW2_MAP=probe). Coordinates are Earth-centred km, as in the exploratory arm."""
    z = np.load(os.path.join(GROUND_8DS, DOFA[name] + "_scores.npz"), allow_pickle=False)
    lat, lon = z["lat"].astype(float), z["lon"].astype(float)
    if name == "dofa_poverty":                        # countries live in the ResNet file of the same samples
        zr = np.load(os.path.join(GROUND_8DS, "poverty_scores.npz"), allow_pickle=False)
        assert np.allclose(zr["lat"], lat) and np.allclose(zr["lon"], lon)
        group = zr["region_country"]
    else:
        group = z["region"]
    return dict(coords=ecef_km(lat, lon), Xrf=np.c_[lon, lat, z["embed"].astype(float)], y=z["y"].astype(float),
                state=pd.factorize(group)[0])


def _build(name):
    if name in DOFA:
        return _dofa(name)
    if name == "poverty_lab":
        # label-table-only variant: covariates of the cluster table, no image features
        m = pd.read_csv(os.path.join(WILDS, "dhs_metadata.csv"))
        X = np.c_[m.urban.astype(float), m.year, m.nl_mean, m.nl_center]
        return dict(coords=to_km(m.lat.values, m.lon.values), Xrf=np.c_[m.lon, m.lat, X].astype(float),
                    y=m.wealthpooled.values.astype(float), state=pd.factorize(m.country)[0])
    if name == "poverty":
        m = pd.read_csv(os.path.join(WILDS, "dhs_metadata.csv"))
        f = pd.read_csv(os.path.join(WILDS, "poverty_features.csv")).sort_values("idx").reset_index(drop=True)
        assert (f.idx.values == np.arange(len(m))).all()       # landsat_poverty_img_<i> = metadata row i
        d = pd.concat([m, f.drop(columns="idx")], axis=1)
        out_csv = os.path.join(os.path.dirname(CACHE), "..", "data", "povertymap_clusters.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        d.to_csv(out_csv, index=False)
        img = [c for c in f.columns if c != "idx"]
        X = np.c_[d.urban.astype(float), d.year, d.nl_mean, d.nl_center, d[img].values]
        return dict(coords=to_km(d.lat.values, d.lon.values), Xrf=np.c_[d.lon, d.lat, X].astype(float),
                    y=d.wealthpooled.values.astype(float), state=pd.factorize(d.country)[0])
    if name in ("fia", "fia_post2000", "fia_db"):
        if name == "fia_db":
            # one row per plot already (experiments/build_fia_fiadb.py); Y in Mg/ha, same X1..X12 layout
            d = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(CACHE)), "data", "fia_fiadb95_plots.csv"),
                            dtype={"fips": str})
        else:
            d = pd.read_csv(os.path.join(DS, "US_Forest_FIA.csv"), dtype={"fips": str})
            d["key"] = d.lat.round(5).astype(str) + "_" + d.lon.round(5).astype(str)
            d = d.sort_values("time").groupby("key").tail(1).reset_index(drop=True)
        if name == "fia_post2000":
            # sensitivity: of the 5,356 plots last inventoried before 2000, the 5,355 in Florida have basal area X5 = 0
            d = d[d.time >= 2000].reset_index(drop=True)
        cols = [f"X{i}" for i in range(1, 13)]
        X = d[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype("category").cat.codes
        d = d.assign(**{c: X[c] for c in cols}).dropna(subset=cols + ["Y", "lat", "lon"])
        X = d[cols].values.astype(float); state = d.region.values
    elif name == "era5":
        d = pd.read_csv(os.path.join(DS, "US_Climate_ERA5_CLIMATE.csv")).dropna()
        X = d[[f"X{i}" for i in range(1, 10)]].values; state = d.region.values
    elif name.startswith("health_"):
        d = pd.read_csv(os.path.join(DS, f"US_Health_{name.split('_')[1].upper()}.csv")).dropna()
        X = d[[f"X{i}" for i in range(1, 9)]].values; state = d.region.values
    elif name == "voting":
        d = pd.read_csv(os.path.join(DS, "US_Politics_Voting.csv")).dropna()
        X = d[[f"X{i}" for i in range(1, 15)]].values; state = d.region.values
        c = np.c_[d.proj_x, d.proj_y] / 1000.0
        return dict(coords=c, Xrf=np.c_[c, X].astype(float), y=d.Y.values.astype(float),
                    state=pd.factorize(state)[0])
    elif name == "camels":
        d = pd.read_csv(os.path.join(DS, "US_Hydro_CAMELS.csv")).dropna()
        X = d[[f"X{i}" for i in range(1, 11)]].values; state = d.region.values
    elif name == "seattle":
        d = pd.read_csv(os.path.join(DS, "seattle_sample_3k.csv")).dropna()
        cols = ["bathrooms", "sqft_living", "sqft_lot", "grade", "condition", "waterfront", "view", "age"]
        c = np.c_[d.UTM_X, d.UTM_Y] / 1000.0
        from sklearn.cluster import KMeans
        zones = KMeans(6, n_init=4, random_state=0).fit_predict(c)
        return dict(coords=c, Xrf=np.c_[c, d[cols].values].astype(float), y=d.log_price.values.astype(float),
                    state=zones)
    else:
        raise ValueError(name)
    return dict(coords=to_km(d.lat.values, d.lon.values), Xrf=np.c_[d.lon, d.lat, X].astype(float),
                y=d.Y.values.astype(float), state=pd.factorize(state)[0])


def load(name):
    if (name in DOFA) != (os.environ.get("QW2_MAP") == "probe"):
        raise SystemExit(f"{name}: the DOFA settings run with QW2_MAP=probe, and only they do (paper/DOFA_FORMAL_DESIGN.md)")
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(f):
        d = dict(np.load(f))
    else:
        d = _build(name)
        N = len(d["y"])
        if N > NMAX:
            idx = np.sort(np.random.default_rng(0).choice(N, NMAX, replace=False))
            d = {k: v[idx] for k, v in d.items()}
        np.savez_compressed(f, **d)
    d["name"] = name; d["N"] = len(d["y"])
    return d


if __name__ == "__main__":
    for nm in [n for n in ALL if (n in DOFA) == (os.environ.get("QW2_MAP") == "probe")]:
        d = load(nm)
        print(nm, d["N"], d["Xrf"].shape, "states", len(np.unique(d["state"])),
              "y range", np.round([d["y"].min(), np.median(d["y"]), d["y"].max()], 2))
