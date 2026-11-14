"""ESA CCI Biomass v7.0 (2020, 100 m) at the FIA plots of data/fia_fiadb95_plots.csv (paper/BIOMASS_PRODUCT_DESIGN.md
section 3, frozen 2026-10-31).

For each plot, the window is every 100 m pixel whose centre lies within 1 km of the public coordinates (equirectangular
distance at the plot's latitude; windows may span tiles). Main: agb = mean AGB, sd = mean AGB_SD over the window.
Sensitivity: agb_ctr, sd_ctr of the pixel containing the coordinates. The product codes non-vegetated land and water as 0
and has no missing-value code, so a plot is dropped only if every pixel of its window is 0 (counted in the log).
Writes data/fia_cci_v7_2020.csv (PLT_CN, region, lat, lon, Y, C, time, agb, sd, agb_ctr, sd_ctr, n_pix, zero_share) and
results/cache/fia_cci_v7_2020_build_log.json. Reads the downloaded tiles, hence python -I:
    ~/miniforge3/envs/favela/bin/python -I experiments/build_fia_cci.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILES = os.path.expanduser("~/data/esacci_biomass_v7/2020")
FMT = "{t}_ESACCI-BIOMASS-L4-{v}-MERGED-100m-2020-fv7.0.tif"
RADIUS_KM = 1.0
KM_PER_DEG = 111.32


def tile_name(lat, lon):
    """Tiles are 10 x 10 degrees named by their upper-left corner (checked on N30W090: 20-30 N, 90-80 W)."""
    top, left = int(np.ceil(lat / 10) * 10), int(np.floor(lon / 10) * 10)
    return f"{'N' if top >= 0 else 'S'}{abs(top):02d}{'W' if left < 0 else 'E'}{abs(left):03d}"


class Tiles:
    def __init__(self):
        self.open = {}

    def get(self, t, v):
        k = (t, v)
        if k not in self.open:
            self.open[k] = rasterio.open(os.path.join(TILES, FMT.format(t=t, v=v)))
        return self.open[k]

    def window(self, lat, lon):
        """AGB and SD of every pixel centred within RADIUS_KM, plus the centre pixel."""
        dlat = RADIUS_KM / KM_PER_DEG; dlon = dlat / np.cos(np.radians(lat))
        corners = {tile_name(a, b) for a in (lat - dlat, lat + dlat) for b in (lon - dlon, lon + dlon)}
        agb, sd = [], []
        for t in corners:
            ra, rs = self.get(t, "AGB"), self.get(t, "AGB_SD")
            b = ra.bounds
            w = from_bounds(max(lon - dlon, b.left), max(lat - dlat, b.bottom), min(lon + dlon, b.right),
                            min(lat + dlat, b.top), ra.transform).round_offsets().round_lengths()
            a = ra.read(1, window=w).astype(float); s = rs.read(1, window=w).astype(float)
            rows, cols = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            x, y = rasterio.transform.xy(ra.window_transform(w), rows.ravel(), cols.ravel())
            d = np.hypot((np.asarray(x) - lon) * np.cos(np.radians(lat)), np.asarray(y) - lat) * KM_PER_DEG
            m = d <= RADIUS_KM
            agb.append(a.ravel()[m]); sd.append(s.ravel()[m])
        t = tile_name(lat, lon); ra, rs = self.get(t, "AGB"), self.get(t, "AGB_SD")
        r, c = ra.index(lon, lat)
        ctr = (float(ra.read(1, window=((r, r + 1), (c, c + 1)))[0, 0]),
               float(rs.read(1, window=((r, r + 1), (c, c + 1)))[0, 0]))
        return np.concatenate(agb), np.concatenate(sd), ctr


def main():
    z = pd.read_csv(os.path.join(ROOT, "data", "fia_fiadb95_plots.csv"),
                    usecols=["PLT_CN", "region", "lat", "lon", "Y", "C", "time"])
    T = Tiles(); rows = []
    for k, p in enumerate(z.itertuples(index=False)):
        a, s, (ac, sc) = T.window(p.lat, p.lon)
        rows.append(dict(agb=a.mean() if a.size else np.nan, sd=s.mean() if s.size else np.nan, agb_ctr=ac,
                         sd_ctr=sc, n_pix=a.size, zero_share=(a == 0).mean() if a.size else np.nan))
        if k % 5000 == 0:
            print(k, flush=True)
    out = pd.concat([z, pd.DataFrame(rows)], axis=1)
    drop = (out.n_pix == 0) | (out.zero_share == 1)
    log = dict(plots_in=len(out), dropped_all_zero_or_empty=int(drop.sum()),
               dropped_by_state=out[drop].region.value_counts().to_dict(), plots_out=int((~drop).sum()),
               n_pix_median=float(out.n_pix.median()), radius_km=RADIUS_KM,
               tiles=sorted(os.path.basename(f) for f in os.listdir(TILES) if f.endswith(".tif")))
    out = out[~drop]
    out.to_csv(os.path.join(ROOT, "data", "fia_cci_v7_2020.csv"), index=False, float_format="%.6g")
    json.dump(log, open(os.path.join(ROOT, "results", "cache", "fia_cci_v7_2020_build_log.json"), "w"), indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
