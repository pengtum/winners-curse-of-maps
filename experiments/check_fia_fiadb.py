"""Checks V1 and V3 of paper/FIA_REBUILD_DESIGN.md section 4 on data/fia_fiadb95_plots.csv (no analysis is run).

V1  expansion factors: share of live trees on standard-design plots (DESIGNCD = 1) whose TPA_UNADJ is the
    subplot (6.018), microplot (74.965) or macroplot (0.999) value; target >= 99%, by state.
V3  descriptive match to the old table US_Forest_FIA.csv (latest record per location, as experiments/data.py):
    pairs by coordinates rounded to 5 decimals; among pairs with the same year, Spearman correlation of the old Y
    with the unexpanded sum of DRYBIO_AG (one tree per acre, the Methods diagnosis) and with the new Y, and the
    overlap of the top 1% and 5%.
Writes results/cache/fia_fiadb95_checks.json.

    ~/miniforge3/envs/favela/bin/python -I -B experiments/check_fia_fiadb.py ~/data/fiadb_9.5/zip
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_fia_fiadb as B                                  # noqa: E402

PROJ = B.PROJ
OLD = os.path.join(os.path.dirname(os.path.dirname(PROJ)), "0-Xiayin_research", "15_RobustGeoCP", "datasets",
                   "US_Forest_FIA.csv")
OUT = os.path.join(PROJ, "results", "cache", "fia_fiadb95_checks.json")
TPA = {"subplot": 6.018046, "microplot": 74.965282, "macroplot": 0.999188}


def top_overlap(a, b, f):
    k = max(1, int(round(f * len(a))))
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def main():
    zdir = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/data/fiadb_9.5/zip")
    d = pd.read_csv(B.OUT, dtype={"PLT_CN": str, "fips": str})
    res = {"V1": {}, "V3": {}}
    unexp = []
    for st in B.STATES:
        t = B.read(zdir, st, "TREE", ["PLT_CN", "STATUSCD", "TPA_UNADJ", "DRYBIO_AG"])
        t = t[t.PLT_CN.isin(d.PLT_CN[d.region == st]) & (t.STATUSCD == 1)]
        std = t[t.PLT_CN.isin(d.PLT_CN[(d.region == st) & (d.DESIGNCD == 1)])]
        shares = {k: float(np.isclose(std.TPA_UNADJ, v, atol=1e-3).mean()) if len(std) else None
                  for k, v in TPA.items()}
        shares["standard_total"] = float(sum(v for v in shares.values() if v is not None)) if len(std) else None
        shares["n_trees_design1"] = len(std)
        shares["n_plots_design1"] = int(((d.region == st) & (d.DESIGNCD == 1)).sum())
        res["V1"][st] = shares
        unexp.append(t.groupby("PLT_CN").DRYBIO_AG.sum())
    d["unexp"] = pd.concat(unexp).reindex(d.PLT_CN).fillna(0.0).values

    o = pd.read_csv(OLD, dtype={"fips": str})
    o["key"] = o.lat.round(5).astype(str) + "_" + o.lon.round(5).astype(str)
    o = o.sort_values("time").groupby("key").tail(1)
    d["key"] = d.lat.round(5).astype(str) + "_" + d.lon.round(5).astype(str)
    m = d.merge(o[["key", "Y", "time"]], on="key", suffixes=("", "_old"))
    same = m[m.time == m.time_old]
    res["V3"] = {"new_plots": len(d), "old_locations": len(o), "pairs_by_coordinates": len(m),
                 "pairs_same_year": len(same),
                 "spearman_oldY_unexpanded": float(spearmanr(same.Y_old, same.unexp)[0]) if len(same) > 2 else None,
                 "spearman_oldY_newY": float(spearmanr(same.Y_old, same.Y)[0]) if len(same) > 2 else None,
                 "top1_overlap": top_overlap(same.Y_old.values, same.Y.values, 0.01) if len(same) else None,
                 "top5_overlap": top_overlap(same.Y_old.values, same.Y.values, 0.05) if len(same) else None,
                 "pairs_by_state": same.region.value_counts().to_dict()}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(pd.DataFrame(res["V1"]).T.round(4).to_string())
    print(json.dumps(res["V3"], indent=1))


if __name__ == "__main__":
    main()
