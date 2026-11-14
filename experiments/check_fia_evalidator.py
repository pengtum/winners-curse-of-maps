"""Check V2 of paper/FIA_REBUILD_DESIGN.md section 4: state means of data/fia_fiadb95_plots.csv against EVALIDator.

For each state's most recent current-area evaluation (wc below, EVALIDator 'MOST_RECENT' = Y on 2026-10-30), the
official ratio estimate of attribute 10 (aboveground biomass of live trees >= 1 inch d.b.h./d.r.c., dry short
tons, on forest land) over attribute 2 (area of forest land, acres), converted to Mg/ha (x 2000 lb x 0.0011209),
is compared with the unweighted mean Y of the table's plots measured in that evaluation's years. Tolerance +-20%.
Queries the FIADB-API (no table is downloaded); the raw responses are kept in results/cache/fia_evalidator/.

    ~/miniforge3/envs/favela/bin/python -I -B experiments/check_fia_evalidator.py
"""
import json, os, urllib.request
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJ, "results", "cache", "fia_evalidator")
TABLE = os.path.join(PROJ, "data", "fia_fiadb95_plots.csv")
OUT = os.path.join(PROJ, "results", "cache", "fia_fiadb95_check_v2.csv")
URL = ("https://apps.fs.usda.gov/fiadb-api/fullreport?snum=10&sdenom=2&wc={wc}"
       "&rselected=None&cselected=None&outputFormat=NJSON")
WC = {"AZ": (42023, range(2014, 2024)), "CA": (62024, range(2014, 2025)), "CO": (82023, range(2014, 2024)),
      "FL": (122025, range(2016, 2026)), "ME": (232025, range(2021, 2026)), "MO": (292025, range(2019, 2026)),
      "MT": (302022, range(2013, 2023)), "NM": (352022, range(2015, 2023)), "VT": (502025, range(2019, 2026)),
      "WA": (532023, range(2014, 2024))}
TONAC = 2000 * 0.0011209                          # dry short tons per acre -> Mg/ha


def main():
    os.makedirs(RAW, exist_ok=True)
    d = pd.read_csv(TABLE)
    rows = []
    for st, (wc, yrs) in WC.items():
        f = os.path.join(RAW, f"{st}_{wc}.json")
        if not os.path.exists(f):
            with urllib.request.urlopen(URL.format(wc=wc), timeout=180) as r, open(f, "wb") as g:
                g.write(r.read())
        j = json.load(open(f))
        e = j["estimates"][0]
        sub = d[(d.region == st) & d.time.isin(list(yrs))]
        off = e["RATIO_ESTIMATE"] * TONAC
        rows.append(dict(state=st, wc=wc, db=j["metadata"]["dbVersion"], official_Mg_ha=off,
                         official_se=e["RATIO_SE"] * TONAC, n_official_plots=e["DENOMINATOR_PLOT_COUNT"],
                         table_mean=sub.Y.mean(), n_table=len(sub), rel_diff=sub.Y.mean() / off - 1))
    r = pd.DataFrame(rows)
    r["within_20pct"] = r.rel_diff.abs() <= 0.20
    r.to_csv(OUT, index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
