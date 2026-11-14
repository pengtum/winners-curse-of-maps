"""Fig. 3 | Study design: where the data are, how every map is evaluated, and which data enter which experiment
(paper/FIGURE_PLAN.md).

a  footprints on a world map (Natural Earth 1:110m countries, equirectangular): Seattle (city scale), the United
   States (country scale, six data sources; the 10 FIA states outlined, Natural Earth 1:110m states), the 23 DHS countries of Africa (continent scale; the 13 used in the
   DOFA and ResNet settings darker), and the M3LEO biomass regions and UrbanSARFloods event areas (global scale). Countries, region
   centroids and event locations only; DHS cluster coordinates only decide which countries the 13-country subset covers.
b  evaluation protocol: 100 random splits of every map (RF and DOFA settings: train / calibration / test = 40 / 30 / 30;
   exploratory ResNet settings: the units outside the backbone's own training fold, structure / calibration / test =
   25 / 37.5 / 37.5), the selection rules applied to the test units, and the metrics.
c  data sources (rows, by scale) x maps and experiments (columns). A dot means the result files exist
   (results/main_*, bl_main_*, sf_main_*, fm_main_*, scg_main_*_l1fix, scg_slack_*, dec_*, cfbh_*, pilot_l3_*,
   carbon_*), so the grid is read from the results, not typed in. 2026-11-08: the DOFA settings became formal
   (paper/DOFA_FORMAL_DESIGN.md; maps refitted in every split, results/main_dofa_*), so DOFA and ResNet have
   separate columns and protocols; the earlier exploratory DOFA files (fm_main_*_dofa) are no longer read.
    FIGURE_QA_HELPERS=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python -I figures/make_fig3.py
writes figs/nc_fig3_design.{pdf,svg,png}, figs/source_data/nc_fig3{a,c}.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402
import basemap as BM                                                  # noqa: E402

W, H = 183, 137
# scale, data source, RF settings, DOFA settings, ResNet combos, units, groups (quota units), ground truth, decision
SOURCES = [
    ("Metropolitan area", "Seattle house price", ["seattle"], [], [], "3,000 sales", "6 k-means zones", "Measured",
     "Housing"),
    ("Country (USA)", "FIA forest biomass", [D.DS.FIA], [], [], {"fia": "10,000 of 28,424 plots", "fia_db": "10,000 of 27,059 plots"}[D.DS.FIA],
     {"fia": "10 states (9 quotas)", "fia_db": "10 states"}[D.DS.FIA], "Measured",           # rebuilt table: every state has a quota
     "Carbon credits"),
    ("Country (USA)", "ERA5 temperature", ["era5"], [], [], "10,000 of 83,731 cells", "9 NOAA regions", "Reanalysis",
     "Climate extremes"),
    ("Country (USA)", "CAMELS runoff", ["camels"], [], [], "670 basins", "18 regions", "Measured", "Water management"),
    ("Country (USA)", "County vote share", ["voting"], [], [], "3,108 counties", "48 states + DC", "Measured",
     "Elections"),
    ("Country (USA)", "CDC PLACES, 4 outcomes*", ["health_stroke", "health_diabetes", "health_bphigh",
                                                  "health_obesity"], [], [], "2,350 counties", "38 states + DC",
     "Model output", "Health funding"),
    ("Country (USA)", "USAVars tree cover†", [], ["dofa_usavars"], [], "10,000 of 11,996 tiles", "9 divisions + other",
     "Satellite product", "Tree planting"),
    ("Continent (Africa)", "DHS wealth", ["poverty", "poverty_lab"], ["dofa_poverty"], ["poverty_resnet"],
     "10,000; DOFA 7,535", "23 countries (DOFA 13)", "Survey", "Aid targeting"),
    ("Global", "M3LEO biomass†", [], ["dofa_m3leo"], ["m3leo_resnet"], "7,156 tiles", "6 regions",
     "Satellite product", "Carbon credits"),
    ("Global", "UrbanSARFloods", [], ["dofa_floods"], ["floods_resnet"], "6,471 tiles", "16 event areas",
     "Reference masks", "Flood relief"),
]
MAPS = [("RF", lambda rf, df, rn: bool(rf)), ("DOFA", lambda rf, df, rn: bool(df)),
        ("ResNet", lambda rf, df, rn: bool(rn))]
# experiment code, column header, result files of one data source (RF settings rf, DOFA settings df, ResNet combos rn)
EXPERIMENTS = [
    ("E1", "Coverage audit", lambda rf, df, rn: [f"main_{n}.csv" for n in rf + df]),
    ("E1b", "Published baselines", lambda rf, df, rn: [f"bl_main_{n}.csv" for n in rf + df]),
    ("E1c", "Score functions", lambda rf, df, rn: [f"sf_main_{n}.csv" for n in rf]),
    ("E2", "ResNet (SI)", lambda rf, df, rn: [f"fm_main_{c}.csv" for c in rn]),
    ("E3", "RegionCP", lambda rf, df, rn: [f"scg_main_{n}_l1fix.csv" for n in rf + df]),
    ("E3b", "Guarantee floors", lambda rf, df, rn: [f"scg_slack_{n}.csv" for n in rf]),
    ("E4", "Decision outcomes", lambda rf, df, rn: [f"dec_{n}.csv" for n in rf]),    # deep decision runs not reported
    ("E5", "Funding (cfBH)", lambda rf, df, rn: [f"cfbh_{n}.csv" for n in rf + df]),
    ("E6", "Certified totals", lambda rf, df, rn: [f"pilot_l3x_{n}.csv" for n in rf]),   # exact envelope
    ("E7", "Carbon protocols", lambda rf, df, rn: ["carbon_site.csv.gz"] if D.DS.FIA in rf else []),
]
NOTES = ["Dot: the experiment's results include this source (read from the result files).",
         "Units: map units used; sources above 10,000 units are subsampled to 10,000; the exploratory",
         "ResNet settings (SI Appendix, Fig. S2) use the units outside the backbone's training fold.",
         "Groups: the units of the quota rule and of the worst-group metric.",
         "* Model output (CDC small-area estimates), not a measurement.  † Satellite-derived product."]
SEATTLE = (-122.33, 47.61)                     # the King County sales (UTM zone 10N in the data)
FIA_STATES = ("AZ", "CA", "CO", "FL", "ME", "MO", "MT", "NM", "VT", "WA")   # the states of the FIA table
LON, LAT = (-170.0, 180.0), (-57.0, 78.0)
LAND, FOOT, FOOT2 = "#ecebe6", S.GREY_L, S.GREY_M
# layout (mm)
MAP = (2.0, 86.0, 118.0, 118.0 * (LAT[1] - LAT[0]) / (LON[1] - LON[0]))
PROT = (126.0, 86.0, 55.0, MAP[3])
GRID_H = 80.0
X_TXT = dict(name=3.0, units=37.0, groups=60.0, truth=84.0, decision=103.0)
X_BB, X_EX = [124.5, 130.0, 135.5], [142.5 + 4.25 * i for i in range(len(EXPERIMENTS))]   # map columns, experiments


# ------------------------------------------------------------------ data
def footprints():
    """Countries, region centroids and event locations of panel a (one row each)."""
    dhs = sorted({iso for iso, _ in D.DHS.values()})
    world = BM.countries()
    g = D.ground("poverty_scores.npz")                       # GeoFM subset: the country holding most clusters
    where = BM.locate({k: world[k] for k in dhs}, np.c_[g["lon"], g["lat"]])
    sub = sorted({pd.Series(where[g["region_country"] == c]).mode()[0] for c in np.unique(g["region_country"])})
    rows = [dict(scale="Continent", data="DHS wealth", kind="country", name=k, geofm=k in sub) for k in dhs]
    rows.append(dict(scale="Country", data="Six US data sources", kind="country", name="USA", geofm=False))
    rows.append(dict(scale="City", data="Seattle house price", kind="city", name="Seattle", lon=SEATTLE[0],
                     lat=SEATTLE[1]))
    for f, data, kind in (("m3leo_scores.npz", "M3LEO biomass", "region"),
                          ("urbansarfloods_scores.npz", "UrbanSARFloods", "event")):
        z = D.ground(f)
        for k, nm in enumerate(z["region_names"]):
            m = z["region"] == k
            rows.append(dict(scale="Global", data=data, kind=kind, name=str(nm), lon=float(z["lon"][m].mean()),
                             lat=float(z["lat"][m].mean()), units=int(m.sum())))
    return pd.DataFrame(rows), world


def grid():
    """One row per data source: the text columns, and True / False per map and experiment."""
    ex = lambda f: os.path.exists(os.path.join(S.RES, f))      # noqa: E731
    rows = []
    for sc, name, rf, df, rn, units, groups, truth, dec in SOURCES:
        r = dict(scale=sc, data=name, units=units, groups=groups, ground_truth=truth, decision=dec)
        r.update({b: f(rf, df, rn) for b, f in MAPS})
        r.update({f"{c} {h}": any(ex(x) for x in files(rf, df, rn)) for c, h, files in EXPERIMENTS})
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ a
def panel_a(fig, F, world):
    ax = S.axes_mm(fig, *MAP)
    box = (LON[0], LON[1], LAT[0], LAT[1])
    ax.set_xlim(*LON); ax.set_ylim(*LAT); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    usa = {"USA": [p for p in world["USA"] if p[0][:, 1].mean() > 24]}    # drop Hawaii (not in PLACES)
    c = F[F.kind == "country"]
    fills = {k: LAND for k in world}
    fills.update({k: FOOT2 if g else FOOT for k, g in zip(c["name"], c["geofm"])})
    BM.fill(ax, dict(world, **usa), lambda r: r, fills, box=box)
    BM.outline(ax, dict(world, **usa), lambda r: r, coast=BM.COAST, coast_lw=0.3, lw=0.25,
               borders_of=set(c["name"]), box=box)
    fia = BM.features(BM.STATES, "postal", keep=lambda p: p["postal"] in FIA_STATES)
    BM.outline(ax, fia, lambda r: r, color=S.INK2, lw=0.3, coast=S.INK2, coast_lw=0.3, box=box, z=4)
    for kind, mk, s, hollow in (("region", "^", 16, True), ("event", "o", 7, False), ("city", "s", 9, False)):
        x = F[F.kind == kind]
        ax.scatter(x.lon, x.lat, s=s, marker=mk, facecolor="white" if hollow else S.INK, edgecolor=S.INK,
                   linewidths=0.6, zorder=5)
    hs = [Patch(facecolor=FOOT, edgecolor="none", label="Study area"),
          Patch(facecolor=FOOT, edgecolor=S.INK2, linewidth=0.3, label="FIA plots (10 states)"),
          Patch(facecolor=FOOT2, edgecolor="none", label="DHS, DOFA and ResNet (13)"),
          Line2D([], [], ls="none", marker="s", ms=2.8, mfc=S.INK, mec=S.INK, label="Seattle area (metropolitan)"),
          Line2D([], [], ls="none", marker="^", ms=3.4, mfc="white", mec=S.INK, mew=0.6, label="M3LEO regions (global)"),
          Line2D([], [], ls="none", marker="o", ms=2.4, mfc=S.INK, mec=S.INK, label="Flood event areas (global)")]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 0.0), borderaxespad=0.3, handlelength=1.0,
              handleheight=0.8, handletextpad=0.4, labelspacing=0.25)
    return ax


# ------------------------------------------------------------------ b
def panel_b(fig):
    ax = S.axes_mm(fig, *PROT); w, h = PROT[2], PROT[3]
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    seg = {"Train": S.GREY_M, "Structure": S.GREY_M, "Calibration": S.GREY_L, "Test": S.SHADE}
    note = dict(fontsize=S.FS["note"], ha="left", va="baseline", color=S.INK)
    y = h - 2.2
    for lab, parts in (("RF (11) and DOFA (4) settings: 100 random splits", (("Train", 40), ("Calibration", 30),
                                                                             ("Test", 30))),
                       ("ResNet settings (3; exploratory, SI Appendix, Fig. S2)", (("Structure", 25),
                        ("Calibration", 37.5), ("Test", 37.5)))):
        ax.text(0, y, lab, **note); x0 = 0.0
        for nm, p in parts:
            dx = w * p / 100
            ax.add_patch(Rectangle((x0, y - 7.0), dx, 4.2, facecolor=seg[nm], edgecolor="white", linewidth=0.6))
            ax.text(x0 + dx / 2, y - 4.9, f"{nm} {p:g}%", fontsize=S.FS["note"], ha="center", va="center",
                    color="white" if seg[nm] == S.GREY_M else S.INK)
            x0 += dx
        y -= 10.5
    for head, lines in (("Selection rules on the test units", ["Top or bottom K (1, 5, 20%); quota in every group",
                                                                "Budget and publication masks: RF and DOFA"]),
                        ("Metrics among the selected units", ["Coverage, overall and in the worst group",
                                                               "Interval width; share of +∞ intervals",
                                                               "Decision outcomes; certified totals"])):
        ax.text(0, y, head, fontsize=S.FS["note"], ha="left", va="baseline", color=S.INK, fontweight="bold")
        for ln in lines:
            y -= 2.7; ax.text(1.2, y, ln, fontsize=S.FS["note"], ha="left", va="baseline", color=S.INK2)
        y -= 4.0
    return ax


# ------------------------------------------------------------------ c
def panel_c(fig, G):
    ax = S.axes_mm(fig, 0.0, 0.0, W, GRID_H)
    ax.set_xlim(0, W); ax.set_ylim(0, GRID_H); ax.axis("off")
    yh = 60.5                                          # baseline of the column headers
    head = dict(fontsize=S.FS["note"], color=S.INK, fontweight="bold", va="baseline")
    for k, lab in (("name", "Data source"), ("units", "Units"), ("groups", "Groups (quota)"),
                   ("truth", "Ground truth"), ("decision", "Decision")):
        ax.text(X_TXT[k], yh - 2.4, lab, ha="left", **head)
    for x, (b, _) in zip(X_BB, MAPS):
        ax.text(x, yh, b, rotation=90, rotation_mode="anchor", ha="left", va="center", fontsize=S.FS["note"], color=S.INK)
    for x, (c, h, _) in zip(X_EX, EXPERIMENTS):
        ax.text(x, yh, h, rotation=90, rotation_mode="anchor", ha="left", va="center", fontsize=S.FS["note"], color=S.INK)
    for i, ln in enumerate(NOTES):                   # footnotes in the space left of the rotated headers
        ax.text(X_TXT["name"], GRID_H - 4.0 - 2.7 * i, ln, fontsize=S.FS["note"], color=S.INK2, ha="left",
                va="baseline")
    y, last = yh - 4.4, None
    cols = [b for b, _ in MAPS] + [f"{c} {h}" for c, h, _ in EXPERIMENTS]
    for _, r in G.iterrows():
        if r["scale"] != last:                         # scale header row
            ax.add_patch(Rectangle((1.5, y - 3.3), W - 3.0, 3.3, facecolor=S.SHADE, edgecolor="none", zorder=0))
            ax.text(X_TXT["name"], y - 1.65, r["scale"], fontsize=S.FS["note"], color=S.INK2, fontstyle="italic",
                    ha="left", va="center")
            y -= 3.3; last = r["scale"]
        yc = y - 2.0
        ax.text(X_TXT["name"], yc, r["data"], fontsize=S.FS["label"], color=S.INK, ha="left", va="center")
        for k in ("units", "groups", "truth", "decision"):
            ax.text(X_TXT[k], yc, r["ground_truth" if k == "truth" else k], fontsize=S.FS["note"], color=S.INK2,
                    ha="left", va="center")
        on = r[cols].astype(bool).values
        xs = np.array(X_BB + X_EX)
        ax.scatter(xs[on], [yc] * on.sum(), s=9, color=S.INK2, linewidths=0, zorder=3)
        ax.scatter(xs[~on], [yc] * (~on).sum(), s=1.2, color=S.GRID, linewidths=0, zorder=2)
        y -= 4.0
    return ax


def main():
    S.apply()
    F, world = footprints(); G = grid()
    fig = S.figure(W, H)
    A = panel_a(fig, F, world); B = panel_b(fig); C = panel_c(fig, G)
    for x, y, l, t in ((2.0, H - 1.0, "a", "Data footprints at four scales"),
                       (PROT[0] - 4.5, H - 1.0, "b", "Evaluation protocol"),
                       (2.0, GRID_H + 3.5, "c", "Data sources, maps and experiments")):
        S.letter_mm(fig, x, y, l)
        S.text_mm(fig, x + 4.5, y - 0.35, t, fontsize=S.FS["title"], fontweight="bold", ha="left", va="top")
    S.source_data("nc_fig3a", F); S.source_data("nc_fig3c", G)
    S.save(fig, "nc_fig3_design", axes=[A, B, C], panel_ids=["a", "b", "c"], row_groups=[["a", "b"]],
           column_groups=[])
    print(F.groupby(["scale", "kind"]).size().to_string())
    print("13-country DHS subset (DOFA, ResNet):", sorted(F[(F.kind == "country") & (F.geofm == True)].name))  # noqa: E712
    print(G.to_string(index=False))


if __name__ == "__main__":
    main()
