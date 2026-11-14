"""Fig. 6 | Aid flagship: from intervals at the selected villages to funding now (paper/FIGURE_PLAN.md, result 3).

Rules: "poorest 5% in each country" is the country quota (quota_bottom: the predicted-poorest 5% of the test
villages in each of the 23 DHS countries); "poorest 5% overall" is bottom5. Random forest on the PovertyMap image
features unless marked DOFA (frozen DOFA under a trained head).
a  country quota: coverage shortfall below 0.90 among each country's selected villages (choropleth) for Split CP
   (the standard default), JOMI and SC-GeoCCP; Natural Earth 1:110m countries, borders drawn only around the 23
   study countries.
b  poorest 5% overall: mean width of the 90% interval at the selected villages and coverage among them (pooled over
   the 100 splits; 95% interval from the split-to-split variation).
c  country quota: coverage in every country (small marks), the worst country (large mark, ISO code on the right) and
   the sampling-noise limit (expected worst country if every country had exactly 0.90, independent binomial; fig_data.noise_limit). Right: SC-GeoCCP's coverage
   in each country against the villages the quota selects there per split, with the Theorem 1 floor
   0.9 - l1 E[1/(|R|+1) | selected] / E[f | selected] for the country indicator f, both under one condition: the
   training fold and map of repetition 0 and 200 calibration / test redraws (floors: Monte Carlo estimates from
   experiments/run_scg_slack.py, results/scg_slack_poverty.csv; coverage: experiments/run_audit_planner.py floors,
   n = 3,000, results/fia_db/planner/floors_poverty.csv).
d  funding now from the poorest 5% overall: false discovery rate (funded but not eligible) and the share of the
   eligible selected villages that are funded; eligible = below the country's first wealth quartile in the fold
   that fitted the scale model (the training fold). Point funding, interval triage (fund when the whole interval lies
   below the threshold) and cfBH on the JOMI reference set (FDR <= q); random forest only since 2026-10-28 (the frozen
   DOFA arm is in Supplementary Fig. 9, figures/make_si_backbones.py, which calls data_d / panel_d with BACKBONES).
Data: results/main_poverty.csv, bl_*, scg_*_l1fix, scg_groups_poverty.csv, state_poverty.csv,
scg_slack_poverty.csv, cfbh_poverty{,_dofa}.csv. The labels are DHS-derived, so only country- and split-level aggregates enter the figure
and its source data.
    FIGURE_QA_HELPERS=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python -I figures/make_fig6.py
writes figs/nc_fig6_aid.{pdf,svg,png}, figs/source_data/nc_fig6{a,b,c,c_floor,d}.csv, figs/qa/
"""
import os, sys
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402
import basemap as BM                                                  # noqa: E402

W, H = 183, 134                                       # 148 until 2026-11-10 (panel d compacted for PNAS)
NM, QUOTA, TOP = "poverty", "quota_bottom", "bottom5"
A_METH = ["SCP |res|", "JOMI stud", "SC-GeoCCP"]
B_METH = ["SCP |res|", "SCP stud", "GeoCP", "Regression kriging", "QRF", "JOMI stud", "SC-GeoCCP"]
C_METH = B_METH[:-1] + ["JOMI|state stud", "SC-GeoCCP"]
LABEL = dict(S.SHORT, **{"SCP |res|": "Split CP (default)", "JOMI|state stud": "JOMI per country"})
# results-file rule -> row label, style key; and the two backbones (results target, legend label)
D_RULES = [("point (mu < c)", "Predicted wealth < c", "point"), ("triage JOMI (U < c)", "Interval < c (JOMI)", "JOMI stud"),
           ("triage SC-GeoCCP (U < c)", "Interval < c (RegionCP)", "SC-GeoCCP"),   # no guarantee: not bolded
           ("cfBH q=0.05", "cfBH, q = 0.05", "cfBH"), ("cfBH q=0.10", "cfBH, q = 0.10", "cfBH")]
BACKBONES = [("poverty", "Random forest, image features"), ("poverty_dofa", "Frozen DOFA (GeoFM)")]
MAIN_BACKBONES = BACKBONES[:1]
# map window (lon, lat) and the labelled countries (coverage below LOW): inside when they fit, else here
LON, LAT, LOW = (-26.0, 52.0), (-35.5, 26.0), 0.85
OUTSIDE = {"SEN": (-21.0, 14.5, False), "CIV": (-5.0, 2.6, True), "ZWE": (22.5, -22.0, True)}
# layout (mm)
MW, MAPX = 49.0, (2.0, 53.5, 105.0)
B1, B2 = (25.0, 46.0, 27.0, 31.0), (56.0, 46.0, 18.0, 31.0)
C1, C2 = (101.0, 46.0, 31.0, 31.0), (151.0, 46.0, 30.0, 31.0)
D1, D2 = (37.0, 10.0, 64.0, 16.0), (112.0, 10.0, 64.0, 16.0)
LETTER_Y = (H - 1.5, 83.0, 34.0)                     # tops of the three rows of panel letters


# ------------------------------------------------------------------ data
@lru_cache(None)
def africa():
    """ADM0_A3 of every DHS group code, Natural Earth's African countries, and the clusters' km map."""
    z = D.DS.load(NM)
    proj = BM.km_projection(z["coords"], z["Xrf"][:, 0], z["Xrf"][:, 1])
    return [D.DHS[c][0] for c in D.dhs_countries()], BM.countries(lambda p: p["CONTINENT"] == "Africa"), proj


def extent():
    (x0, y0), (x1, y1) = africa()[2](np.array([[LON[0], LAT[0]], [LON[1], LAT[1]]]))
    return x0, x1, y0, y1


def data_c():
    """Coverage among each country's selected villages under the country quota (pooled over 100 splits)."""
    iso, name = africa()[0], {a: b for a, b in D.DHS.values()}
    g = D.groups(NM, "state", main=("SCP |res|",))
    g = g[(g.rule == QUOTA) & g.method.isin(C_METH)]
    T = pd.DataFrame(dict(key=g.method.values, method=g.method.map(LABEL).values,
                          iso=[iso[int(k)] for k in g.group], n_selected_pooled=g.n.values.astype(int),
                          coverage=(g.c / g.n).values, inf_share=(g.inf / g.n).values))
    T["country"] = T.iso.map(name); T["shortfall"] = np.maximum(0.0, 1 - S.ALPHA - T.coverage)
    n = T[T.key == "JOMI stud"].n_selected_pooled.values
    assert all(np.array_equal(np.sort(T[T.key == m].n_selected_pooled.values), np.sort(n)) for m in C_METH)
    return T, D.noise_limit("poverty", "state", n[n >= D.NMIN])


def data_floor(T):
    """Theorem 1 floor of every country (RegionCP, country quota) beside its coverage under the same conditions: the
    training fold and map of repetition 0 and 200 redraws of the 3,000-village audit and the candidates. The floors are
    run_scg_slack.py's Monte Carlo estimates; the coverage is from run_audit_planner.py floors (n = 3,000), which repeats
    those redraws (same seeds, same floors) and records coverage in each column. Until 2026-11-10 the coverage here was
    that of the main runs (100 splits, map refitted in each), a different condition from the floors'."""
    sl = pd.read_csv(os.path.join(S.RES, f"scg_slack_{NM}.csv"))
    assert (sl.rule == QUOTA).all()
    u, iso = sl[sl.kind == "unit"].set_index("unit"), africa()[0]
    pl = pd.read_csv(os.path.join(S.RES, "fia_db", "planner", f"floors_{NM}.csv"))
    pl = pl[(pl.n == 3000) & (pl.kind == "unit")].set_index("unit").reindex(u.index)
    assert pl.floor.notna().all() and np.allclose(pl.floor, u.floor, atol=1e-4)          # same redraws, same floors
    G = pd.DataFrame(dict(iso=[iso[int(k)] for k in u.index], selected_per_split=(u.Ef * u.selected_per_split).values,
                          floor=u.floor.values, slack=u.slack.values, coverage=pl.cov_scg.values,
                          coverage_jomi=pl.cov_jomi.values, redraws=int(pl.redraws.iloc[0])))
    x = T[T.key == "SC-GeoCCP"].set_index("iso")
    G["country"], G["coverage_100_splits_refitted"] = G.iso.map(x.country), G.iso.map(x.coverage)
    G["n_bar"], G["l1"], G["splits"], G["K"] = sl.nbar.iloc[0], sl.l1.iloc[0], sl.splits.iloc[0], sl.selected_per_split.iloc[0]
    return G.sort_values("floor").reset_index(drop=True)


def data_b():
    P = D.pooled(D.runs(NM)).loc[TOP]
    ref = P.loc["JOMI stud", "width"]
    return pd.DataFrame([dict(key=m, method=LABEL[m], coverage=P.loc[m, "cov"], lo=P.loc[m, "lo"], hi=P.loc[m, "hi"],
                              mean_width=2 * P.loc[m, "width"], width_vs_jomi=P.loc[m, "width"] / ref,
                              inf_share=P.loc[m, "inf"], selected_per_split=P.loc[m, "K"]) for m in B_METH])


def data_d(backbones=MAIN_BACKBONES):
    rows = []
    for tgt, blab in backbones:
        c = pd.read_csv(os.path.join(S.RES, f"cfbh_{tgt}.csv"))
        c = c[c.rule == "bottom 5%"]
        for code, lab, key in D_RULES:
            x = c[c.method == code]; reach = x.n_fund_elig / x.n_elig
            rows.append(dict(backbone=blab, target=tgt, code=code, rule=lab, key=key, splits=len(x),
                             fdr=x.fdp.mean(), fdr_se=x.fdp.std(ddof=1) / np.sqrt(len(x)),
                             reached=x.n_fund_elig.sum() / x.n_elig.sum(), reached_se=reach.std(ddof=1) / np.sqrt(len(x)),
                             funded_mean=x.n_fund.mean(), eligible_mean=x.n_elig.mean(), selected_mean=x.n_sel.mean(),
                             share_splits_funding_none=(x.n_fund == 0).mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ a
def map_axes(fig, x, y, mh):
    ax = S.axes_mm(fig, x, y, MW, mh)
    x0, x1, y0, y1 = extent()
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def panel_a(fig, T, y, mh):
    iso, af, proj = africa()
    cmap, norm = S.ramp("Reds", 0.18), Normalize(0, 0.16)
    axs = []
    for k, mth in enumerate(A_METH):
        ax = map_axes(fig, MAPX[k], y, mh); axs.append(ax)
        x = T[T.key == mth].set_index("iso")
        fills = {c: S.GREY_L if v <= 0 else cmap(norm(v)) for c, v in x.shortfall.items()}
        BM.fill(ax, af, proj, fills, box=extent())
        BM.outline(ax, af, proj, coast=BM.COAST, borders_of=set(iso), box=extent())
        w = x.coverage.idxmin()
        ax.text(0, 1.0, f"{S.SHORT[mth]}: worst country {x.coverage.min():.2f} ({w})", transform=ax.transAxes,
                fontsize=S.FS["note"], ha="left", va="bottom", color=S.INK)
        low = {c: c for c in x.index[x.coverage < LOW]}
        if low:
            fit = BM.label_features(ax, af, proj, low, fills, outside=OUTSIDE)
            print(f"{S.SHORT[mth]} labels (margin): " + ", ".join(f"{c} {v:.2f}" for c, v in fit.items()))
    cax = S.axes_mm(fig, 158.0, y + 2.0, 1.8, mh - 11.0)
    cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax)
    cb.set_ticks([0, 0.05, 0.1, 0.15]); cb.set_ticklabels(["0", "0.05", "0.10", "0.15"])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=1.5, width=0.4)
    cb.set_label("Shortfall below 0.90", fontsize=S.FS["label"], labelpad=2)
    cax.legend(handles=[Patch(facecolor=S.GREY_L, edgecolor="none", label="≥ 0.90"),
                        Patch(facecolor="white", edgecolor=BM.COAST, linewidth=0.4, label="Not in DHS data")],
               loc="lower left", bbox_to_anchor=(-0.3, 1.015), handlelength=0.8, handleheight=0.8, borderaxespad=0,
               handletextpad=0.3, labelspacing=0.3)
    return axs, cax


# ------------------------------------------------------------------ b
def rows(ax, n):
    ax.set_ylim(n - 0.45, -0.55); ax.tick_params(axis="y", length=0); S.clean(ax, left=False)


def panel_b(fig, T):
    b1, b2 = S.axes_mm(fig, *B1), S.axes_mm(fig, *B2)
    n = len(T)
    for ax in (b1, b2):
        rows(ax, n)
    for i, r in enumerate(T.itertuples()):
        c = S.METHODS[r.key][1]
        b1.barh(i, r.mean_width, height=0.5, color=c, edgecolor="none", zorder=2)
        b1.annotate(f"{r.width_vs_jomi:.2f}×", (r.mean_width, i), xytext=(2, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=S.FS["note"], color=S.INK2)
        b2.plot([r.lo, r.hi], [i, i], color=c, lw=S.LW["err"], zorder=2)
        S.dot(b2, [r.coverage], [i], r.key, s=12, z=3)
    S.target(b2)
    b1.set_yticks(range(n)); b1.set_yticklabels(T.method, fontsize=S.FS["label"])
    b2.set_yticks([])
    b1.set_xlim(0, 1.7); b1.set_xticks([0, 0.5, 1.0, 1.5]); b1.set_xticklabels(["0", "0.5", "1.0", "1.5"])
    b2.set_xlim(0.85, 1.0); b2.set_xticks([0.85, 0.9, 0.95, 1.0]); b2.set_xticklabels(["0.85", "0.9", "0.95", "1"])
    b1.set_xlabel("Interval width, wealth index\n(× JOMI's width)"); b2.set_xlabel("Coverage")
    return [b1, b2]


# ------------------------------------------------------------------ c
def panel_c(fig, T, lim, G):
    c1 = S.axes_mm(fig, *C1)
    n = len(C_METH); rows(c1, n)
    c1.set_xlim(0.70, 1.008)
    S.target(c1); c1.axvline(lim, color=S.INK2, lw=S.LW["ref"], zorder=1)
    rng = np.random.default_rng(0)                    # vertical jitter of the marks only; no data are simulated
    for i, mth in enumerate(C_METH):
        x = T[T.key == mth].sort_values("coverage")
        S.dot(c1, x.coverage.values[1:], i + rng.uniform(-0.18, 0.18, len(x) - 1), mth, s=6, z=3, alpha=0.6)
        S.dot(c1, x.coverage.values[:1], [i], mth, s=20, z=4, edge=S.INK)
        c1.text(1.03, i, f"{x.iso.iloc[0]} {x.coverage.iloc[0]:.2f}", transform=c1.get_yaxis_transform(),
                ha="left", va="center", fontsize=S.FS["note"], color=S.INK2)
        if mth == "JOMI|state stud":
            inf = x.inf_share @ x.n_selected_pooled / x.n_selected_pooled.sum()
            c1.text(0.705, i, f"+∞ for {inf:.0%}", ha="left", va="center", fontsize=S.FS["note"], color=S.INK2)
    c1.set_yticks(range(n)); c1.set_yticklabels([LABEL[m] for m in C_METH], fontsize=S.FS["label"])
    c1.set_xticks([0.7, 0.8, 0.9, 1.0]); c1.set_xticklabels(["0.7", "0.8", "0.9", "1"])
    c1.set_xlabel("Coverage among a country's\nselected villages")
    c1.text(lim - 0.004, -0.62, "Noise limit", ha="right", va="bottom", fontsize=S.FS["note"], color=S.INK2)
    c1.text(1.03, -0.62, "Worst", transform=c1.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=S.FS["note"], color=S.INK2, fontstyle="italic")
    # right: RegionCP (SC-GeoCCP in the results files) in every country against its Theorem 1 floor, a function of the villages selected there
    c2 = S.axes_mm(fig, *C2); S.clean(c2)
    lo, c = 0.38, G.l1[0] * G.K[0] / G.n_bar[0]          # floor(k) = 0.9 - c / k; start where it enters the axes
    k = np.linspace(c / (1 - S.ALPHA - lo), 16.0, 200)
    c2.plot(k, 1 - S.ALPHA - c / k, color=S.VIOLET, lw=S.LW["line"], zorder=2)
    S.target(c2, orient="h")
    by = G.groupby("selected_per_split").coverage                     # spread countries that share a quota
    off = (by.rank(method="first") - 1 - (by.transform("size") - 1) / 2) * 0.22
    S.dot(c2, G.selected_per_split + off, G.coverage, "SC-GeoCCP", s=10, z=3, emph=False)
    for iso, dxy, ha, va in (("CIV", (-3, -2), "right", "center"), ("GIN", (0, -4), "center", "top")):   # clear of crowded marks
        j = G.index[G.iso == iso][0]
        c2.annotate(iso, (G.selected_per_split[j] + off[j], G.coverage[j]), xytext=dxy, textcoords="offset points",
                    ha=ha, va=va, fontsize=S.FS["note"], color=S.INK2)
    c2.set_xlim(0, 16.5); c2.set_ylim(lo, 1.02)
    c2.set_xticks([2, 5, 10, 15]); c2.set_yticks([0.4, 0.6, 0.8, 1.0]); c2.set_yticklabels(["0.4", "0.6", "0.8", "1"])
    c2.set_xlabel("Villages selected per split"); c2.set_ylabel("Coverage, map fixed", labelpad=1)
    hs = [Line2D([], [], color=S.VIOLET, lw=S.LW["line"], label="Theorem 1 floor"),
          S.handle("SC-GeoCCP", label="RegionCP, a country")]
    c2.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.0, 0.0), handletextpad=0.3, borderaxespad=0.2)
    S.text_mm(fig, C2[0], C2[1] + C2[3] + 1.0, "RegionCP vs its floor", fontsize=S.FS["note"], ha="left",
              va="bottom", color=S.INK)
    return [c1, c2]


# ------------------------------------------------------------------ d
def panel_d(fig, T, backbones=MAIN_BACKBONES, d_ax=(D1, D2), size=(W, H), fdr_max=0.30):
    d1, d2 = S.axes_mm(fig, *d_ax[0]), S.axes_mm(fig, *d_ax[1])
    n = len(D_RULES); off = {"poverty": -0.17, "poverty_dofa": 0.17} if len(backbones) > 1 else {"poverty": 0.0}
    for ax, xl in ((d1, (-0.01, fdr_max)), (d2, (-0.02, 1.02))):
        rows(ax, n); ax.set_xlim(*xl); S.hgrid(ax, range(n))
    for i, (code, lab, key) in enumerate(D_RULES):
        for tgt, _ in backbones:
            r = T[(T.target == tgt) & (T.code == code)].iloc[0]; yy = i + off[tgt]; c = S.METHODS[key][1]
            for ax, v, se in ((d1, r.fdr, r.fdr_se), (d2, r.reached, r.reached_se)):
                ax.plot([max(0, v - 1.96 * se), v + 1.96 * se], [yy, yy], color=c, lw=S.LW["err"], zorder=2)
                S.dot(ax, [v], [yy], key, s=24 if key == "point" else 13, z=3, hollow=tgt != "poverty")
        if key == "cfBH":
            q = float(code.split("=")[1])
            d1.plot([q, q], [i - 0.42, i + 0.42], color=S.INK2, lw=S.LW["ref"], ls=(0, (2, 1.5)), zorder=1)
    d1.set_yticks(range(n)); d1.set_yticklabels([lab for _, lab, _ in D_RULES], fontsize=S.FS["label"])
    d2.set_yticks([])
    tk = [v for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5) if v <= fdr_max + 1e-9]
    d1.set_xticks(tk); d1.set_xticklabels([f"{v:g}" for v in tk])
    d2.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); d2.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    d1.set_xlabel("False discovery rate (funded but not eligible)")
    d2.set_xlabel("Share of eligible selected villages funded")
    hs = [Line2D([], [], ls=(0, (2, 1.5)), lw=S.LW["ref"], color=S.INK2, label="cfBH level q")]
    if len(backbones) > 1:
        hs = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=S.INK2, mec=S.INK2, label=backbones[0][1]),
              Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=S.INK2, mew=0.75, label=backbones[1][1])] + hs
    Wf, Hf = size; a0 = d_ax[0]
    fig.legend(handles=hs, ncol=3, loc="lower left", bbox_to_anchor=(a0[0] / Wf, (a0[1] + a0[3] + 1.4) / Hf),
               handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
    return [d1, d2]


def main():
    S.apply()
    Tc, lim = data_c(); Tb, Td, G = data_b(), data_d(), data_floor(Tc)
    x0, x1, y0, y1 = extent(); mh = MW * (y1 - y0) / (x1 - x0)
    my = H - 7.5 - mh
    fig = S.figure(W, H)
    A, cb = panel_a(fig, Tc[Tc.key.isin(A_METH)], my, mh)
    B = panel_b(fig, Tb); C = panel_c(fig, Tc, lim, G); Dd = panel_d(fig, Td)
    ya, yb, yd = LETTER_Y
    for x, y, l, t in ((2.0, ya, "a", "Poorest 5% in each country: coverage among the selected villages"),
                       (2.0, yb, "b", "Poorest 5% overall: interval width and coverage"),
                       (77.0, yb, "c", "Poorest 5% in each country: coverage by country"),
                       (2.0, yd, "d", "Funding now from the poorest 5% overall")):
        S.letter_mm(fig, x, y, l)
        S.text_mm(fig, x + 4.5, y - 0.35, t, fontsize=S.FS["title"], fontweight="bold", ha="left", va="top")
    S.text_mm(fig, 64.0, yd - 0.6, "c: eligibility threshold, the 25th percentile of the country's wealth in the fold that fitted the scale model",
              fontsize=S.FS["note"], color=S.INK2, ha="left", va="top")
    S.source_data("nc_fig6a", Tc[Tc.key.isin(A_METH)]); S.source_data("nc_fig6b", Tb)
    S.source_data("nc_fig6c", Tc.assign(noise_limit=lim)); S.source_data("nc_fig6c_floor", G)
    S.source_data("nc_fig6d", Td)
    S.save(fig, "nc_fig6_aid", axes=A + B + C + Dd, panel_ids=["a1", "a2", "a3", "b1", "b2", "c1", "c2", "d1", "d2"],
           row_groups=[["a1", "a2", "a3"], ["b1", "b2"], ["b2", "c1"], ["c1", "c2"], ["d1", "d2"]], column_groups=[],
           exclude_axes=[cb])
    print(f"map {MW:.0f} x {mh:.1f} mm; noise limit {lim:.3f}")
    print(Tb[["method", "coverage", "width_vs_jomi"]].round(3).to_string(index=False))
    w = Tc.loc[Tc.groupby("key").coverage.idxmin(), ["method", "country", "coverage"]]
    print(w.round(3).to_string(index=False))
    print(f"Theorem 1 floor by country: {G.floor.min():.3f} ({G.iso[0]}) to {G.floor.max():.3f}; "
          f"countries at or above their floor (map fixed, same redraws): {(G.coverage >= G.floor).sum()} of {len(G)}; "
          f"coverage {G.coverage.min():.3f} to {G.coverage.max():.3f}")


if __name__ == "__main__":
    main()
