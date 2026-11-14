"""Fig. 7 | Carbon flagship: declared risk, and crediting plots vs crediting the project (paper/FIGURE_PLAN.md, result 4).

VCS MR v5.0 deducts 0.4307 standard errors and declares an over-crediting risk of 1/3 (ART TREES: the SI figure,
figures/make_si_carbon.py, which reuses data_ac and panel_ac with fam="ART").
a-c  realized over-crediting risk of the project total against the selection strength (top 0.5-10% of the candidate
     plots by the map) for three maps (site attributes; site + canopy height; full ground inventory): map total
     without deduction; the protocol rule with the map error layer, with the survey-calibrated (model-assisted)
     estimate, and with the selection-aware estimate (ours); the exact selection-aware bound (ours). Thin line beneath
     them: the domain estimate within ten fixed strata (deciles of each map's training-fold predictions, the bins of
     Mondrian CP) credited with the selection-aware series' formula (VCS deduction without the waiver), i.e. column
     "Domain, 10 strata, formula without waiver" of Table D1 (results/RESULTS_CARBON_DOMAIN.md); asserted to 3 decimals.
d    top 1%: realized risk of every map (training fold; 20 per map type) for the protocol rule with the map error
     layer and with the selection-aware estimate; bars are medians.
e    site map, top 1%, shared x = credited / true carbon. Top: every selected plot credited at the lower end of its
     90% interval (share of plots over-credited). Bottom: the project total credited with the VCS formula (project
     over-crediting risk), the series of a-c.
Audit share 0.5; 20 maps x 100 audits per map type (experiments/run_carbon.py, run_carbon_units.py). Risks are
frequencies over all maps and audits; the map-clustered 95% intervals are in the source data (table C1).
Data: results/carbon_{site,siteheight,full}.csv.gz, carbon_units_*.csv.gz, through experiments/make_carbon.py
(which writes results/RESULTS_CARBON.md); the thin line from results/carbon_domain_*.csv.gz (experiments/
run_carbon_domain.py: same maps, audits and selections), joined row by row by experiments/make_carbon_domain.py.
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_fig7.py
writes figs/nc_fig7_carbon.{pdf,svg,png}, figs/source_data/nc_fig7{a_c,d,e}.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
sys.path.insert(0, os.path.join(S.ROOT, "experiments"))
import make_carbon as MC                                              # noqa: E402  (also sets the old rcParams)
import make_carbon_domain as MCD                                      # noqa: E402  (joined, r3: Table D1)

W, H = 183, 128
FAM, DELTA = "VCS", 1 / 3
ROLES = ["point", "map", "ma", "ours", "exact"]                       # make_carbon roles, legend order
KEYS = {p: {r: MC.col(p, r) for r in ROLES} for p in MC.FAM}          # -> results column = style.METHODS key
KEY = KEYS[FAM]
LABEL = {"point": "Map total, no deduction"}
DECLARED = {"VCS": "Declared risk (VCS, 1/3)", "ART": "Declared risk (ART TREES, 0.30)"}
# plot-level methods of e (results/carbon_units_*.csv.gz) -> style.METHODS key
UNIT = {"point": "point", "Normalized CP": "SCP stud", "GeoCP": "GeoCP", "QRF": "QRF",
        "Bootstrap": "Bootstrap ensemble + residual", "JOMI": "JOMI stud",
        "SC-GeoCCP": "SC-GeoCCP"}                 # RegionCP (ours), computed at the 1% share only (added 2026-11-08)
MAP_TICK = {"site": "Site", "site+height": "Site +\nheight", "full": "Full\ninventory"}
# a-c thin line (Fig. 7 only; Supplementary Fig. 15 calls panel_ac without it): results column, its column in Table D1,
# legend label. Muted mauve, a hue no method family uses; all-pairs check against the a-c inks (black, red, aqua, blue,
# declared-risk grey): normal-vision dE >= 16.3, CVD dE >= 9.7, 4.2:1 on white. Drawn under every other series.
FIXED = dict(key="VCS domain10, formula", d1="Domain, 10 strata, formula without waiver",
             label="Fixed deciles, same formula", color="#94708c", gid="fixed-deciles")
# layout (mm)
TOP = [(14.0 + 58.0 * i, 76.0, 50.0, 36.0) for i in range(3)]
DD = (14.0, 12.0, 66.0, 46.0)
E1, E2 = (100.0, 37.5, 52.0, 20.5), (100.0, 12.0, 52.0, 20.5)


def label(role):
    return LABEL.get(role, S.SHORT[KEY[role]])


def marker_kw(key, ms=3.2):
    _, c, m, f = S.METHODS[key]
    return dict(color=c, marker=m, ms=ms * 1.5 if m == "*" else ms, mfc=c if f else "white", mec=c,
                mew=S.LW["marker"] if f else 0.75)


def fixed_kw():
    c = FIXED["color"]
    return dict(color=c, lw=S.LW["ref"], marker="^", ms=2.4, mfc=c, mec=c, mew=0.4)


# ------------------------------------------------------------------ data
def data_ac(df, fam=FAM):
    key, a, rows = KEYS[fam], df[df.audit == 0.5], []
    for mp in MC.MAPS:
        y = a[a["map"] == mp]
        for role in ROLES:
            rk, q = MC.risk(y, key[role], ["frac"]), MC.ratio(y, key[role], ["frac"])
            rows += [dict(map=MC.TITLE[mp], code=mp, heldout_r2=y.r2.mean(), series=label(role), key=key[role],
                          top_share=f, risk=r.risk, ci95_halfwidth=r.ci, risk_min_map=r.fmin,
                          risk_median_map=r.fmed, risk_max_map=r.fmax, maps=int(r.nf),
                          credited_over_true=q[f]) for f, r in rk.iterrows()]
    return pd.DataFrame(rows)


def data_fixed():
    """a-c thin line, in the columns of data_ac: the same maps, audits (share 0.5) and selections, from the domain run
    joined to the run_carbon rows by make_carbon_domain.joined (as for Table D1)."""
    a, k, rows = MCD.joined(), FIXED["key"], []
    a = a[a.audit == 0.5]
    for mp in MC.MAPS:
        y = a[a["map"] == mp]
        rk, q = MC.risk(y, k, ["frac"]), MC.ratio(y, k, ["frac"])
        rows += [dict(map=MC.TITLE[mp], code=mp, heldout_r2=y.r2.mean(), series=FIXED["label"], key=k, top_share=f,
                      risk=r.risk, ci95_halfwidth=r.ci, risk_min_map=r.fmin, risk_median_map=r.fmed, risk_max_map=r.fmax,
                      maps=int(r.nf), credited_over_true=q[f]) for f, r in rk.iterrows()]
    return pd.DataFrame(rows)


def table_d1(col):
    """One column of Table D1 in results/RESULTS_CARBON_DOMAIN.md as printed: {(map title, top share), '0.526'}."""
    L = open(os.path.join(MC.RES, "RESULTS_CARBON_DOMAIN.md"), encoding="utf-8").read().splitlines()
    i = next(k for k, s in enumerate(L) if s.startswith("## 表 D1"))
    split = lambda s: [c.strip() for c in s.strip().strip("|").split("|")]
    j, out = split(L[i + 2]).index(col), {}
    for s in L[i + 4:]:
        if not s.startswith("|"):
            break
        c = split(s); out[(c[0], c[1])] = c[j].split()[0]
    return out


def check_fixed(axs, Tac):
    """The plotted thin line equals Table D1's column to 3 decimals (half up, as the table rounds), and the filled
    circles it is set against equal Table D1's selection-aware column (same audits, so the comparison is paired)."""
    d1, sa = table_d1(FIXED["d1"]), table_d1("Selection-aware (reference set)")
    plotted = {}
    for ax, mp in zip(axs, MC.MAPS):
        ln, = [l for l in ax.lines if (l.get_gid() or "").startswith(FIXED["gid"])]
        plotted.update({(MC.TITLE[mp], f"{x:g}%"): MCD.r3(v) for x, v in zip(ln.get_xdata(), ln.get_ydata())})
    assert len(d1) == 15 and plotted == d1, ("thin line != Table D1", plotted, d1)
    ours = {(r.map, MC.pct(r.top_share)): MCD.r3(r.risk) for r in Tac[Tac.key == KEY["ours"]].itertuples()}
    assert ours == sa, ("selection-aware series != Table D1", ours, sa)
    return plotted


def data_d(df):
    b, rows = df[(df.audit == 0.5) & (df.frac == 0.01)], []
    for mp in MC.MAPS:
        y = b[b["map"] == mp]
        for role in ("map", "ours"):
            pf = (y[KEY[role]] > y.total).groupby(y.fold).mean()
            rows += [dict(map=MC.TITLE[mp], code=mp, series=label(role), key=KEY[role], fold=int(k), risk=v)
                     for k, v in pf.items()]
    return pd.DataFrame(rows)


def data_e(df, units):
    rows = []
    x = units[(units["map"] == "site") & (units.frac == 0.01)]
    for m, key in UNIT.items():
        g = x[x.method == m]
        rows.append(dict(part="credit each plot", method="Map value μ" if m == "point" else S.SHORT[key], key=key,
                         credited_over_true=g.credit.sum() / g.total.sum(),
                         share_plots_overcredited=g.n_over.sum() / g.K.sum(),
                         project_overcredit_risk=np.mean(g.credit > g.total)))
    y = df[(df.audit == 0.5) & (df.frac == 0.01) & (df["map"] == "site")]
    for role in ROLES:
        rows.append(dict(part="credit the project", method=label(role), key=KEY[role],
                         credited_over_true=y[KEY[role]].sum() / y.total.sum(),
                         project_overcredit_risk=np.mean(y[KEY[role]] > y.total)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ panels
def panel_ac(fig, T, fam=FAM, top=TOP, legend_y=118.0, fixed=None):
    key, (w, h) = KEYS[fam], fig.get_size_inches() / S.MM
    axs = [S.axes_mm(fig, *r) for r in top]
    x = np.sort(T.top_share.unique()) * 100
    for i, (ax, mp) in enumerate(zip(axs, MC.MAPS)):
        t = T[T.code == mp]
        for role in ROLES:
            r = t[t.key == key[role]].sort_values("top_share")
            ax.plot(x, r.risk.values, lw=S.LW["line"], zorder={"ours": 4, "point": 3.5}.get(role, 3),
                    **marker_kw(key[role]))
        if fixed is not None:                # thin line under the main series (zorder 3-4), over the reference line (1)
            r = fixed[fixed.code == mp].sort_values("top_share")
            assert np.allclose(r.top_share.values * 100, x), (mp, r.top_share.values)
            ax.plot(x, r.risk.values, zorder=2.5, gid=f"{FIXED['gid']}-{'abc'[i]}", **fixed_kw())   # SVG ids unique
        S.target(ax, v=MC.FAM[fam]["delta"], orient="h"); S.hgrid(ax, [0.2, 0.4, 0.6, 0.8])
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_xlim(0.4, 12.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_ylim(-0.03, 0.85); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        if i:
            ax.set_yticklabels([])
        S.text_mm(fig, top[i][0], top[i][1] + top[i][3] + 1.2, f"{MC.TITLE[mp]} (held-out R² = {t.heldout_r2.iloc[0]:.2f})",
                  fontsize=S.FS["label"], ha="left", va="bottom")
    axs[0].set_ylabel("Realized over-crediting risk")
    axs[1].set_xlabel("Top share of the candidate plots selected by the map (%)")
    hs = [Line2D([], [], lw=S.LW["line"], label=label(r), **marker_kw(key[r])) for r in ROLES]
    hs.append(Line2D([], [], color=S.INK2, lw=S.LW["ref"], ls=(0, (3, 2)), label=DECLARED[fam]))
    if fixed is not None:                    # a fourth legend column: columns 1-3 keep their places
        hs.append(Line2D([], [], label=FIXED["label"], **fixed_kw()))
    fig.legend(handles=hs, ncol=3 if fixed is None else 4, loc="lower left", bbox_to_anchor=(top[0][0] / w, legend_y / h),
               handlelength=1.7, columnspacing=1.0, borderaxespad=0)
    return axs


def panel_d(fig, T):
    ax = S.axes_mm(fig, *DD); rng = np.random.default_rng(0)  # horizontal jitter of the marks only
    for i, mp in enumerate(MC.MAPS):
        for off, role in ((-0.18, "map"), (0.18, "ours")):
            v = T[(T.code == mp) & (T.key == KEY[role])].risk.values
            S.dot(ax, i + off + rng.uniform(-0.07, 0.07, len(v)), v, KEY[role], s=9, z=3, alpha=0.85)
            ax.plot([i + off - 0.13, i + off + 0.13], [np.median(v)] * 2, color=S.INK, lw=S.LW["line"], zorder=4)
    S.target(ax, v=DELTA, orient="h"); S.hgrid(ax, [0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(range(len(MC.MAPS))); ax.set_xticklabels([MAP_TICK[m] for m in MC.MAPS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(-0.04, 1.04); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_ylabel("Realized risk of each map")
    S.text_mm(fig, DD[0], DD[1] + DD[3] + 1.2, "Top 1%: every map (20 per type); bar = median",
              fontsize=S.FS["label"], ha="left", va="bottom")
    return ax


def panel_e(fig, T):
    e1, e2 = S.axes_mm(fig, *E1), S.axes_mm(fig, *E2)
    for ax, part, ycol in ((e1, "credit each plot", "share_plots_overcredited"),
                           (e2, "credit the project", "project_overcredit_risk")):
        for r in T[T.part == part].itertuples():
            star = S.METHODS[r.key][2] == "*"          # star underneath: in e2 it coincides with the survey square
            S.dot(ax, [r.credited_over_true], [getattr(r, ycol)], r.key, s=26 if star else 13, z=3 if star else 4)
        ax.axvline(1.0, color=S.INK2, lw=S.LW["ref"], ls=(0, (1, 1.5)), zorder=1)
        ax.set_xlim(-0.03, 1.15); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    e1.set_xticklabels([]); e2.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    e1.set_ylim(-0.04, 0.78); e1.set_yticks([0, 0.25, 0.5]); e1.set_yticklabels(["0", "0.25", "0.5"])
    S.hgrid(e1, [0.25, 0.5])
    e2.set_ylim(-0.04, 0.86); e2.set_yticks([0, DELTA, 2 * DELTA]); e2.set_yticklabels(["0", "1/3", "2/3"])
    S.target(e2, v=DELTA, orient="h"); S.hgrid(e2, [2 * DELTA])
    e1.set_ylabel("Plots over-\ncredited", labelpad=2); e2.set_ylabel("Project over-\ncrediting risk", labelpad=2)
    e2.set_xlabel("Credited / true carbon")
    e1.text(0.02, 0.97, "Credit each plot at its 90% lower bound", transform=e1.transAxes, ha="left", va="top",
            fontsize=S.FS["note"], color=S.INK2)
    e2.text(0.02, 0.97, "Credit the project total (VCS formula)", transform=e2.transAxes,
            ha="left", va="top", fontsize=S.FS["note"], color=S.INK2)
    low = T[(T.part == "credit each plot") & ~T.key.isin(["point", "GeoCP"])]
    e1.annotate(f"{len(low)} methods credit ≤ {100 * low.credited_over_true.max():.0f}%", xy=(0.07, 0.03),
                xytext=(0.07, 0.37), textcoords="data", ha="left", va="center", fontsize=S.FS["note"], color=S.INK2,
                arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.4, shrinkA=1, shrinkB=2))
    hs = [Line2D([], [], ls="none", label=r.method, **marker_kw(r.key))
          for r in T[T.part == "credit each plot"].itertuples()]
    e1.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.03, 1.0), borderaxespad=0, handletextpad=0.3)
    S.text_mm(fig, E1[0], E1[1] + E1[3] + 1.2, "Site map, top 1%: crediting plots or the project (bottom: series of a–c)",
              fontsize=S.FS["label"], ha="left", va="bottom")
    return [e1, e2]


def main():
    matplotlib.rcdefaults(); S.apply()       # importing make_carbon applied make_figs' rcParams (tight bbox, 9 pt)
    df, units = MC.load(), MC.load_units()
    Tac, Td, Te, Tfx = data_ac(df), data_d(df), data_e(df, units), data_fixed()
    assert np.allclose(Tfx.groupby("code").heldout_r2.first(), Tac.groupby("code").heldout_r2.first())
    fig = S.figure(W, H)
    A = panel_ac(fig, Tac, fixed=Tfx); Dd = panel_d(fig, Td); E = panel_e(fig, Te)
    d1 = check_fixed(A, Tac)
    for x, y, l in ((2.0, 116.5, "a"), (66.0, 116.5, "b"), (124.0, 116.5, "c"), (2.0, 62.0, "d"), (90.0, 62.0, "e")):
        S.letter_mm(fig, x, y, l)
    S.source_data("nc_fig7a_c", pd.concat([Tac, Tfx], ignore_index=True))       # thin line: the last 15 rows
    S.source_data("nc_fig7d", Td); S.source_data("nc_fig7e", Te)
    S.save(fig, "nc_fig7_carbon", axes=A + [Dd] + E, panel_ids=["a", "b", "c", "d", "e1", "e2"],
           row_groups=[["a", "b", "c"]], column_groups=[["e1", "e2"]])
    t = Tac[Tac.top_share == 0.01].pivot(index="series", columns="map", values="risk")
    print("top 1% risk:\n" + t.round(3).to_string())
    print(Te.round(3).to_string(index=False))
    sa = table_d1("Selection-aware (reference set)")
    print(f"thin line = Table D1 '{FIXED['d1']}' (3 decimals, asserted); vs selection-aware:\n" +
          "\n".join(f"  {m:22s} {s:>4s}: {v} vs {sa[(m, s)]}" for (m, s), v in d1.items()))


if __name__ == "__main__":
    main()
