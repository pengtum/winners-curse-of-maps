"""Five questions a decision asks: the answer, and whether the answer holds. Every value comes from a stored result; nothing is rerun.

a  interval: figs/source_data/nc_fig5a.csv, nc_fig4a.csv (selected 5%, 11 random-forest settings)
b  project total: nc_fig7a_c.csv, si_fig_carbon_protocol_d_f.csv (FIA, site maps, highest 1%, audit share 0.5)
c  funding: nc_fig6d.csv (DHS, random forest with image features, poorest 5%)
d  region: nc_fig6c_floor.csv (DHS, country quota)
e  audit size: nc_fig9b.csv, results/fia_db/planner/summary_carbon.csv (30 plots among 6,000 FIA plots)
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_fig_questions.py
writes figs/nc_questions.{pdf,svg,png}, figs/source_data/nc_questions_{a,b,c,d,e}.csv
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "package", "src")]; sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
from make_fig_overview import bubble, text                            # noqa: E402

SD = os.path.join(ROOT, "figs", "source_data")
W, H, RH, TOP = 183, 146, 27.6, 139.8
AX, AW, TX, VX, VW, NX = 54.5, 32.5, 89.0, 126.5, 28.5, 157.0
PH, DY = 17.5, 6.6
JITTER = np.random.default_rng(3)


def rows_y(i):
    return TOP - (i + 1) * RH


def frame(fig, i, x, w):
    a = S.axes_mm(fig, x, rows_y(i) + DY, w, PH); S.clean(a)
    a.tick_params(pad=1.5); a.xaxis.labelpad = 1; a.yaxis.labelpad = 1
    return a


def note(ax, x, y, big, rows, col=S.INK):
    ax.text(x, y, big, fontsize=S.FS["title"] + .5, fontweight="bold", color=col, va="baseline")
    for k, r in enumerate(rows):
        ax.text(x, y - 3.1 - k * 2.55, r, fontsize=S.FS["note"], color=S.INK2, va="baseline")


def row_interval(fig, ax, i):
    w = pd.read_csv(os.path.join(SD, "nc_fig5a.csv")); w = w[w.label == "Split CP"]
    a = frame(fig, i, AX, AW)
    y = np.linspace(.2, .8, len(w)); o = np.argsort(w.width_rel_jomi.to_numpy())
    a.scatter(w.width_rel_jomi.to_numpy()[o], np.linspace(.15, .85, len(w)), s=7, color=S.GREY_D, lw=0, zorder=3)
    a.axvline(1, color=S.BLUE, lw=S.LW["ref"]); a.set_ylim(0, 1); a.set_yticks([]); a.set_xlim(0, 2); a.set_xticks([0, 1, 2])
    a.set_xlabel("Split-CP width / width needed")
    a.text(.96, .93, "Too narrow", ha="right", va="center", fontsize=S.FS["note"], color=S.MISS)
    a.text(1.04, .30, "Too wide", ha="left", va="center", fontsize=S.FS["note"], color=S.BLUE)
    note(ax, TX, rows_y(i) + 20.5, f"{w.width_rel_jomi.min():.1f}–{w.width_rel_jomi.max():.1f} times",
         ["the width needed: what", "split CP gives, against", "JOMI's, by setting"])
    c = pd.read_csv(os.path.join(SD, "nc_fig4a.csv")); c = c[c.method.isin(["Split CP", "Normalized CP", "JOMI"])]
    v = frame(fig, i, VX, VW)
    for k, (m, col, lab) in enumerate((("Split CP", S.GREY_M, "Split CP"), ("Normalized CP", S.GREY_D, "Normalized"), ("JOMI", S.BLUE, "JOMI"))):
        g = c[c.method == m]; v.scatter(g.coverage, np.full(len(g), 2 - k) + JITTER.uniform(-.17, .17, len(g)), s=6, color=col, lw=0, zorder=3)
    v.axvline(.9, color=S.INK2, lw=S.LW["ref"], ls=":"); v.set_ylim(-.6, 2.6); v.set_yticks([2, 1, 0]); v.set_yticklabels(["Split CP", "Normalized", "JOMI"])
    v.set_xlim(.3, 1.02); v.set_xticks([.4, .6, .8, 1.0]); v.set_xticklabels(["0.4", "0.6", "0.8", "1"]); v.set_xlabel("Coverage at selected units")
    jm = c[c.method == "JOMI"].coverage.min(); nm = c[c.method == "Normalized CP"].coverage.min(); sm = c[c.method == "Split CP"].coverage
    note(ax, NX, rows_y(i) + 20.5, f"≥ {np.floor(jm * 100) / 100:.2f}", ["JOMI in all 11, infinite", f"counted; split CP < 0.85", f"in {int((sm < .85).sum())}; normalized ≥ {nm:.2f}"], col=S.BLUE)
    return [a, v], pd.concat([w.assign(part="width"), c.assign(part="coverage")])


def row_total(fig, ax, i):
    c = pd.read_csv(os.path.join(SD, "nc_fig7a_c.csv")); c = c[(c.code == "site") & np.isclose(c.top_share, .01)]
    show = (("VCS model-assisted", "Survey-calibrated", S.RED), ("ours: VCS formula, sel.-aware", "Selection-aware", S.BLUE), ("ours: exact, risk VCS", "Exact bound", S.VIOLET))
    a = frame(fig, i, AX, AW)
    vals = [float(c[c.key == k].credited_over_true.iloc[0]) for k, _, _ in show]
    a.barh([2, 1, 0], vals, color=[x[2] for x in show], height=.55, zorder=2)
    a.axvline(1, color=S.INK2, lw=S.LW["ref"], ls=":"); a.set_yticks([2, 1, 0]); a.set_yticklabels(["Survey", "Sel.-aware", "Exact"]); a.set_xlim(0, 1.15); a.set_xticks([0, .5, 1]); a.set_xticklabels(["0", "0.5", "1"])
    a.set_xlabel("Credit / true total")
    note(ax, TX, rows_y(i) + 20.5, f"{vals[1]:.2f} of the total", ["selection-aware, at", f"declared risk 1/3; exact", f"bound {vals[2]:.2f}"], col=S.BLUE)
    d = pd.read_csv(os.path.join(SD, "si_fig_carbon_protocol_d_f.csv")); d = d[d.code == "site"]
    v = frame(fig, i, VX, VW)
    v.plot([0, .5], [0, .5], color=S.INK2, lw=S.LW["ref"], ls=":", zorder=1)
    for s, col in (("Map error layer", S.GREY_M), ("Survey-calibrated estimate", S.RED), ("Selection-aware estimate (ours)", S.BLUE)):
        g = d[d.series == s]; v.plot(g.delta, g.risk, color=col, lw=S.LW["line"], zorder=3)
    v.text(.10, .82, "Survey-calibrated", color=S.RED, fontsize=S.FS["note"], ha="left", va="center")
    v.set_xlim(0, .5); v.set_ylim(0, 1); v.set_xticks([0, .25, .5]); v.set_xticklabels(["0", "0.25", "0.5"]); v.set_yticks([0, .5, 1]); v.set_yticklabels(["0", "0.5", "1"])
    v.set_xlabel("Declared risk"); v.set_ylabel("Realized risk")
    g = d[d.series == "Selection-aware estimate (ours)"]; gap = float((g.risk - g.delta).abs().max())
    s13 = d[(d.series == "Survey-calibrated estimate") & np.isclose(d.delta, .335)].risk.iloc[0]
    note(ax, NX, rows_y(i) + 20.5, "On the line", [f"selection-aware, an", f"asymptotic result; survey", f"{s13:.2f} at 1/3; grey, map layer"], col=S.BLUE)
    return [a, v], pd.concat([c.assign(part="credit"), d.assign(part="risk")]), gap


def row_funding(fig, ax, i):
    f = pd.read_csv(os.path.join(SD, "nc_fig6d.csv")).set_index("code")
    show = (("point (mu < c)", "Map alone", S.GREY_D, None), ("cfBH q=0.10", "cfBH, q = 0.10", S.BLUE, .10), ("cfBH q=0.05", "cfBH, q = 0.05", S.BLUE, .05))
    a = frame(fig, i, AX, AW)
    a.barh([2, 1, 0], [f.funded_mean[k] for k, *_ in show], color=[x[2] for x in show], height=.55, zorder=2)
    a.set_yticks([2, 1, 0]); a.set_yticklabels(["Map alone", "q = 0.10", "q = 0.05"]); a.set_xlim(0, 65); a.set_xticks([0, 30, 60]); a.set_xlabel("Villages funded, of 150")
    r = f.loc["cfBH q=0.10"]
    note(ax, TX, rows_y(i) + 20.5, f"{r.funded_mean:.0f} villages", [f"at q = 0.10, {r.reached:.0%} of the", f"eligible; none in {r.share_splits_funding_none:.0%} of", "audits; survey the rest"], col=S.BLUE)
    v = frame(fig, i, VX, VW)
    for y, (k, lab, col, q) in zip((2, 1, 0), show):
        v.errorbar(f.fdr[k], y, xerr=1.96 * f.fdr_se[k], fmt="o", ms=3, color=col, lw=S.LW["err"], zorder=3)
        if q:
            v.plot([q, q], [y - .32, y + .32], color=S.INK, lw=1.0, zorder=2)
    v.set_yticks([2, 1, 0]); v.set_yticklabels(["Map alone", "q = 0.10", "q = 0.05"]); v.set_ylim(-.6, 2.6); v.set_xlim(0, .27); v.set_xticks([0, .1, .2]); v.set_xticklabels(["0", "0.1", "0.2"]); v.set_xlabel("False discovery rate")
    note(ax, NX, rows_y(i) + 20.5, "Below target", [f"{r.fdr:.2f} and {f.fdr['cfBH q=0.05']:.2f}; map", f"alone {f.fdr['point (mu < c)']:.2f}; tick =", "target q"], col=S.BLUE)
    return [a, v], f.reset_index()


def row_region(fig, ax, i):
    g = pd.read_csv(os.path.join(SD, "nc_fig6c_floor.csv"))
    a = frame(fig, i, AX, AW)
    K = np.linspace(1.5, 16, 100); a.plot(K, .9 - .95 * 3000 / (3001 * K), color=S.GREY_M, lw=S.LW["line"], zorder=1)
    a.scatter(g.selected_per_split, g.floor, s=7, color=S.VIOLET, lw=0, zorder=3)
    a.axhline(.9, color=S.INK2, lw=S.LW["ref"], ls=":"); a.set_xlim(0, 16); a.set_ylim(.3, 1); a.set_xticks([0, 5, 10, 15]); a.set_yticks([.4, .6, .8, 1]); a.set_yticklabels(["0.4", "0.6", "0.8", "1"])
    a.set_xlabel("Selected villages in the country"); a.set_ylabel("Floor")
    lo, hi = g.sort_values("floor").iloc[0], g.sort_values("floor").iloc[-1]
    note(ax, TX, rows_y(i) + 20.5, f"Floor {lo.floor:.2f}–{hi.floor:.2f}", [f"{lo.selected_per_split:.0f} to {hi.selected_per_split:.0f} villages; a floor", "guarantees, it does not", "forecast coverage"], col=S.VIOLET)
    v = frame(fig, i, VX, VW)
    v.plot([.3, 1], [.3, 1], color=S.INK2, lw=S.LW["ref"], ls=":", zorder=1)
    v.scatter(g.floor, g.coverage, s=7, color=S.VIOLET, lw=0, zorder=3)
    v.set_xlim(.3, 1); v.set_ylim(.3, 1); v.set_xticks([.4, .7, 1]); v.set_xticklabels(["0.4", "0.7", "1"]); v.set_yticks([.4, .7, 1]); v.set_yticklabels(["0.4", "0.7", "1"])
    v.set_xlabel("Floor"); v.set_ylabel("Coverage")
    assert (g.coverage > g.floor).all()
    note(ax, NX, rows_y(i) + 20.5, "All 23 above", ["the floor; lowest", f"country {g.coverage.min():.2f}, {int((g.coverage < .85).sum())} under", "0.85"], col=S.VIOLET)
    return [a, v], g


def row_audit(fig, ax, i):
    b = pd.read_csv(os.path.join(SD, "nc_fig9b.csv")); s = pd.read_csv(os.path.join(ROOT, "results", "fia_db", "planner", "summary_carbon.csv"))
    a = frame(fig, i, AX, AW)
    for lab, col in (("Whole map", S.GREY_M), ("Highest 5%", S.BLUE)):
        g = b[(b.stratum == lab) & (b.n >= 20)]; a.plot(g.n, g.planned, color=col, lw=S.LW["line"], zorder=2)
    a.axhline(.9, color=S.INK2, lw=S.LW["ref"], ls=":"); a.set_xscale("log"); a.set_xlim(20, 3300); a.minorticks_off(); a.set_xticks([100, 1000]); a.set_xticklabels(["100", "1,000"]); a.set_ylim(-.03, 1.03); a.set_yticks([0, .5, 1]); a.set_yticklabels(["0", "0.5", "1"])
    a.set_xlabel("Audited plots"); a.set_ylabel("Chance finite")
    a.text(24, .55, "Top 5%", color=S.BLUE, fontsize=S.FS["note"]); a.text(180, .35, "Whole map", color=S.GREY_D, fontsize=S.FS["note"])
    note(ax, TX, rows_y(i) + 20.5, "92 or 1,869", ["in the top 5% or across the", "map, for a new audit that", "must precede the decision"], col=S.BLUE)
    v = frame(fig, i, VX, VW)
    v.plot([0, 1], [0, 1], color=S.INK2, lw=S.LW["ref"], ls=":", zorder=1)
    for m, mk, fc in (("cci", "s", "white"), ("site", "o", S.BLUE)):
        g = s[s["map"] == m]; v.scatter(g.pf, g.fin, marker=mk, s=9, facecolor=fc, edgecolor=S.BLUE, lw=S.LW["marker"], zorder=3 if m == "cci" else 4)
    v.set_xlim(-.03, 1.03); v.set_ylim(-.03, 1.03); v.set_xticks([0, .5, 1]); v.set_xticklabels(["0", "0.5", "1"]); v.set_yticks([0, .5, 1]); v.set_yticklabels(["0", "0.5", "1"])
    v.set_xlabel("Planned chance"); v.set_ylabel("Realized")
    t5 = s[(s.q == .05) & (s.n == 92)]
    assert (t5.pf.round(1) == .9).all() and (s.pf - s.fin).abs().max() < .02
    note(ax, NX, rows_y(i) + 20.5, "As planned", ["16 audit designs. At 92", f"plots, risk {t5.sa_always.min():.2f}, {t5.sa_always.max():.2f}.", "Circle site, square CCI"], col=S.BLUE)
    return [a, v], s[["map", "q", "n", "pf", "fin", "sa_always"]]


def main():
    S.apply(); fig = S.figure(W, H); ax = S.canvas(fig, 0, 0, W, H)
    for x, s in ((1.5, "Question"), (AX - 10, "Answer"), (VX - 11.4, "Does the answer hold?")):
        ax.text(x, H - 4.2, s, fontsize=S.FS["title"] + .5, fontweight="bold", va="baseline")
    ax.plot([0.5, W - .5], [H - 6.2, H - 6.2], color=S.INK, lw=.7)
    for x in (AX - 11.2, VX - 12.6):
        ax.plot([x, x], [1.5, H - 6.2], color=S.GRID, lw=.6)
    q = (("a", "Interval", "How wide at each selected place?", "11 settings, selected 5%"), ("b", "Project total", "How much carbon can I credit?", "FIA, site maps, highest 1%"),
         ("c", "Funding", "Which villages can I fund?", "DHS, poorest 5% of villages"), ("d", "Region", "Does it hold in my country?", "DHS, 5% quota, 23 countries"), ("e", "Audit size", "How many plots must I visit?", "FIA, 30 plots among 6,000"))
    for i, (l, a, b, c) in enumerate(q):
        y = rows_y(i)
        ax.text(1.0, y + 21.0, l, fontsize=S.FS["panel"], fontweight="bold", va="baseline")
        bubble(ax, 5.0, y + 11.2, 34.2, 11.0, a, b)
        ax.text(5.6, y + 6.8, c, fontsize=S.FS["note"], color=S.INK2, va="baseline")
        if i:
            ax.plot([0.5, W - .5], [y + RH, y + RH], color=S.GRID, lw=.5)
    fn = (row_interval, row_total, row_funding, row_region, row_audit); axes = []
    for i, f in enumerate(fn):
        out = f(fig, ax, i); axes += out[0]
        S.source_data(f"nc_questions_{'abcde'[i]}", out[1])
    ids = [f"{l}{k}" for l in "abcde" for k in (1, 2)]
    S.save(fig, "nc_questions", axes=axes, panel_ids=ids, column_groups=[[f"{l}1" for l in "abcde"], [f"{l}2" for l in "abcde"]])


if __name__ == "__main__":
    main()
