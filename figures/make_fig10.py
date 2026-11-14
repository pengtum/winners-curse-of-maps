"""Fig. 10 | Computation at the scale of operational maps (paper/PACKAGE_BENCHMARK_DESIGN.md).

a  reference sets and JOMI thresholds for a budget against candidates (audit = candidates / 10, selected = candidates / 100): mapselect (measured),
   a careful per-unit implementation and literal swaps (both extrapolated from samples); top-K and quota for mapselect.
b  publication mask against audited units: dense kernel sums and the k-d tree (same selected units).
c  SC-GeoCCP, seconds per selected unit against reference-set size (p = 60): the code behind the manuscript, full primal and dual, and the warm-started active set.
Data: results/fia_db/package/summary_{refsets,mask,lp}.csv (experiments/make_package_benchmark.py).
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_fig10.py
writes figs/nc_fig10_package.{pdf,svg,png}, figs/source_data/nc_fig10{a,b,c}.csv
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402

D = os.path.join(ROOT, "results", "fia_db", "package")
W, H = 183, 62
Y = 12; PH = 42


def label(ax, x, y, s, col, ha="left", va="center"):
    ax.text(x, y, s, color=col, fontsize=S.FS["note"], ha=ha, va=va)


def logs(ax, xlab, ylab=None):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off(); ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    S.clean(ax)


def panel_a(fig):
    r = pd.read_csv(os.path.join(D, "summary_refsets.csv")); r = r[r.n == r.m // 10]; r = r[r.K == r.m // 100]
    ax = S.axes_mm(fig, 15, Y, 50, PH)
    b = r[r.rule == "budget"]; c = b.dropna(subset=["t_competent_est"])
    ax.plot(c.m, c.t_swap_est, color=S.GREY_D, lw=S.LW["line"], ls="--", marker="^", ms=2.6)
    ax.plot(c.m, c.t_competent_est, color=S.ORANGE, lw=S.LW["line"], ls="--", marker="s", ms=2.6)
    ax.plot(b.m, b.ours, color=S.BLUE, lw=S.LW["line"], marker="o", ms=2.8)
    q = r[r.rule == "quota"]; t = r[r.rule == "top_k"]
    ax.plot(q.m, q.ours, color=S.VIOLET, lw=S.LW["line"], marker="D", ms=2.4)
    ax.plot(t.m, t.ours, color=S.AQUA, lw=S.LW["line"], marker="v", ms=2.6)
    logs(ax, "Candidates", "Seconds")
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8]); ax.set_xticklabels(["$10^{3}$", "", "$10^{5}$", "", "$10^{7}$", ""])
    ax.set_yticks([1e-4, 1e-1, 1e2, 1e5, 1e8, 1e11]); ax.set_yticklabels(["$10^{-4}$", "$10^{-1}$", "$10^{2}$", "$10^{5}$", "$10^{8}$", "$10^{11}$"])
    ax.set_xlim(7e2, 1.5e8); ax.set_ylim(5e-6, 1e12)
    label(ax, 9e2, 1e8, "Literal swaps, budget", S.GREY_D); label(ax, 9e2, 1e6, "Per-unit, budget", S.ORANGE)
    label(ax, 9e2, 1e4, "mapselect: budget", S.BLUE); label(ax, 9e2, 1e2, "quota", S.VIOLET); label(ax, 9e2, 1, "top-K", S.AQUA)
    return ax, r


def panel_b(fig):
    m = pd.read_csv(os.path.join(D, "summary_mask.csv")); ax = S.axes_mm(fig, 78, Y, 36, PH)
    for meth, col, mk, lab, xy in (("dense", S.ORANGE, "s", "Dense", (9e2, 300)), ("tree", S.BLUE, "o", "k-d tree", (9e2, 60))):
        g = m[m.method == meth].sort_values("n")
        ax.plot(g.n, g.t_select, color=col, lw=S.LW["line"], marker=mk, ms=2.8); label(ax, *xy, lab, col)
    logs(ax, "Audited units"); ax.set_xticks([1e3, 1e4, 1e5, 1e6]); ax.set_xticklabels(["$10^{3}$", "$10^{4}$", "$10^{5}$", "$10^{6}$"])
    ax.set_yticks([1e-2, 1, 1e2]); ax.set_yticklabels(["0.01", "1", "100"]); ax.set_xlim(7e2, 2.2e6); ax.set_ylim(3e-3, 2e3)
    return ax, m


def panel_c(fig):
    l = pd.read_csv(os.path.join(D, "summary_lp.csv")); l = l[(l.p == 60) & l.t_first.notna()]; ax = S.axes_mm(fig, 127, Y, 52, PH)
    sty = (("published", S.GREY_D, "^", "Code behind this paper", "t_rest"), ("primal", S.ORANGE, "s", "Sparse primal", "t_rest"),
           ("dual", S.VIOLET, "D", "Full dual", "t_rest"), ("active", S.BLUE, "o", "Active set, warm", "t_rest"))
    for form, col, mk, lab, y in sty:
        g = l[l.form == form].sort_values("n"); ax.plot(g.n, g[y], color=col, lw=S.LW["line"], marker=mk, ms=2.8)
    logs(ax, "Reference-set size", "Seconds per selected unit")
    ax.set_xticks([1e2, 1e3, 1e4, 1e5]); ax.set_xticklabels(["$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$"]); ax.set_yticks([1e-3, 1e-2, 0.1, 1, 10]); ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.set_xlim(70, 1.5e5); ax.set_ylim(1e-3, 60)
    label(ax, 90, 40, "Code behind this paper", S.GREY_D); label(ax, 90, 18, "Sparse primal", S.ORANGE)
    label(ax, 90, 8, "Full dual", S.VIOLET); label(ax, 1.3e5, 0.012, "Active set, warm", S.BLUE, ha="right")
    return ax, l


def main():
    S.apply(); fig = S.figure(W, H)
    A, Ta = panel_a(fig); B, Tb = panel_b(fig); C, Tc = panel_c(fig)
    for x, l in ((2.0, "a"), (70.0, "b"), (117.0, "c")):
        S.letter_mm(fig, x, H - 1.5, l)
    for k, T in zip("abc", (Ta, Tb, Tc)):
        S.source_data(f"nc_fig10{k}", T)
    S.save(fig, "nc_fig10_package", axes=[A, B, C], panel_ids=["a", "b", "c"], row_groups=[["a", "b"]])


if __name__ == "__main__":
    main()
