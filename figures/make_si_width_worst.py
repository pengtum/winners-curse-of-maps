"""Supplementary Fig. 23 | Width against the worst group under the quota rules (paper/MONDRIAN_NOISE_DESIGN.md §3; [S23]).

a  DHS wealth, country quota (the poorest-predicted 5% of the test villages in each country)
b  FIA biomass, state quota (the highest-predicted 5% in each state)
x: mean finite half-width among the selected units relative to JOMI (averaged over 100 splits); y: coverage in the worst
country / state (groups selected at least 50 times). Methods whose intervals are infinite for part of the
selected units are annotated with that share (their width counts finite intervals only). Dashed line 0.90; solid line the sampling-noise limit (independent binomial,
fig_data.noise_limit). Data: results/main_*, bl_*, scg_*_l1fix, md_* (experiments/run_mondrian.py).
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_si_width_worst.py
writes figs/si_fig_width_worst.{pdf,svg,png}, figs/source_data/si_fig_width_worst.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402

W, H = 183, 76
METH = ["SCP |res|", "SCP stud", "CQR", "Mondrian CP", "GeoCP", "LSCP", "RLCP", "Regression kriging", "QRF",
        "Bootstrap ensemble + residual", "JOMI stud", "JOMI|state stud", "SC-GeoCCP"]
PANELS = [("a", "poverty", "DHS wealth, country quota (poorest 5%)", (0.84, 1.96), (0.72, 0.95)),
          ("b", D.DS.FIA, "FIA biomass, state quota (highest 5%)", (0.15, 1.5), (0.0, 1.0))]
AX = ((16.0, 22.0, 68.0, 42.0), (106.0, 22.0, 68.0, 42.0))
LABEL = dict(S.SHORT, **{"JOMI|state stud": "JOMI per group"})          # countries in a, states in b


def data():
    rows = []
    for _, nm, _, _, _ in PANELS:
        r = D.quota_rule(nm)
        md = pd.read_csv(os.path.join(D.RES, f"md_main_{nm}.csv"))
        P = pd.concat([D.pooled(D.runs(nm)), D.pooled(md[["rule", "method", "rep", "n_sel", "n_cov", "n_inf", "hw_mean_fin"]])])
        g = D.groups(nm, "state", main=("SCP |res|",))
        mg = pd.read_csv(os.path.join(D.RES, f"md_groups_{nm}.csv")); g = pd.concat([g, mg[mg.gtype == "state"]], ignore_index=True)
        wj = P.loc[(r, "JOMI stud"), "width"]
        for m in METH:
            if (r, m) not in P.index:
                continue
            w, wi, ng, _ = D.worst(g, r, m)
            rows.append(dict(dataset=nm, rule=r, method=LABEL[m], key=m, width_ratio=P.loc[(r, m), "width"] / wj,
                             worst=w, inf_in_worst=wi, inf_share=P.loc[(r, m), "inf"], groups=ng,
                             noise_limit=D.noise_limit(nm, "state")))
    return pd.DataFrame(rows)


def main():
    S.apply()
    T = data()
    fig = S.figure(W, H); A = []
    for (l, nm, title, xl, yl), box in zip(PANELS, AX):
        ax = S.axes_mm(fig, *box); A.append(ax)
        x = T[T.dataset == nm]
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        S.target(ax, orient="h"); ax.axhline(x.noise_limit.iloc[0], color=S.INK2, lw=S.LW["ref"], zorder=1)
        ax.axvline(1.0, color=S.GRID, lw=S.LW["grid"], zorder=0)
        for _, r in x.iterrows():
            S.dot(ax, [r.width_ratio], [r.worst], r.key, s=16, z=3)
            if r.inf_share > 0.01:                       # finite-only width: say how many intervals were infinite
                ax.annotate(f"+∞ {r.inf_share:.0%}", (r.width_ratio, r.worst), xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=S.FS["note"], color=S.INK2)
        ax.set_xlabel("Mean finite half-width (× JOMI)"); ax.set_ylabel("Coverage in the worst group")
        ax.text(xl[0] + 0.015 * (xl[1] - xl[0]), x.noise_limit.iloc[0] - 0.012 * (yl[1] - yl[0]), "noise limit", ha="left", va="top",
                fontsize=S.FS["note"], color=S.INK2)
        S.letter_mm(fig, box[0] - 13.0, box[1] + box[3] + 9.0, l)
        S.text_mm(fig, box[0] - 8.5, box[1] + box[3] + 8.65, title, fontsize=S.FS["title"], fontweight="bold", ha="left", va="top")
    hs = [S.handle(m, label=LABEL[m]) for m in METH]
    fig.legend(handles=hs, ncol=7, loc="lower left", bbox_to_anchor=(3.0 / W, 1.5 / H), borderaxespad=0, handletextpad=0.25,
               columnspacing=0.9, labelspacing=0.35)
    S.source_data("si_fig_width_worst", T)
    S.save(fig, "si_fig_width_worst", axes=A, panel_ids=["a", "b"], row_groups=[["a", "b"]], column_groups=[])
    print(T[["dataset", "method", "width_ratio", "worst", "inf_share", "noise_limit"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
