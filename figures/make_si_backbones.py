"""Supplementary Fig. 9 | Exploratory ResNet settings (paper/FIGURE_PLAN.md, S9; SI section [S30]).

Coverage among the selected units at the 5% policy rule for normalized CP, GeoCP, JOMI and SC-GeoCCP, with the
ResNet predictions of GROUND, a separate study (manuscript under review; read-only, no pickle). The backbones' heads,
losses and the UrbanSARFloods framing were chosen after inspecting the non-training units, so these settings carry no
formal guarantee.

History: from 2026-10-28 this figure also showed that study's DOFA predictions (open markers) and a funding
panel. On 2026-11-08 the DOFA settings became formal (paper/DOFA_FORMAL_DESIGN.md; a linear probe refitted in every
split, Fig. 4a and SI section [S29]), so the exploratory DOFA arm and its funding panel were dropped here; the earlier
version is in paper/archive_2026-11-08/.
Data: figures/fig_data.geofm and experiments/run_geofm.py (results/fm_main_<dataset>_resnet.csv).
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_si_backbones.py
writes figs/si_fig_backbones.{pdf,svg,png}, figs/source_data/si_fig_backbones_a.csv, figs/qa/
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402
import make_fig5 as F5                                                # noqa: E402

W, H = 125, 47
AX = (37.0, 11.0, 58.0, 26.0)                    # axes (mm)
METH = F5.B_METH                                 # normalized CP, GeoCP, JOMI, SC-GeoCCP
ROWS = [d for d in D.FM if D.geofm(d, "resnet") is not None]   # DHS, M3LEO, UrbanSARFloods (no ResNet for USAVars)


def data():
    rows = []
    for ds in ROWS:
        lab, pol = D.FM[ds]
        P = D.pooled(D.geofm(ds, "resnet")[D.COLS])
        for mth in METH:
            r = P.loc[(f"{pol}5", mth)]
            rows.append(dict(dataset=ds, label=lab, backbone="resnet", rule=f"{pol}5", method=S.SHORT[mth], key=mth,
                             coverage=r["cov"], ci_lo=r.lo, ci_hi=r.hi, inf_share=r.inf))
    return pd.DataFrame(rows)


def panel(fig, T):
    ax = S.axes_mm(fig, *AX)
    y = {d: i for i, d in enumerate(ROWS)}
    ax.set_ylim(len(ROWS) - 0.5, -0.5); ax.set_xlim(0.15, 1.02)
    S.hgrid(ax, range(len(ROWS))); S.target(ax)
    off = dict(zip(METH, np.linspace(-0.27, 0.27, len(METH))))
    for mth in METH:
        for d in ROWS:
            r = T[(T.dataset == d) & (T.key == mth)].iloc[0]
            yy = y[d] + off[mth]
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=S.METHODS[mth][1], lw=S.LW["err"], zorder=2,
                    solid_capstyle="butt")
            S.dot(ax, [r.coverage], [yy], mth, s=9, z=3)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([F5.LAB_B[d] + (" ↓" if D.FM[d][1] == "bottom" else " ↑") for d in ROWS], fontsize=S.FS["tick"])
    ax.tick_params(axis="y", length=0); S.clean(ax, left=False)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 0.9]); ax.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "0.9"])
    ax.set_xlabel("Coverage among selected units", labelpad=1.5)
    return ax


def main():
    S.apply()
    T = data()
    fig = S.figure(W, H)
    ax = panel(fig, T)
    hs = [S.handle(m, label=S.SHORT[m]) for m in METH]
    fig.legend(handles=hs, ncol=1, loc="lower left", bbox_to_anchor=((AX[0] + AX[2] + 4.0) / W, AX[1] / H),
               handletextpad=0.3, borderaxespad=0, labelspacing=0.5)
    S.text_mm(fig, 2.0, H - 1.5, "ResNet trained on the task: coverage at the 5% policy rule", fontsize=S.FS["title"],
              fontweight="bold", ha="left", va="top")
    S.text_mm(fig, 2.0, H - 5.6, "Exploratory: heads chosen after inspecting these units.  † Remote-sensing product label.",
              fontsize=S.FS["note"], color=S.INK2, ha="left", va="top")
    S.source_data("si_fig_backbones_a", T)
    S.save(fig, "si_fig_backbones", axes=[ax], panel_ids=["a"], row_groups=[], column_groups=[])
    print(T.pivot_table(index="label", columns="method", values="coverage").round(3).to_string())


if __name__ == "__main__":
    main()
