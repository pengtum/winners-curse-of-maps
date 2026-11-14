"""Supplementary Fig. 2 | Coverage under a greedy travel-cost budget (paper/FIGURE_PLAN.md, S2).

Fig. 4a with the budget rule in place of the policy-direction 5% rule, drawn by make_fig4.panel_a: 11 random-forest
settings x 12 published or selective methods, one facet per method family; (n) after a method = settings with
coverage < 0.85. The rule (excp_select.refsets.knapsack_greedy; experiments/run_main.py) sorts the test units by
value / travel cost and keeps the prefix that fits the budget B = 0.05 m x the mean training cost; cost is 1 + the
distance to the training-fold centroid over its median, value is the prediction (DHS: need, the predicted shortfall
from the highest training prediction). Bars: 95% intervals over the 100 random splits. Data: results/main_*,
bl_main_*, scg_main_*_l1fix (rules knapsack, knapsack_need); the table is results/RESULTS_BUDGET_MASK.md.
    PYTHONPATH=<figure-QA helpers> ~/miniforge3/envs/favela/bin/python figures/make_si_budget.py
writes figs/si_fig_budget.{pdf,svg,png}, figs/source_data/si_fig_budget.csv, figs/qa/
"""
import os, sys
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.dont_write_bytecode = True
import style as S                                                     # noqa: E402
import fig_data as D                                                  # noqa: E402
import make_fig4 as F4                                                # noqa: E402

F4.AY0 = 13.0                                   # panel a alone: drop the room that Fig. 4 keeps for panel b
W, H = 183, 13.0 + F4.AH + 14.0


def arrow(nm):
    return "↓" if D.budget_rule(nm) == "knapsack_need" else "↑"


def main():
    S.apply()
    P = {nm: D.pooled(D.runs(nm)) for nm, *_ in S.DATASETS}
    fig = S.figure(W, H)
    A, src = F4.panel_a(fig, P, rule=D.budget_rule, arrow=arrow,
                        key="↑ / ↓  value / need per cost\n(n)  settings below 0.85, of 11")
    S.source_data("si_fig_budget", src)
    S.save(fig, "si_fig_budget", axes=A, panel_ids=["a1", "a2", "a3", "a4"], row_groups=[["a1", "a2", "a3", "a4"]],
           column_groups=[])
    pr = src.pivot_table(index="dataset", columns="method", values="coverage")
    print("settings with coverage < 0.85 (of 11):", (pr < 0.85).sum().to_dict())
    print("median:", pr.median().round(3).to_dict())
    print("selected per split:", src.groupby("dataset").n_selected.mean().round(1).to_dict())
    print("JOMI / RegionCP inf share:", src[src.method.isin(["JOMI", S.SHORT["SC-GeoCCP"]])].pivot_table(
        index="dataset", columns="method", values="inf_share").round(3).to_dict())


if __name__ == "__main__":
    main()
