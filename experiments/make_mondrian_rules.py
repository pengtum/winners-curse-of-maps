"""Tables for experiments B and D of paper/MONDRIAN_ADVANTAGE_DESIGN.md.
    ~/miniforge3/envs/favela/bin/python experiments/make_mondrian_rules.py
reads results/mr_main_<ds>.csv (run_mondrian_rules.py) and results/cfbh_md_<target>.csv (run_cfbh_mondrian.py);
writes results/RESULTS_MONDRIAN_RULES.md, results/RESULTS_MONDRIAN_RULES_SI.md (Tables S24a-c),
results/mondrian_rules_summary.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "figures")); sys.dont_write_bytecode = True
import fig_data as D                                                  # noqa: E402
import style as S                                                     # noqa: E402
from make_mondrian_noise import r3                                    # noqa: E402
from decimal import Decimal, ROUND_HALF_UP                            # noqa: E402


def r2(x):
    return str(Decimal(repr(round(float(x), 9))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

RES = os.path.join(ROOT, "results")
RULES = [("policy", "Map's prediction (5% rule)"), ("upper", "Upper bound, μ̂ ± 1.645σ̂"), ("lower", "Lower bound, μ̂ ∓ 1.645σ̂"),
         ("second map", "Second map (coordinates only)")]
METH = [("SCP |res|", "Split CP"), ("SCP stud", "Normalized CP"), ("Mondrian CP", "Mondrian CP, bins of μ̂"),
        ("Mondrian CP, rule bins 10", "Mondrian CP, 10 bins of the rule's score"),
        ("Mondrian CP, rule bins 20", "Mondrian CP, 20 bins of the rule's score"), ("JOMI stud", "JOMI"), ("JOMI |res|", "JOMI (additive)")]
TARGETS = [("poverty", "DHS wealth (RF, imagery)"), ("poverty_lab", "DHS wealth (RF, no imagery)"),
           ("poverty_resnet", "DHS wealth (ResNet)"), ("poverty_dofa", "DHS wealth (DOFA)")]
DM = [("cfBH", "cfBH (reference set)"), ("cfBH, full calibration", "cfBH, full calibration"),
      ("Mondrian cfBH, 10 bins", "Mondrian cfBH, 10 bins"), ("Mondrian cfBH, 20 bins", "Mondrian cfBH, 20 bins")]


def part_b():
    A = pd.concat([pd.read_csv(os.path.join(RES, f"mr_main_{nm}.csv")) for nm, _, _ in S.DATASETS], ignore_index=True)
    g = A.groupby(["dataset", "rule", "method"])
    P = pd.DataFrame({"cov": g.n_cov.sum() / g.n_sel.sum(), "inf": g.n_inf.sum() / g.n_sel.sum(), "width": g.hw_mean_fin.mean(),
                      "overlap": g.overlap_policy.mean(), "sig_ratio": g.sig_ratio.mean(), "resid": g.resid_mean.mean(),
                      "cor": g.cor_mu_mu2.mean()}).reset_index()
    wj = P[P.method == "JOMI stud"].set_index(["dataset", "rule"]).width
    P["width_ratio"] = P.width / P.set_index(["dataset", "rule"]).index.map(wj).values
    lab = {nm: l for nm, l, _ in S.DATASETS}
    L = ["# JOMI 与 Mondrian CP：规则不对齐（B）与资助层（D）（2026-10-30）", "",
         "设计：`paper/MONDRIAN_ADVANTAGE_DESIGN.md`（跑前写定）。B：`experiments/run_mondrian_rules.py`（与 run_main 同种子；"
         "政策规则下 JOMI、normalized CP 与 Mondrian CP 逐划分计数核对一致）；D：`experiments/run_cfbh_mondrian.py`（参考集 cfBH 逐划分复现）。"
         "表：`experiments/make_mondrian_rules.py`。覆盖为 100 次划分合并；+∞ 计为覆盖。", ""]
    for rn, rl in RULES:
        L += [f"## B　{rl}：被选单元上的覆盖（+∞ 比例；有限半宽与 JOMI 之比）", "",
              "| 设定 | 与政策规则重合 | " + " | ".join(m for _, m in METH) + " |", "|---|---|" + "---|" * len(METH)]
        for nm, l, _ in S.DATASETS:
            x = P[(P.dataset == nm) & (P.rule == rn)].set_index("method")
            L.append(f"| {l} | {r2(x.loc['JOMI stud', 'overlap'])} | " + " | ".join(
                f"{r3(x.loc[m, 'cov'])} ({r2(x.loc[m, 'inf'])}; {r2(x.loc[m, 'width_ratio'])})" for m, _ in METH) + " |")
        y = P[P.rule == rn]
        L.append("| **低于 0.85 / 高于 0.95 的设定数** | | " + " | ".join(
            f"**{int((y[y.method == m]['cov'] < 0.85).sum())} / {int((y[y.method == m]['cov'] > 0.95).sum())}**" for m, _ in METH) + " |")
        L.append("")
    # SI Table S24a
    T = ["Table S24a. Selection rules that rank something other than the map's prediction (5% of the test units, policy "
         "direction; 11 settings, 100 splits): settings with coverage among the selected units below 0.85, in parentheses "
         "the lowest coverage, and the number above 0.95.", "",
         "| Rule (overlap with the 5% rule) | " + " | ".join(m for _, m in METH) + " |", "|---|" + "---|" * len(METH)]
    out = []
    for rn, rl in RULES:
        y = P[P.rule == rn]; ov = y[y.method == "JOMI stud"].overlap
        cells = []
        for m, _ in METH:
            c = y[y.method == m]["cov"]
            cells.append(f"{int((c < 0.85).sum())} ({r3(c.min())}); {int((c > 0.95).sum())}")
            out.append(dict(part="B", rule=rn, method=m, below085=int((c < 0.85).sum()), min_cov=c.min(),
                            above095=int((c > 0.95).sum()), max_cov=c.max(),
                            median_width_ratio=float(y[y.method == m].width_ratio.median()), max_inf=float(y[y.method == m].inf.max())))
        T.append(f"| {rl} ({r2(ov.min())}–{r2(ov.max())}) | " + " | ".join(cells) + " |")
    T += ["", "Table S24b. Median over the 11 settings of the mean finite half-width relative to JOMI, and the largest share "
          "of infinite intervals.", "", "| Rule | " + " | ".join(m for _, m in METH) + " |", "|---|" + "---|" * len(METH)]
    for rn, rl in RULES:
        y = P[P.rule == rn]
        T.append(f"| {rl} | " + " | ".join(f"{r2(y[y.method == m].width_ratio.median())} ({r2(y[y.method == m].inf.max())})"
                                          for m, _ in METH) + " |")
    return L, T, out, P


def part_d():
    fs = [os.path.join(RES, f"cfbh_md_{t}.csv") for t, _ in TARGETS]
    if not all(os.path.exists(f) for f in fs):
        return ["## D　尚未运行", ""], ["Table S24c. (not yet run)"], []
    A = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    L = ["## D　资助层：参考集 cfBH 与 Mondrian p 值（FDR；* 表示 FDR − q > 1.96 × 标准误；括号内为合格村被资助的比例；校准集大小 ÷ 参考集）", ""]
    T = ["Table S24c. Funding inside the selected villages with BH at level q: false discovery rate (mean over 100 splits; "
         "* exceeds q by more than 1.96 standard errors), and in parentheses the share of eligible selected villages funded. "
         "The last column gives the calibration set of Mondrian cfBH with ten bins of μ̂ cut over all countries (not the within-country "
         "bins) relative to the reference set; columns marked (post hoc) were added after the adversarial check.", ""]
    out = []
    for rule in ("bottom 5%", "quota"):
        meths = ([("cfBH", "cfBH, mixed reference set (heuristic)")] + DM[1:] +
                 [("Mondrian cfBH, within-country 10 bins", "Mondrian cfBH, within-country 10 bins (post hoc)"),
                  ("Mondrian cfBH, within-country 20 bins", "Mondrian cfBH, within-country 20 bins (post hoc)"),
                  ("Mondrian cfBH, country", "Mondrian cfBH, country"), ("cfBH (group-wise)", "cfBH, same-country reference set")]
                 if rule == "quota" else DM)
        hdr = "| Target | q | " + " | ".join(m for _, m in meths) + " | Mondrian triage (U < c) | Bin / reference set |"
        T += [f"*{'Predicted-poorest 5%' if rule == 'bottom 5%' else 'Country quota (5%)'}*", "", hdr, "|---|---|" + "---|" * (len(meths) + 2)]
        L += [f"### {rule}", "", hdr.replace("Target", "目标"), "|---|---|" + "---|" * (len(meths) + 2)]
        for t, tl in TARGETS:
            for q in (0.05, 0.10):
                cells = []
                for m, _ in meths:
                    x = A[(A.target == t) & (A.rule == rule) & (A.method == f"{m} q={q:.2f}")]
                    fdr, se = x.fdp.mean(), x.fdp.std(ddof=1) / np.sqrt(len(x))
                    exc = fdr - q > 1.96 * se
                    reach = x.n_fund_elig.sum() / max(x.n_elig.sum(), 1)
                    cells.append(f"{r3(fdr)}{'*' if exc else ''} ({r2(reach)})")
                    out.append(dict(part="D", target=t, rule=rule, method=m, q=q, fdr=fdr, se=se, exceeds=bool(exc), reach=reach,
                                    cal_size=x.cal_size.mean(), ref_size=x.ref_size.mean(), funded=x.n_fund.mean()))
                x = A[(A.target == t) & (A.rule == rule) & (A.method == "Mondrian triage (U < c)")]
                tri = f"{r3(x.fdp.mean())} ({r2(x.n_fund_elig.sum() / max(x.n_elig.sum(), 1))})"
                z = A[(A.target == t) & (A.rule == rule) & (A.method == f"Mondrian cfBH, 10 bins q={q:.2f}")]
                row = f"| {tl} | {q:.2f} | " + " | ".join(cells) + f" | {tri if q == 0.05 else ''} | {z.cal_size.mean() / z.ref_size.mean():.1f} |"
                T.append(row); L.append(row)
        T.append(""); L.append("")
    return L, T, out


def main():
    Lb, Tb, ob, _ = part_b()
    Ld, Td, od = part_d()
    open(os.path.join(RES, "RESULTS_MONDRIAN_RULES.md"), "w", encoding="utf-8").write("\n".join(Lb + Ld) + "\n")
    open(os.path.join(RES, "RESULTS_MONDRIAN_RULES_SI.md"), "w", encoding="utf-8").write("\n".join(Tb + [""] + Td) + "\n")
    pd.DataFrame(ob + od).to_csv(os.path.join(RES, "mondrian_rules_summary.csv"), index=False, float_format="%.5f")
    print("\n".join(Tb + [""] + Td))


if __name__ == "__main__":
    main()
