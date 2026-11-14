"""Summarize the audit-planner runs (paper/AUDIT_PLANNER_DESIGN.md sections 5-6) into results/fia_db/RESULTS_AUDIT_PLANNER.md.
    ~/miniforge3/envs/favela/bin/python experiments/make_audit_planner.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import data as DS                                                     # noqa: E402
from excp_select import planner as P                                  # noqa: E402

D = os.path.join(DS.FIA_RES, "planner")
RISK = 1 / 3


def md(T):
    """Markdown table (pandas' to_markdown needs tabulate, which the environment does not have)."""
    return "\n".join(["| " + " | ".join(map(str, T.columns)) + " |", "|" + "---|" * T.shape[1]] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in T.itertuples(index=False)])


def ci(x, g):
    """Map-clustered 95% half-width of a per-audit quantity (the clustering used for Fig. 7)."""
    m = x.groupby(g).mean()
    return 1.96 * m.std(ddof=1) / np.sqrt(len(m))


def carbon(kind):
    d = pd.read_csv(os.path.join(D, f"carbon_{kind}.csv.gz")); out = []
    base = d[(d.q == 1) & (d.n == 3000)]
    for (q, n), g in d.groupby(["q", "n"], sort=False):
        m = int(round(q * 6000)) - n                                    # candidates in the audited population (design L4)
        pf = P.p_finite(n, m, 30)
        over = {c: (g[c] > g.total) for c in ["vcs_sa_always", "vcs_sa_rule", "vcs_ma_rule", "vcs_ma_always", "exact"]}
        fin = g[g.finite == 1]
        se_pl = g.se_layer * np.sqrt(1 + 30 / P.mean_refset(n, m, 30))
        miss = int((g.finite == 0).sum()); hit = int((g.finite == 1).sum())
        ok = np.isfinite(g.se_sa)
        out.append(dict(map=kind, q=q, n=n, m=m, ER=P.mean_refset(n, m, 30), nR=g.nR.mean(), pf=pf, fin=g.finite.mean(),
                        z=(g.finite.mean() - pf) / np.sqrt(max(pf * (1 - pf), 1e-9) / len(g)), p_binom=binomtest(hit, len(g), min(pf, 1.0)).pvalue if 0 < pf < 1 else np.nan, cov_jomi=g.cov_jomi.mean(),
                        cov_fin=fin.cov_jomi.mean() if len(fin) else np.nan, hw_jomi=fin.hw_jomi.mean() if len(fin) else np.nan,
                        cov_split=g.cov_split.mean(), hw_split=g.hw_split.mean(),
                        sa_always=over["vcs_sa_always"].mean(), sa_ci=ci(over["vcs_sa_always"].astype(float), g.fold),
                        sa_rule=over["vcs_sa_rule"].mean(), ma_rule=over["vcs_ma_rule"].mean(), ma_rule_ci=ci(over["vcs_ma_rule"].astype(float), g.fold),
                        ma_always=over["vcs_ma_always"].mean(), exact=over["exact"].mean(),
                        total=g.total.mean(), credit_sa=g.vcs_sa_always.mean(), credit_ma=g.vcs_ma_rule.mean(), credit_exact=g.exact.mean(),
                        credit_ratio=g.vcs_sa_always.mean() / base.vcs_sa_always.mean(), total_ratio=g.total.mean() / base.total.mean(),
                        share_sa=g.vcs_sa_always.sum() / g.total.sum(), share_exact=g.exact.sum() / g.total.sum(),
                        se_ratio=(se_pl[ok] / g.se_sa[ok]).median() if ok.any() else np.nan,
                        waiver_sa=(1.6449 * g.se_sa[ok] / g.sa[ok] <= 0.10).mean() if ok.any() else np.nan, audit_saving=np.nan))
    return pd.DataFrame(out)


def stored():
    rows = []
    for kind, f, col in [("site", "carbon_site.csv.gz", "se_rf"), ("cci", "product/carbon_product_win.csv.gz", "se_map")]:
        d = pd.read_csv(os.path.join(DS.FIA_RES, f), usecols=["audit", "frac", "K", "nR", col, "se_sa"])
        d = d[np.isfinite(d.se_sa) & (d.se_sa > 0)]
        for (a, fr), g in d.groupby(["audit", "frac"]):
            K = int(g.K.iloc[0]); n = int(a * 6000); m = 6000 - n
            pl = g[col] * np.sqrt(1 + K / P.mean_refset(n, m, K))
            rows.append(dict(map=kind, audit=a, frac=fr, K=K, nR_mean=g.nR.mean(), nR_plan=P.mean_refset(n, m, K), ratio=(pl / g.se_sa).median()))
    return pd.DataFrame(rows)


def floors(name):
    d = pd.read_csv(os.path.join(D, f"floors_{name}.csv")); d = d[d.kind != "intercept"].copy()
    d["law"] = 0.9 - 0.95 * 3000 / ((d.n + 1) * d.M_u)
    d["sel"] = d.M_u * d.redraws
    bse = 2 * np.sqrt(np.maximum(d.cov_scg * (1 - d.cov_scg), 1e-4) / d.sel.clip(lower=1))
    d["violation"] = (d.cov_scg + bse < d.floor) & (d.sel >= 50)
    return d


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    C = pd.concat([carbon("site"), carbon("cci")]); C.to_csv(os.path.join(D, "summary_carbon.csv"), index=False, float_format="%.5g")
    St = stored(); St.to_csv(os.path.join(D, "summary_stored_se.csv"), index=False, float_format="%.5g")
    Fl = pd.concat([floors("poverty"), floors(DS.FIA)]); Fl.to_csv(os.path.join(D, "summary_floors.csv"), index=False, float_format="%.5g")
    L = ["# Audit planner results (`paper/AUDIT_PLANNER_DESIGN.md`)", "", "`experiments/make_audit_planner.py`. Risk is the VCS over-crediting probability; 'always' deducts 0.4307 SE, 'rule' keeps the waiver.", ""]
    L += ["## Carbon project (K = 30 of 6,000)", "", md(C[["map", "q", "n", "m", "ER", "nR", "pf", "fin", "z", "p_binom", "cov_jomi", "cov_fin", "cov_split", "hw_jomi", "hw_split"]].round(3)), "",
          md(C[["map", "q", "n", "sa_always", "sa_ci", "sa_rule", "ma_rule", "ma_rule_ci", "ma_always", "exact", "total_ratio", "credit_ratio", "share_sa", "share_exact", "se_ratio", "waiver_sa"]].round(3)), ""]
    tgt = C[((C.q == 0.05) & (C.n == 92)) | ((C.q == 0.10) & (C.n == 186))]
    L += ["## Judgements", "",
          f"- D1: largest |z| of finite share against the law over arms with pf in (0.01, 0.99): {C.z[(C.pf > 0.01) & (C.pf < 0.99)].abs().max():.2f} (limit 3); over all arms, largest |z| {C.z.abs().max():.2f} (site, whole map, 3,000: 2 infinite of 2,000, expected 0.28), smallest exact binomial p {C.p_binom.min():.3g}.",
          f"- D2: floor violations among regions selected at least 50 times: {int(Fl.violation.sum())} of {int((Fl.sel >= 50).sum())}.",
          f"- D3(a) JOMI coverage on planner arms >= 0.88: {bool((tgt.cov_jomi >= 0.88).all())} (lowest {tgt.cov_jomi.min():.3f}).",
          f"- D3(b) selection-aware always-deducted risk <= 0.383: {bool((tgt.sa_always <= 0.383).all())} (highest {tgt.sa_always.max():.3f}).",
          f"- D3(c) credits >= 0.75 of whole-map 3,000: {bool((tgt.credit_ratio >= 0.75).all())} (lowest {tgt.credit_ratio.min():.3f}).", ""]
    a = St.groupby("map").ratio.median()
    L += ["## D5: planned / realized selection-aware SE, stored runs", "", md(St.round(3)), "",
          f"Median ratio: site {a['site']:.3f}, CCI {a['cci']:.3f}.", ""]
    L += ["## D6: floors and realized coverage (regions selected at least 50 times)", ""]
    for name, g0 in Fl.groupby("dataset", sort=False):
        u = g0[g0.kind == "unit"]; rows = []
        for n, g in u.groupby("n"):
            g = g[g.sel >= 50]; w = g.loc[g.floor.idxmin()]
            rows.append(dict(n=n, worst_floor=g.floor.min(), law_at_worst=w.law, cov_at_worst=w.cov_scg, jomi_at_worst=w.cov_jomi, lowest_cov_scg=g.cov_scg.min(), lowest_cov_jomi=g.cov_jomi.min(),
                             median_floor=g.floor.median(), inf_share=g.inf_share.iloc[0], K=g.K.iloc[0], max_law_err=(g.floor - g.law).abs().max()))
        L += [f"### {name}", "", md(pd.DataFrame(rows).round(3)), ""]
    plan = []
    for lab, m, K in [("1% of map, 3,000 candidates", 3000, 30), ("5% of map, 3,000 candidates", 3000, 150), ("20% of map, 3,000 candidates", 3000, 600), ("1% of map, 30,000 candidates", 30000, 300), ("1% of map, 300,000 candidates", 300000, 3000), ("30 plots of 600,000 candidates", 600000, 30)]:
        plan.append(dict(rule=lab, n50=P.n_for_finite(m, K, 0.5), n90=P.n_for_finite(m, K, 0.9)))
    L += ["## Audit size for a finite interval, fixed candidate pool", "", md(pd.DataFrame(plan)), ""]
    open(os.path.join(DS.FIA_RES, "RESULTS_AUDIT_PLANNER.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
