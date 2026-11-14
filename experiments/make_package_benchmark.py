"""Tables of the runtime benchmark (paper/PACKAGE_BENCHMARK_DESIGN.md) from results/fia_db/package/benchmark.jsonl and parity.csv.
    ~/miniforge3/envs/favela/bin/python experiments/make_package_benchmark.py
writes results/fia_db/RESULTS_PACKAGE_BENCHMARK.md and results/fia_db/package/summary_{refsets,mask,lp}.csv
"""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "fia_db", "package")


def md(T, fmt="%.3g"):
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return "–" if np.isnan(x) else fmt % x
        return str(x)
    return "\n".join(["| " + " | ".join(map(str, T.columns)) + " |", "|" + "---|" * T.shape[1]] +
                     ["| " + " | ".join(cell(x) for x in r) + " |" for r in T.itertuples(index=False)])


def lp_gaps(l):
    rows = []
    for (n, p), g in l.groupby(["n", "p"]):
        o = {x.form: (json.loads(x.obj), json.loads(x.q)) for x in g.itertuples() if isinstance(x.obj, str) and x.obj != "[]"}
        if "active" in o and "dual" in o:
            k = min(len(o["active"][0]), len(o["dual"][0]))
            rows.append(dict(n=n, p=p, units=k, obj_gap=max(abs(a - b) for a, b in zip(o["active"][0][:k], o["dual"][0][:k])),
                             q_gap=max(abs(a - b) for a, b in zip(o["active"][1][:k], o["dual"][1][:k]))))
    return pd.DataFrame(rows)


def main():
    d = pd.DataFrame([json.loads(x) for x in open(os.path.join(D, "benchmark.jsonl"))])
    r = d[d.kind == "refsets"].copy(); r["ours"] = r.t_select + r.t_jomi
    r["x_competent"] = r.t_competent_est / r.ours; r["x_swap"] = r.t_swap_est / r.ours
    cols = ["rule", "m", "n", "K", "R_mean", "finite", "t_select", "t_jomi", "ours", "t_competent_est", "x_competent", "t_swap_est", "x_swap", "rss_gb_ours"]
    r[cols].to_csv(os.path.join(D, "summary_refsets.csv"), index=False, float_format="%.5g")
    m = d[d.kind == "mask"]; m = m[["n", "method", "selected", "t_select", "t_refs_est", "checksum", "rss_gb_ours"]]
    m.to_csv(os.path.join(D, "summary_mask.csv"), index=False, float_format="%.5g")
    l = d[d.kind == "lp"]
    lt = l[["n", "p", "form", "units_done", "t_first", "t_rest", "rss_gb_ours"]].copy(); lt["failed"] = l.get("failed")
    lt.to_csv(os.path.join(D, "summary_lp.csv"), index=False, float_format="%.5g")
    gaps = lp_gaps(l)
    p = pd.read_csv(os.path.join(D, "parity.csv"))
    L = ["# mapselect runtime benchmark (`paper/PACKAGE_BENCHMARK_DESIGN.md`)", "",
         "`experiments/make_package_benchmark.py`. One core of an Apple M6 with 32 GB. Times in seconds. `t_competent_est` and `t_swap_est` are extrapolated from a sample of units or swaps, except where the sample is the whole; the competent comparator's precomputation is not charged to it.", "",
         "## Reference sets and JOMI thresholds", "", md(r[cols]), "",
         "## Publication mask", "", md(m), "",
         "## SC-GeoCCP, per unit (`t_first` cold, `t_rest` warm for the active set)", "", md(lt), "",
         "Active set against the full dual, same units:", "", md(gaps), "",
         f"Largest objective gap {gaps.obj_gap.max():.3g}; largest threshold gap {gaps.q_gap.max():.3g}.", ""]
    for k, lab in (("lp_full", "End to end, all selected units"), ("real", "The manuscript's quota, one split, old code against package"), ("totals", "Exact envelope"), ("planner", "Planner at 10^8 candidates"), ("demo", "ESA CCI pixels as candidates, FIA plots as audit: times only")):
        x = d[d.kind == k].dropna(axis=1, how="all").drop(columns=["cell", "kind", "compare"], errors="ignore")
        L += [f"## {lab}", "", md(x, "%.5g"), ""]
    L += ["## Parity with the code behind the results (`experiments/check_package_parity.py`)", "",
          f"{len(p)} dataset-rule pairs. Identical selected units: {bool(p.same_selected.all())}. Identical reference sets: {bool(p.same_reference.all())}. "
          f"Largest JOMI threshold difference: {p.jomi_max_diff.max():.3g}. Same infinite intervals: {bool(p.jomi_same_inf.all())}. "
          f"SC-GeoCCP on up to 40 units per pair: primal form against the published code, largest threshold difference {p.primal_q_max_diff.max():.3g}; "
          f"dual against primal, largest objective gap {p.objective_max_gap.max():.3g}, largest threshold difference {p.dual_q_max_diff.max():.3g} (the programme has several minimizers there).", ""]
    open(os.path.join(ROOT, "results", "fia_db", "RESULTS_PACKAGE_BENCHMARK.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[-30:])[:3500])
    print(gaps.to_string())


if __name__ == "__main__":
    main()
