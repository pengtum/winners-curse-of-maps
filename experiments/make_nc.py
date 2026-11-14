"""Nature Communications tables / figures: published baselines (WP1) + GeoFM backbones (WP9).

Merges results/main_<ds>.csv (SCP, BY, JOMI), results/bl_main_<ds>.csv (published baselines),
results/scg_main_<ds>_l1fix.csv (SC-GeoCCP, l1 = 0.95) and results/fm_main_<combo>.csv.
    ~/miniforge3/envs/favela/bin/python experiments/make_nc.py
writes results/RESULTS_NC.md, figs/nc_fig2_audit.png, figs/nc_fig3_backbone.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import data as DS  # noqa: E402
from make_figs import md, INK, INK2, GRID  # noqa: E402
from style import STY  # noqa: E402  (method colour / marker, figures/style.py)

RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
ALPHA = 0.1
MAIN_METH = ["SCP |res|", "SCP stud", "BY stud", "JOMI stud", "JOMI|state stud"]
BL_METH = ["CQR", "GeoCP", "Regression kriging", "QRF", "Bootstrap ensemble + residual", "Bootstrap ensemble only",
           "LSCP", "RLCP"]
ORDER = ["SCP |res|", "SCP stud", "CQR", "GeoCP", "Regression kriging", "QRF", "Bootstrap ensemble + residual",
         "BY stud", "JOMI stud", "SC-GeoCCP"]
SHORT = {"SCP |res|": "Split CP", "SCP stud": "Normalized CP", "CQR": "CQR", "GeoCP": "GeoCP",
         "Regression kriging": "Reg. kriging", "QRF": "QRF", "Bootstrap ensemble + residual": "Bootstrap",
         "BY stud": "BY", "JOMI stud": "JOMI", "SC-GeoCCP": "SC-GeoCCP (ours)"}
QUOTA = {"poverty": "quota_bottom", "poverty_lab": "quota_bottom"}


def policy_rule(nm):
    return f"{DS.POLICY[nm]}5"


def load(nm):
    parts = []
    f = os.path.join(RES, f"main_{nm}.csv")
    if os.path.exists(f):
        x = pd.read_csv(f).rename(columns={"hw_mean": "hw_mean_fin"}); parts.append(x[x.method.isin(MAIN_METH)])
    f = os.path.join(RES, f"bl_main_{nm}.csv")
    if os.path.exists(f):
        parts.append(pd.read_csv(f))
    f = os.path.join(RES, f"scg_main_{nm}_l1fix.csv")
    if os.path.exists(f):
        x = pd.read_csv(f); x["method"] = "SC-GeoCCP"; parts.append(x)
    cols = ["rule", "method", "rep", "n_sel", "n_cov", "n_inf", "hw_med", "hw_mean_fin"]
    return pd.concat([p[cols] for p in parts], ignore_index=True) if parts else None


def groups(nm, gtype="state"):
    parts = []
    f = os.path.join(RES, f"bl_groups_{nm}.csv")
    if os.path.exists(f):
        parts.append(pd.read_csv(f))
    f = os.path.join(RES, f"scg_groups_{nm}_l1fix.csv")
    if os.path.exists(f):
        x = pd.read_csv(f); x["method"] = "SC-GeoCCP"; parts.append(x)
    f = os.path.join(RES, f"scg_groups_{nm}.csv")          # SCP stud / JOMI / JOMI|state per group
    if os.path.exists(f):
        x = pd.read_csv(f); parts.append(x[x.method.isin(["SCP stud", "JOMI stud", "JOMI|state stud"])])
    g = pd.concat(parts, ignore_index=True)
    return g[g.gtype == gtype]


def worst(g, rule, mth, nmin=50):
    x = g[(g.rule == rule) & (g.method == mth) & (g.n >= nmin)]
    if x.empty:
        return np.nan, np.nan
    cv = x.c / x.n; k = cv.values.argmin()
    return cv.min(), (x.inf / x.n).values[k]


def main():
    names = [n for n in DS.DISPLAY if os.path.exists(os.path.join(RES, f"bl_main_{n}.csv"))]
    L = ["# NC 结果：已发表基线（WP1）+ GeoFM 骨干（WP9）（自动生成：experiments/make_nc.py）\n\n",
         "α = 0.1；与主实验相同的 100 次随机划分和被选集合。SC-GeoCCP 取 l1 = 0.95。"
         "GeoCP 为论文版（Lou et al. 2025，无 +∞ atom）。* 为 PLACES 模型输出。\n\n"]
    # ---- Table N1: selected-unit coverage, policy 5% and policy quota
    rows, P = [], {}
    for nm in names:
        df = load(nm)
        for rn in [policy_rule(nm), QUOTA.get(nm, "quota")]:
            d = df[df.rule == rn]; row = {"data": DS.LABEL[nm], "rule": rn}
            for mth in ORDER:
                x = d[d.method == mth]
                row[SHORT[mth]] = x.n_cov.sum() / x.n_sel.sum() if len(x) else np.nan
                P[(nm, rn, mth)] = row[SHORT[mth]]
            j = d[d.method == "JOMI stud"].hw_mean_fin.mean()
            for mth in ["QRF", "Bootstrap ensemble + residual", "Regression kriging", "SC-GeoCCP"]:
                x = d[d.method == mth]
                row[f"w {SHORT[mth]}/JOMI"] = x.hw_mean_fin.mean() / j if len(x) else np.nan
            rows.append(row)
    T1 = pd.DataFrame(rows)
    L += ["## 表 N1　被选单元覆盖（政策方向 5% 与政策方向配额）与宽度比（相对 JOMI）\n\n", md(T1), "\n"]
    pr = T1[T1.rule.str.endswith("5")]
    cnt = {SHORT[m]: int((pr[SHORT[m]] < 0.85).sum()) for m in ORDER}
    L += [f"政策方向 5% 选择下，覆盖 < 0.85 的数据集数（共 {len(pr)} 个）："
          + "；".join(f"{k} {v}" for k, v in cnt.items()) + "。\n\n"]
    # ---- Table N2: worst state under quota
    rows = []
    for nm in names:
        g = groups(nm); rn = QUOTA.get(nm, "quota"); row = {"data": DS.LABEL[nm], "rule": rn}
        for mth in ["SCP stud", "CQR", "GeoCP", "Regression kriging", "QRF", "Bootstrap ensemble + residual",
                    "JOMI stud", "JOMI|state stud", "SC-GeoCCP"]:
            w, wi = worst(g, rn, mth)
            row[SHORT.get(mth, "JOMI|state")] = w
            if mth == "JOMI|state stud":
                row["JOMI|state +inf (worst)"] = wi
        rows.append(row)
    T2 = pd.DataFrame(rows)
    L += ["## 表 N2　配额规则下的最差州/国覆盖（合并 100 次划分，每组被选 ≥ 50 次）\n\n", md(T2), "\n"]
    # ---- Table N3: GeoFM backbones
    fm = [f for f in ["poverty_resnet", "poverty_dofa", "m3leo_resnet", "m3leo_dofa", "floods_resnet", "floods_dofa",
                      "usavars_dofa"] if os.path.exists(os.path.join(RES, f"fm_main_{f}.csv"))]
    rows, FM = [], {}
    for cb in fm:
        d = pd.read_csv(os.path.join(RES, f"fm_main_{cb}.csv")); g = pd.read_csv(os.path.join(RES, f"fm_groups_{cb}.csv"))
        g = g[g.gtype == "region"]
        pol = d.policy.iloc[0]; rn = f"{pol}5"
        row = {"data": d.dataset.iloc[0], "backbone": d.backbone.iloc[0], "rule": rn}
        for mth in ["SCP |res|", "SCP stud", "GeoCP", "BY stud", "JOMI stud", "SC-GeoCCP"]:
            x = d[(d.rule == rn) & (d.method == mth)]
            row[SHORT[mth]] = x.n_cov.sum() / x.n_sel.sum() if len(x) else np.nan
            FM[(cb, mth)] = row[SHORT[mth]]
        for mth in ["SCP stud", "GeoCP", "JOMI stud", "JOMI|region stud", "SC-GeoCCP"]:
            w, wi = worst(g, "quota", mth)
            row[f"worst region {SHORT.get(mth, 'JOMI|region')}"] = w
            if mth == "JOMI|region stud":
                row["+inf JOMI|region (worst)"] = wi
        rows.append(row)
    if rows:
        T3 = pd.DataFrame(rows)
        L += ["## 表 N3　骨干不变性：ResNet vs 冻结 DOFA（GeoFM）——政策方向 5% 的被选覆盖与配额下最差区域\n\n",
              "标签说明：M3LEO 生物量 = ESA-CCI 产品，USAVars 树冠 = 遥感产品（非实测）；PovertyMap = DHS 调查；"
              "UrbanSARFloods = 标注洪水掩膜。\n\n", md(T3), "\n"]
    L += table_scorefns(names)
    open(os.path.join(RES, "RESULTS_NC.md"), "w").write("".join(L))
    fig_audit(P, names)
    fig_tradeoff(names)
    if rows:
        fig_backbone(FM, fm)
    print("".join(L)[:6000])


SF = ["QRF (raw)", "Bootstrap (raw)", "split CQR-QRF", "JOMI CQR-QRF", "SC-GeoCCP CQR-QRF",
      "split norm-boot", "JOMI norm-boot", "SC-GeoCCP norm-boot"]


def load_sf(nm):
    f = os.path.join(RES, f"sf_main_{nm}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def table_scorefns(names):
    """Table N4: QRF / bootstrap as score functions; width relative to JOMI with the RF+sigma score."""
    rows = []
    for nm in names:
        sf = load_sf(nm)
        if sf is None:
            continue
        base = load(nm)
        gs = pd.read_csv(os.path.join(RES, f"sf_groups_{nm}.csv")); gs = gs[gs.gtype == "state"]
        g0 = groups(nm)
        for rn in [policy_rule(nm), QUOTA.get(nm, "quota")]:
            jw = base[(base.rule == rn) & (base.method == "JOMI stud")].hw_mean_fin.mean()
            row = {"data": DS.LABEL[nm], "rule": rn}
            for mth in ["JOMI stud", "SC-GeoCCP"]:
                x = base[(base.rule == rn) & (base.method == mth)]
                row[f"cov {SHORT[mth]}"] = x.n_cov.sum() / x.n_sel.sum()
                row[f"w {SHORT[mth]}"] = x.hw_mean_fin.mean() / jw
                if rn.startswith("quota"):
                    row[f"worst {SHORT[mth]}"] = worst(g0, rn, mth)[0]
            for mth in SF:
                x = sf[(sf.rule == rn) & (sf.method == mth)]
                row[f"cov {mth}"] = x.n_cov.sum() / x.n_sel.sum()
                row[f"w {mth}"] = x.hw_mean_fin.mean() / jw
                if rn.startswith("quota"):
                    row[f"worst {mth}"] = worst(gs, rn, mth)[0]
            rows.append(row)
    if not rows:
        return []
    T = pd.DataFrame(rows)
    cov_cols = ["data", "rule"] + [c for c in T.columns if c.startswith("cov ")]
    w_cols = ["data", "rule"] + [c for c in T.columns if c.startswith("w ")]
    q = T[T.rule.str.startswith("quota")]
    wc_cols = ["data", "rule"] + [c for c in T.columns if c.startswith("worst ")]
    pol = T[~T.rule.str.startswith("quota")]
    summ = []
    for mth in ["JOMI CQR-QRF", "SC-GeoCCP CQR-QRF", "JOMI norm-boot", "SC-GeoCCP norm-boot"]:
        summ.append(f"{mth}：政策方向 5% 覆盖最小 {pol['cov ' + mth].min():.3f}，宽度（相对 JOMI stud）中位 "
                    f"{pol['w ' + mth].median():.2f}（范围 {pol['w ' + mth].min():.2f}–{pol['w ' + mth].max():.2f}）")
    for mth in ["QRF (raw)", "Bootstrap (raw)"]:
        summ.append(f"{mth}：覆盖最小 {pol['cov ' + mth].min():.3f}，宽度中位 {pol['w ' + mth].median():.2f}")
    summ.append(f"split CQR-QRF（边际校准）：政策方向 5% 覆盖最小 {pol['cov split CQR-QRF'].min():.3f}")
    return ["## 表 N4　QRF / bootstrap 作为打分函数（CQR-QRF：QRF 分位数；norm-boot：bootstrap 中心与尺度）\n\n",
            "宽度为相对\"JOMI + RF 学生化分数\"的倍数。split = 边际校准（无选择条件保证）。\n\n",
            "**被选处覆盖**\n\n", md(T[cov_cols]), "\n**宽度比**\n\n", md(T[w_cols]), "\n",
            "**配额规则下的最差州/国**\n\n", md(q[wc_cols]), "\n", "；\n".join(summ) + "。\n\n"]


def fig_audit(P, names):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(names))[::-1]; off = np.linspace(-0.36, 0.36, len(ORDER))
    ax.axvline(1 - ALPHA, color=INK2, lw=1, ls="--", zorder=0)
    for k, mth in enumerate(ORDER):
        c, mk, fill = STY[mth]
        v = [P.get((nm, policy_rule(nm), mth), np.nan) for nm in names]
        ax.scatter(v, y + off[k], s=22, marker=mk, color=c if fill else "none", edgecolor=c, lw=1.1,
                   label=SHORT[mth], zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{DS.LABEL[n]} ({DS.POLICY[n]} 5%)" for n in names], fontsize=8)
    ax.set_xlabel("coverage among selected units  P(Y ∈ C | selected)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=5, frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01))
    fig.savefig(os.path.join(FIG, "nc_fig2_audit.png")); plt.close(fig)


def fig_tradeoff(names):
    """Coverage among selected units vs interval width relative to JOMI (policy 5% rule), one
    point per method x dataset; method medians drawn large. Guarantee status in the legend."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(1 - ALPHA, color=INK2, lw=1, ls="--", zorder=0); ax.axvline(1, color=GRID, lw=1, zorder=0)
    for mth in [m for m in ORDER if m not in ("BY stud",)]:
        c, mk, fill = STY[mth]; xs, ys = [], []
        for nm in names:
            d = load(nm); d = d[d.rule == policy_rule(nm)]
            j = d[d.method == "JOMI stud"].hw_mean_fin.mean(); x = d[d.method == mth]
            if len(x):
                xs.append(x.hw_mean_fin.mean() / j); ys.append(x.n_cov.sum() / x.n_sel.sum())
        ax.scatter(xs, ys, s=14, marker=mk, color=c if fill else "none", edgecolor=c, lw=0.8, alpha=0.55, zorder=2)
        ax.scatter([np.median(xs)], [np.median(ys)], s=70, marker=mk, color=c if fill else "white", edgecolor=c,
                   lw=1.6, zorder=4, label=SHORT[mth])
    for mth, c, mk in [("JOMI CQR-QRF", "#2a78d6", "^"), ("SC-GeoCCP CQR-QRF", "#6250d6", "*")]:
        xs, ys = [], []
        for nm in names:
            sf = load_sf(nm)
            if sf is None:
                continue
            d = load(nm); d = d[d.rule == policy_rule(nm)]; j = d[d.method == "JOMI stud"].hw_mean_fin.mean()
            x = sf[(sf.rule == policy_rule(nm)) & (sf.method == mth)]
            if len(x):
                xs.append(x.hw_mean_fin.mean() / j); ys.append(x.n_cov.sum() / x.n_sel.sum())
        if xs:
            ax.scatter(xs, ys, s=14, marker=mk, color="none", edgecolor=c, lw=0.8, alpha=0.55, zorder=2)
            ax.scatter([np.median(xs)], [np.median(ys)], s=80, marker=mk, color="white", edgecolor=c, lw=1.6,
                       zorder=4, label=mth.replace("CQR-QRF", "+ QRF score"))
    ax.set_xscale("log"); ax.set_xlabel("mean interval width relative to JOMI (log scale)")
    ax.set_xticks([0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.25×", "0.5×", "1×", "2×"]); ax.minorticks_off()
    ax.set_ylabel("coverage among selected units"); ax.set_ylim(0.15, 1.02)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, fontsize=7, loc="lower right")
    fig.savefig(os.path.join(FIG, "nc_fig_tradeoff.png")); plt.close(fig)


def fig_backbone(FM, fm):
    ds = []
    for cb in fm:
        d = cb.rsplit("_", 1)[0]
        if d not in ds:
            ds.append(d)
    meths = ["SCP stud", "GeoCP", "JOMI stud", "SC-GeoCCP"]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(ds))[::-1]; off = np.linspace(-0.3, 0.3, len(meths))
    ax.axvline(1 - ALPHA, color=INK2, lw=1, ls="--", zorder=0)
    lab = {"poverty": "DHS wealth (bottom 5%)", "m3leo": "Biomass, ESA-CCI (top 5%)",
           "floods": "Flood fraction (top 5%)", "usavars": "Tree cover (bottom 5%)"}
    for k, mth in enumerate(meths):
        c, mk, _ = STY[mth]
        for i, dsn in enumerate(ds):
            r, f = FM.get((f"{dsn}_resnet", mth), np.nan), FM.get((f"{dsn}_dofa", mth), np.nan)
            yy = y[i] + off[k]
            if np.isfinite(r) and np.isfinite(f):
                ax.plot([r, f], [yy, yy], color=c, lw=1, alpha=0.6, zorder=2)
            ax.scatter([r], [yy], s=24, marker=mk, color=c, edgecolor=c, zorder=3,
                       label=f"{SHORT[mth]}" if i == 0 else None)
            ax.scatter([f], [yy], s=24, marker=mk, color="none", edgecolor=c, lw=1.1, zorder=3)
    ax.scatter([], [], s=24, marker="o", color=INK2, label="ResNet (filled)")
    ax.scatter([], [], s=24, marker="o", color="none", edgecolor=INK2, label="frozen DOFA GeoFM (hollow)")
    ax.set_yticks(y); ax.set_yticklabels([lab[d] for d in ds], fontsize=8)
    ax.set_xlabel("coverage among selected units")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01))
    fig.savefig(os.path.join(FIG, "nc_fig3_backbone.png")); plt.close(fig)


if __name__ == "__main__":
    main()
