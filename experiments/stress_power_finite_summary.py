"""Summary of experiments/stress_power_finite.py output (written by the pre-run design review, 2026-10-30): per-setting
 expectations, joint count distributions and the arm-B expectations anchored to the policy-rule coverage reported in
 results/RESULTS_MONDRIAN_RULES.md. Reads results/stress_power_finite_300.json only.
    ~/miniforge3/envs/favela/bin/python experiments/stress_power_finite_summary.py
"""
import json, numpy as np, pandas as pd
d = json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "results", "stress_power_finite_300.json")))
rows = []
for name, k, o in d["B"]:
    r = dict(name=name, k=k, **{kk: v[0] for kk, v in o.items() if kk != "cor"}, **{kk + "|inf": v[1] for kk, v in o.items() if kk != "cor"})
    r.update({f"cor|{a}": b for a, b in o["cor"].items()})
    rows.append(r)
B = pd.DataFrame(rows)
order = ["health_obesity", "poverty", "poverty_lab", "voting", "seattle", "health_stroke", "health_bphigh", "camels", "fia", "health_diabetes", "era5"]
R2 = {"camels": .9144, "seattle": .8457, "voting": .8049, "health_stroke": .8938, "health_diabetes": .9288, "health_bphigh": .9106,
      "health_obesity": .7495, "fia": .9152, "era5": .9512, "poverty_lab": .7768, "poverty": .7633}
POLICY_B = {"poverty": .908, "poverty_lab": .898, "fia": .890, "camels": .898, "voting": .889, "seattle": .907, "era5": .888,
            "health_stroke": .918, "health_diabetes": .911, "health_bphigh": .940, "health_obesity": .913}   # RESULTS_MONDRIAN_RULES.md, policy 5%, bins of mu
pd.set_option("display.width", 250)
tab = []
for nm in order:
    g = B[B.name == nm]
    row = dict(setting=nm, R2=R2[nm])
    for ru in ("tau0.5", "tau1", "tau2", "control", "policy"):
        x = g[f"{ru}|naive10"]; row[f"{ru}"] = round(x.mean(), 3); row[f"sd_{ru}"] = round(x.std(), 3); row[f"P<.85_{ru}"] = round((x < .85).mean(), 2)
    drop = g["tau0.5|naive10"].mean() - g["policy|naive10"].mean()
    row["B_policy"] = POLICY_B[nm]; row["shifted_tau0.5"] = round(POLICY_B[nm] + drop, 3)
    row["P<.85_shifted"] = round(((g["tau0.5|naive10"] - g["policy|naive10"].mean() + POLICY_B[nm]) < .85).mean(), 2)
    row["jomi_tau0.5"] = round(g["tau0.5|jomi"].mean(), 3); row["jomi_inf"] = round(g["tau0.5|jomi|inf"].mean(), 2)
    row["P_jomi<.85(any rule)"] = round((g[[f"{ru}|jomi" for ru in ("tau0.5", "tau1", "tau2", "control")]] < .85).any(axis=1).mean(), 3)
    row["inf10_t.5"] = round(g["tau0.5|inf10"].mean(), 3); row["inf20_t.5"] = round(g["tau0.5|inf20"].mean(), 3)
    row["P_inf10<.85"] = round((g["tau0.5|inf10"] < .85).mean(), 2); row["P_inf20<.85"] = round((g["tau0.5|inf20"] < .85).mean(), 2)
    row["mono"] = round(((g["tau0.5|naive10"] < g["tau1|naive10"]) & (g["tau1|naive10"] < g["tau2|naive10"])).mean(), 2)
    row["mono_incl_policy"] = round(((g["tau0.5|naive10"] < g["tau1|naive10"]) & (g["tau1|naive10"] < g["tau2|naive10"]) & (g["tau2|naive10"] <= g["policy|naive10"])).mean(), 2)
    row["|ctrl-pol|<=.02"] = round(((g["control|naive10"] - g["policy|naive10"]).abs() <= .02).mean(), 2)
    row["sd(ctrl-pol)"] = round((g["control|naive10"] - g["policy|naive10"]).std(), 3)
    row["cor.5/1/2"] = "/".join(f"{g[f'cor|{t}'].mean():.3f}" for t in ("tau0.5", "tau1", "tau2"))
    tab.append(row)
T = pd.DataFrame(tab)
cols1 = ["setting", "R2", "tau0.5", "sd_tau0.5", "P<.85_tau0.5", "tau1", "P<.85_tau1", "tau2", "P<.85_tau2", "control", "policy", "B_policy", "shifted_tau0.5", "P<.85_shifted"]
print(T[cols1].to_string(index=False))
cols2 = ["setting", "jomi_tau0.5", "jomi_inf", "P_jomi<.85(any rule)", "inf10_t.5", "P_inf10<.85", "inf20_t.5", "P_inf20<.85", "mono", "mono_incl_policy", "|ctrl-pol|<=.02", "sd(ctrl-pol)", "cor.5/1/2"]
print(T[cols2].to_string(index=False))
# joint count distributions: study k is independent across settings
W = B.pivot(index="k", columns="name")
def cnt(ru, me="naive10", names=order, thr=.85):
    return sum((W[(f"{ru}|{me}", nm)] < thr).astype(int) for nm in names)
for ru in ("tau0.5", "tau1", "tau2", "control"):
    c = cnt(ru); print(ru, "count naive10<.85 distribution:", c.value_counts().sort_index().to_dict(), " P(>=2)", round((c >= 2).mean(), 3))
c = cnt("tau0.5"); print("P(count in {4,5}) tau0.5:", round(c.isin([4, 5]).mean(), 3))
c1 = cnt("tau1"); print("P(count in {0,1}) tau1:", round(c1.isin([0, 1]).mean(), 3))
meas = ["poverty", "poverty_lab", "voting", "seattle", "camels", "fia"]
cm = cnt("tau0.5", names=meas); print("measured-only count tau0.5:", cm.value_counts().sort_index().to_dict())
# distinct sources: DHS counts once
dhs = ((W[("tau0.5|naive10", "poverty")] < .85) | (W[("tau0.5|naive10", "poverty_lab")] < .85)).astype(int)
cs = dhs + sum((W[("tau0.5|naive10", nm)] < .85).astype(int) for nm in ["voting", "seattle", "camels", "fia"])
print("measured distinct-source count tau0.5:", cs.value_counts().sort_index().to_dict(), " P(>=2)", round((cs >= 2).mean(), 3))
# monotone in >= 9 settings
mono = sum(((W[("tau0.5|naive10", nm)] < W[("tau1|naive10", nm)]) & (W[("tau1|naive10", nm)] < W[("tau2|naive10", nm)])).astype(int) for nm in order)
print("number of monotone settings:", mono.value_counts().sort_index().to_dict(), " P(>=9)", round((mono >= 9).mean(), 3))
ctrl = sum((((W[("control|naive10", nm)] - W[("policy|naive10", nm)]).abs()) <= .02).astype(int) for nm in order)
print("settings with |control-policy|<=.02:", ctrl.value_counts().sort_index().to_dict(), " P(all 11)", round((ctrl == 11).mean(), 3), " P(>=10)", round((ctrl >= 10).mean(), 3))
jany = sum((W[[(f"{ru}|jomi", nm) for ru in ("tau0.5", "tau1", "tau2", "control")]] < .85).any(axis=1).astype(int) for nm in order)
print("P(JOMI 0 settings below .85 under every tau and control):", round((jany == 0).mean(), 3))
W = B.pivot(index="k", columns="name")
sh = {}
for nm in order:
    sh[nm] = W[("tau0.5|naive10", nm)] - W[("policy|naive10", nm)].mean() + POLICY_B[nm]
S = pd.DataFrame(sh)
c = (S < .85).sum(axis=1)
print("anchored count tau0.5 (all 11):", c.value_counts().sort_index().to_dict(), " P(4-5)", round(c.isin([4, 5]).mean(), 3), " P(>=2)", round((c >= 2).mean(), 3))
meas = ["poverty", "poverty_lab", "voting", "seattle", "camels", "fia"]
cm = (S[meas] < .85).sum(axis=1); print("anchored measured-only:", cm.value_counts().sort_index().to_dict())
dhs = ((S["poverty"] < .85) | (S["poverty_lab"] < .85)).astype(int)
cs = dhs + (S[["voting", "seattle", "camels", "fia"]] < .85).sum(axis=1)
print("anchored measured distinct sources:", cs.value_counts().sort_index().to_dict(), " P(>=2)", round((cs >= 2).mean(), 3))
places = ["health_obesity", "health_stroke", "health_bphigh", "health_diabetes"]
print("anchored PLACES count:", (S[places] < .85).sum(axis=1).value_counts().sort_index().to_dict())
