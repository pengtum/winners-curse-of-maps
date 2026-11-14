"""Pre-run expectations for the stress test of paper/STRESS_TEST_DESIGN.md, from a Gaussian model only (no project data).

Model: y = mu + r with mu ~ N(0, R2) and r ~ N(0, 1 - R2), homoscedastic, so normalized CP and Mondrian CP with bins of
mu share one interval mu +/- 1.645 sd(r); the printed number is that interval's coverage among the selected units.
Arm A (two maps): y = a1 u1 + a2 u2 + a3 u3 + e with variances 0.39, 0.28, 0.25, 0.08, so the site, site+height and
full maps E[y | ...] have R2 0.39, 0.67 and 0.92 (the held-out R2 of the carbon maps); intervals come from the site map.
Arm B (field signal): Z = y + tau sd(r) eps and the rule ranks S = (tau^2 mu + Z) / (tau^2 + 1); control Z0 = mu + sd(r) eps.
    ~/miniforge3/envs/favela/bin/python experiments/stress_power_gaussian.py
"""
import numpy as np

Z = 1.6449


def cover(score, r, s, frac):
    sel = score > np.quantile(score, 1 - frac)
    return float(np.mean(np.abs(r[sel]) <= Z * s)), float(r[sel].mean() / s)


def main(N=2_000_000, seed=20261030):
    rng = np.random.default_rng(seed)
    print("Arm B: coverage of the map's intervals among the units selected by S (top 5% / 1%), and by Z alone (5%)")
    for R2 in (0.75, 0.80, 0.85, 0.92, 0.95):
        mu = rng.normal(0, np.sqrt(R2), N); s = np.sqrt(1 - R2); r = rng.normal(0, s, N); y = mu + r
        eps = rng.normal(0, 1, N)
        for tau in (0.5, 1.0, 2.0):
            Zt = y + tau * s * eps; S = (tau ** 2 * mu + Zt) / (tau ** 2 + 1)
            c5, b5 = cover(S, r, s, 0.05); c1, _ = cover(S, r, s, 0.01); cz, _ = cover(Zt, r, s, 0.05)
            print(f"  R2 {R2:.2f} tau {tau:3.1f}: 5% {c5:.3f} (mean residual {b5:+.2f} sd), 1% {c1:.3f}; Z alone 5% {cz:.3f}")
        c0, _ = cover((mu + mu + s * eps) / 2, r, s, 0.05)
        print(f"  R2 {R2:.2f} control: 5% {c0:.3f}")
    print("Arm A: coverage of the site map's intervals among the units selected by each map")
    u = rng.normal(0, 1, (4, N))
    mus = np.sqrt(0.39) * u[0]; muh = mus + np.sqrt(0.28) * u[1]; muf = muh + np.sqrt(0.25) * u[2]
    y = muf + np.sqrt(0.08) * u[3]; r = y - mus; s = np.sqrt(0.61)
    for nm, sc in (("site (own map)", mus), ("site+height", muh), ("full", muf)):
        for f in (0.05, 0.01):
            c, b = cover(sc, r, s, f)
            print(f"  selected by {nm:14s} top {f:.0%}: {c:.3f} (mean residual {b:+.2f} sd)")


if __name__ == "__main__":
    main()
