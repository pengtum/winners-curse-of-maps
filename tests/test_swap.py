"""Case-by-case check of every closed-form reference set against the literal swap.

    ~/miniforge3/envs/favela/bin/python tests/test_swap.py [n_instances]

For each rule, random small instances (n_cal, m in 8..40, random groups with tiny / empty
groups, random budgets and bandwidths); for every selected j the closed-form R_j must equal the
brute-force R_j exactly. Prints the number of (instance, j) cases compared and mismatches.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.dont_write_bytecode = True
from excp_select import rules, refsets, bruteforce  # noqa: E402

ALPHA = 0.1


def inst(rng):
    n, m = int(rng.integers(8, 41)), int(rng.integers(8, 41))
    G = int(rng.integers(1, 6))
    def unit(k):
        return dict(mu=rng.normal(size=k), g=rng.integers(0, G, k), v=rng.gamma(2.0, size=k) + 0.1,
                    c=rng.uniform(0.5, 3.0, k), s=rng.uniform(0, 10, (k, 2)), sigma=rng.gamma(2.0, size=k))
    return unit(n), unit(m), G


def same(a, b):
    return np.array_equal(a.selected, b.selected) and np.array_equal(a.member, b.member)


def whole_bin_rule(lo, hi):
    """Select every test unit whose prediction lies in the fixed bin (lo, hi]; its reference set is the bin (Mondrian CP)."""
    def rule(cal, test):
        x = np.asarray(test["mu"], float)
        return (x > lo) & (x <= hi)
    rule.name = "whole-bin"
    return rule


def whole_bin_closed(mu_cal, mu_test, lo, hi):
    sidx = np.flatnonzero((np.asarray(mu_test) > lo) & (np.asarray(mu_test) <= hi))
    row = (np.asarray(mu_cal) > lo) & (np.asarray(mu_cal) <= hi)
    return refsets.Selection(sidx, np.broadcast_to(row, (len(sidx), len(row))).copy(), "whole-bin")


def main(N=200):
    rng = np.random.default_rng(2026)
    rng_bin = np.random.default_rng(20261030)        # separate stream, so the other rules' instances are unchanged
    stats = {}
    for t in range(N):
        cal, test, G = inst(rng)
        m = len(test["mu"])
        K = int(rng.integers(0, m + 1))
        Kg = {u: int(rng.integers(0, 4)) for u in range(G)}
        B = float(rng.uniform(0.0, 0.6) * test["c"].sum())
        h = float(rng.choice([0.5, 1.0, 2.0, 4.0]))
        c_thr = float(np.quantile(test["sigma"], rng.uniform(0.1, 0.9)))
        lo = float(rng_bin.normal()); hi = lo + float(rng_bin.uniform(0.2, 2.0))
        cases = {
            "top-K": (rules.topk(K), refsets.topk(cal["mu"], test["mu"], K)),
            "bottom-K": (rules.topk(K, largest=False), refsets.topk(cal["mu"], test["mu"], K, largest=False)),
            "quota": (rules.quota(Kg), refsets.quota(cal["mu"], test["mu"], cal["g"], test["g"], Kg)),
            "quota-bottom": (rules.quota(Kg, largest=False),
                             refsets.quota(cal["mu"], test["mu"], cal["g"], test["g"], Kg, largest=False)),
            "knapsack": (rules.knapsack_greedy(B), refsets.knapsack_greedy(cal["v"], cal["c"], test["v"], test["c"], B)),
            "geocp-mask": (rules.kernel_mask(h, ALPHA), refsets.kernel_mask(cal["s"], test["s"], h, ALPHA)),
            "threshold": (rules.threshold(c_thr), refsets.threshold(cal["sigma"], test["sigma"], c_thr)),
            "whole-bin": (whole_bin_rule(lo, hi), whole_bin_closed(cal["mu"], test["mu"], lo, hi)),
        }
        for name, (rule, closed) in cases.items():
            brute = bruteforce.reference_sets(rule, cal, test)
            s = stats.setdefault(name, [0, 0, 0, 0])
            s[0] += 1; s[1] += len(brute); s[2] += int(not same(closed, brute))
            s[3] += int(brute.member.sum())
    print(f"{'rule':14s} instances  selected-j  mismatched-instances  total|R|")
    ok = True
    for name, (a, b, c, d) in stats.items():
        print(f"{name:14s} {a:9d} {b:11d} {c:21d} {d:9d}")
        ok &= c == 0
    print("ALL CLOSED FORMS MATCH BRUTE FORCE" if ok else "MISMATCH FOUND")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 200) else 1)
