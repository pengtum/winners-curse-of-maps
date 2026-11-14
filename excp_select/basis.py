"""Function classes for SC-GeoCCP (theory/SC_GeoCCP.md).

A basis must be FIXED given the training fold (it may use training coordinates, never
calibration / test labels). Columns are nonnegative and <= 1, so the L1 penalty acts on all
directions on the same scale; with l1 >= tau = 1 - alpha the LP is unbounded exactly when
JOMI is infinite (theory/SC_GeoCCP.md Prop. 2).

  Multiscale(s_train, levels)   intercept + Gaussian bumps at several spatial scales, centres
                                by k-means on the training coordinates, bandwidth = median
                                nearest-centre distance at that level
  .with_units(units)            + indicators of administrative / decision units (MAUP: the
                                units the policy is decided on)
"""
import numpy as np


class Multiscale:
    def __init__(self, s_train, levels=(4, 12, 36), seed=0, min_per_centre=15):
        from sklearn.cluster import KMeans
        from scipy.spatial import cKDTree
        s_train = np.asarray(s_train, float)
        self.centres, self.h, self.level = [], [], []
        for lv, k in enumerate(levels):
            k = int(min(k, max(2, len(s_train) // min_per_centre)))
            c = KMeans(k, n_init=4, random_state=seed).fit(s_train).cluster_centers_
            d = cKDTree(c).query(c, k=2)[0][:, 1]
            self.centres.append(c); self.h += [np.median(d)] * k; self.level += [lv] * k
        self.centres = np.vstack(self.centres); self.h = np.array(self.h); self.level = np.array(self.level)
        self.units = None

    def with_units(self, units):
        self.units = np.unique(units)
        return self

    def __call__(self, s, units=None):
        s = np.asarray(s, float)
        D2 = ((s[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1)
        cols = [np.ones((len(s), 1)), np.exp(-0.5 * D2 / self.h[None, :] ** 2)]
        if self.units is not None:
            cols.append((np.asarray(units)[:, None] == self.units[None, :]).astype(float))
        return np.hstack(cols)

    @property
    def penalize(self):
        """Only the intercept is unpenalized (exact selection-conditional coverage); every bump and
        unit direction gets the L1 slack l1 * E[1/|R|] / E[f | selected] (Theorem 1)."""
        p = 1 + len(self.h) + (0 if self.units is None else len(self.units))
        pen = np.ones(p, bool); pen[0] = False
        return pen

    def names(self):
        out = ["intercept"] + [f"bump L{l}#{i}" for i, l in enumerate(self.level)]
        if self.units is not None:
            out += [f"unit {u}" for u in self.units]
        return out
