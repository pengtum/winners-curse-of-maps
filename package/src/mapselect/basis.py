"""Function class for RegionCP (SC-GeoCCP in earlier versions): intercept, Gaussian bumps at several spatial scales, indicators of decision units.
The basis must be fixed given the training fold; it may use training coordinates, never audited or candidate outcomes."""
import numpy as np


class Multiscale:
    """Columns lie in [0, 1], so one L1 penalty acts on all of them on the same scale. Needs scikit-learn (k-means)."""

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
        """Only the intercept is unpenalized, which gives exact coverage among all selected units."""
        p = 1 + len(self.h) + (0 if self.units is None else len(self.units))
        pen = np.ones(p, bool); pen[0] = False
        return pen
