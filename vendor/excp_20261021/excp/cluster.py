"""Cluster (PSU) designs: unit-level targets with cluster-level randomness.

Under an SRS of clusters (or Poisson on clusters with odds weights), the exchangeable objects
are whole clusters. Cluster-atom conformal: for a test unit in unsampled cluster g, impute +inf
for ALL m_g units of that cluster and take the (1 - alpha) quantile of the unit-weighted
calibration scores plus that atom. Then (ratio form, GeoCP_framework III(b'))

    E[ sum_{test units} 1{y_u not in C_u} ] <= alpha * E[ |T| ],

for any within- and between-cluster dependence. Wider than unit-level methods because only
the number of CALIBRATION CLUSTERS carries information -- that is the honest price.
"""
import numpy as np


def cluster_atom_quantile(S_cal, cl_cal, cl_test, sizes, alpha, w_cluster=None):
    """S_cal: unit scores of calibration clusters; cl_*: cluster ids; sizes[g]: number of units of
    cluster g in the test set (whole cluster); w_cluster: optional cluster-level exchange weights
    (dict or array indexed by cluster id), default uniform."""
    S = np.asarray(S_cal, float); o = np.argsort(S); Ss = S[o]
    wu = np.ones(len(S)) if w_cluster is None else np.asarray(w_cluster, float)[np.asarray(cl_cal)]
    cs = np.cumsum(wu[o])
    wt = np.ones(len(cl_test)) if w_cluster is None else np.asarray(w_cluster, float)[np.asarray(cl_test)]
    atom = wt * np.asarray(sizes, float)[np.asarray(cl_test)]
    tot = cs[-1] + atom
    idx = np.searchsorted(cs, (1 - alpha) * tot - 1e-9 * tot, side="left")
    q = np.full(len(cl_test), np.inf); ok = idx < len(Ss)
    q[ok] = Ss[idx[ok]]
    return q
