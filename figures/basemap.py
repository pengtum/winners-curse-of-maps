"""Natural Earth 1:110m outlines for the map panels: US states (Fig. 8b, 8d) and countries (Figs 3a, 6a).

The two GeoJSON files in data/natural_earth_110m_v5.1.2/ are public domain (release v5.1.2; source, sizes and
hashes in STATUS.md). They are downloaded data: only the standard json module parses them, and the figure
scripts that import this module run with python -I. Legends credit them with CREDIT.

    import basemap as BM
    proj = BM.km_projection(coords, lon, lat)      # the dataset's own km coordinates (experiments/data.py)
    st = BM.conus()                                # {postal code: polygons}
    BM.fill(ax, st, proj, {"AZ": colour})          # choropleth; unlisted states stay empty
    BM.outline(ax, st, proj, coast=BM.COAST)       # state lines, darker outer border
    BM.label_features(ax, st, proj, {"AZ": "AZ"}, fills, outside={"VT": (lon, lat, True)})
"""
import json, os
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Bbox

from style import ROOT, INK, INK2, clip_line as _clip_line

NE = os.path.join(ROOT, "data", "natural_earth_110m_v5.1.2")
STATES, COUNTRIES = "ne_110m_admin_1_states_provinces.geojson", "ne_110m_admin_0_countries.geojson"
CREDIT = "Made with Natural Earth."
EDGE = "#cfcdc6"              # shared borders: darker than the gridlines, lighter than the light-grey data colour
COAST = "#9d9b94"             # outer border (coast, national border): darker than the light-grey data colour


def features(layer, key, keep=None):
    """{properties[key]: [polygon, ...]}, each polygon a list of (n, 2) lon / lat rings, exterior first."""
    with open(os.path.join(NE, layer), encoding="utf-8") as f:
        fc = json.load(f)
    out = {}
    for ft in fc["features"]:
        p, g = ft["properties"], ft["geometry"]
        if keep is None or keep(p):
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            out.setdefault(p[key], []).extend([np.asarray(r, float) for r in poly] for poly in polys)
    return out


def conus():
    """The 48 contiguous states and DC, keyed by postal code."""
    return features(STATES, "postal", keep=lambda p: p["postal"] not in ("AK", "HI"))


def countries(keep=None):
    """Countries keyed by ADM0_A3 (Natural Earth's default view: de facto boundaries)."""
    return features(COUNTRIES, "ADM0_A3", keep)


def km_projection(coords, lon, lat):
    """lon / lat rings -> the dataset's km coordinates. experiments/data.py::to_km is a plate carree,
    x = lon * 111.32 cos(lat0), y = lat * 110.57 with lat0 the dataset's mean latitude; both scale factors are
    recovered from the dataset itself (least squares through the origin) and must reproduce it exactly."""
    kx, ky = coords[:, 0] @ lon / (lon @ lon), coords[:, 1] @ lat / (lat @ lat)
    err = np.abs(np.c_[lon * kx, lat * ky] - coords).max()
    if err > 1e-6 * np.abs(coords).max():
        raise ValueError(f"coordinates are not a plate carree of lon / lat (max error {err:.3g} km)")
    return lambda r: np.c_[r[:, 0] * kx, r[:, 1] * ky]


def _path(polys, proj):
    verts, codes = [], []
    for poly in polys:
        for r in poly:
            verts.append(proj(r)); codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY]
    return Path(np.concatenate(verts), codes)


def fill(ax, feats, proj, colours, box=None, z=1):
    """Choropleth fills without edges (outline() draws the boundaries); unlisted features stay empty. box =
    (x0, x1, y0, y1) crops the polygons to the map window, so the PDF holds only the visible geometry."""
    for k, c in colours.items():
        path = _path(feats[k], proj)
        if box is not None:
            path = path.clip_to_bbox(Bbox([[box[0], box[2]], [box[1], box[3]]]))
        if len(path.vertices):
            ax.add_patch(PathPatch(path, facecolor=c, edgecolor="none", zorder=z))


def outline(ax, feats, proj, color=EDGE, lw=0.3, coast=None, coast_lw=0.4, borders_of=None, box=None, z=3):
    """Boundary lines, each shared border drawn once. With coast set, the borders that belong to one feature
    only (coastline, national border) are drawn in that colour, which gives the map its outer silhouette. With
    borders_of, a shared border is drawn only when a feature on either side is in that set (the study countries):
    borders between two other countries, disputed ones included, are left out. box = (x0, x1, y0, y1) crops the
    lines to the map window: the axes clip only the drawing, while a PDF keeps every stroke to the page edge."""
    rings = [(f, r) for f, polys in feats.items() for poly in polys for r in poly]
    segs = [[_key(a, b) for a, b in zip(r[:-1], r[1:])] for _, r in rings]
    count, owners = {}, {}
    for (f, _), ss in zip(rings, segs):
        for k in ss:
            count[k] = count.get(k, 0) + 1; owners.setdefault(k, set()).add(f)
    keep = None if borders_of is None else set(borders_of)
    drawn, runs = set(), {1: [], 2: []}
    for (_, r), ss in zip(rings, segs):
        lab = np.zeros(len(ss), int)                   # 0 skipped / already drawn, 1 shared border, 2 outer border
        for i, k in enumerate(ss):
            if count[k] == 1:
                lab[i] = 2 if coast is not None else 1
            elif k not in drawn and (keep is None or owners[k] & keep):
                drawn.add(k); lab[i] = 1
        cut = np.flatnonzero(np.diff(lab)) + 1
        for lo, hi in zip(np.r_[0, cut], np.r_[cut, len(lab)]):
            if lab[lo]:
                line = proj(r[lo:hi + 1])
                runs[lab[lo]] += [line] if box is None else _clip_line(line, box)
    kw = dict(zorder=z, capstyle="round", joinstyle="round")
    out = [ax.add_collection(LineCollection(runs[1], colors=color, linewidths=lw, **kw))]
    if runs[2]:
        out.append(ax.add_collection(LineCollection(runs[2], colors=coast, linewidths=coast_lw, **kw)))
    return out


def locate(feats, pts, proj=None):
    """Key of the feature containing each lon / lat point (rows of pts), None outside all of them."""
    proj = proj or (lambda r: r)
    q, out = proj(np.asarray(pts, float)), np.full(len(pts), None, dtype=object)
    for k, polys in feats.items():
        free = np.array([o is None for o in out])
        out[free & _path(polys, proj).contains_points(q)] = k
    return out


def label_features(ax, feats, proj, labels, fills=None, fit=1.1, outside=None, fontsize=6.2):
    """Short labels ({feature: text}): inside the feature where the label box fits with margin fit (label_spot),
    in white or ink, whichever contrasts more with the fill; otherwise at outside[feature] = (lon, lat, leader) in
    open space, in secondary ink, with a leader line to the feature when leader. Returns {feature: margin}."""
    fills, outside, rend, margin = fills or {}, outside or {}, ax.figure.canvas.get_renderer(), {}
    for k, s in labels.items():
        t = ax.text(0, 0, s, fontsize=fontsize, ha="center", va="center", zorder=5)
        bb = t.get_window_extent(rend).transformed(ax.transData.inverted())
        xy, margin[k] = label_spot(feats[k], proj, bb.width / 2, bb.height / 2)
        if margin[k] >= fit:
            fc = fills.get(k, "white")
            t.set_position(xy); t.set_color(max(("white", INK), key=lambda c: _contrast(c, fc)))
            continue
        if k not in outside:
            raise ValueError(f"label {s!r} does not fit inside {k} (margin {margin[k]:.2f} < {fit}); "
                             f"give it an outside position")
        t.remove(); lon, lat, leader = outside[k]; pos = proj(np.array([[lon, lat]], float))[0]
        ax.annotate(s, xy=xy, xytext=pos, fontsize=fontsize, color=INK2, ha="center", va="center", zorder=5,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=0) if leader else None)
    return margin


def label_spot(polys, proj, hw, hh, n=70):
    """Centre for a label of half-width hw and half-height hh (projected units) inside the feature's largest
    polygon: the grid point whose label box keeps the widest margin to the boundary. Returns (centre, margin),
    the margin being the L-inf distance to the boundary in half-sizes: above 1 the label fits inside."""
    rings = [proj(r) for r in max(polys, key=lambda poly: _area(proj(poly[0])))]
    lo, hi = rings[0].min(0), rings[0].max(0)
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)), -1).reshape(-1, 2)
    ok = Path(rings[0]).contains_points(g)
    for h in rings[1:]:
        ok &= ~Path(h).contains_points(g)
    g = g[ok]
    e = np.concatenate([_densify(r, min(hw, hh) / 6) for r in rings])
    m = np.abs((g[:, None, :] - e[None]) / np.array([hw, hh])).max(-1).min(1)
    k = m.argmax()
    return g[k], m[k]


def _contrast(a, b):
    """WCAG contrast ratio of two colours."""
    lum = lambda c: np.dot([0.2126, 0.7152, 0.0722],
                           [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in to_rgb(c)])
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _key(a, b):
    a, b = (round(a[0], 6), round(a[1], 6)), (round(b[0], 6), round(b[1], 6))
    return (a, b) if a < b else (b, a)


def _densify(r, step):
    out = [r[:1]]
    for a, b in zip(r[:-1], r[1:]):
        k = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.concatenate(out)


def _area(r):
    return 0.5 * abs(np.dot(r[:-1, 0], r[1:, 1]) - np.dot(r[1:, 0], r[:-1, 1]))
