"""One visual style for every manuscript figure (Nature Communications; paper/FIGURE_PLAN.md).

Sizes are physical: 89 mm single / 183 mm double column, 6.2-6.5 pt Arial (at least 6 pt in print), 8 pt bold panel letters.
Colour = method family, marker = family member, so identity never rests on colour alone. The family hues
pass the colour-vision check on ALL pairs (blue / violet / orange / aqua; grey is the neutral family):
the original violet #6250d6 was too close to JOMI blue (normal-vision dE 10.5, deuteranope 4.5) and is
re-stepped to #4a3aa7 (16.3 / 13.0). Red (carbon protocols) never shares a panel with GeoCP orange.

    import style as S
    S.apply()                       # rcParams; nothing is changed at import time
    fig = S.figure(183, 120)
    S.dot(ax, x, y, "JOMI stud")    # family colour + member marker
    S.save(fig, "nc_fig4_prevalence", axes=[...], panel_ids=[...])   # alignment gate, then PDF/SVG/PNG

Schematic figures (method flow, concept) are built from the parts at the end of this file: a millimetre canvas per
panel, cards, flow arrows, step badges, guarantee strips, data-role markers for map units, and map schematics
(a shaded surface, Voronoi admin units, oblique layer stacks of basis functions).
"""
import os, sys
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, ScaledTranslation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
QA, SRC = os.path.join(FIG, "qa"), os.path.join(FIG, "source_data")
MM = 1 / 25.4
WIDTH = {"single": 89, "double": 183}
ALPHA = 0.1

# ---- ink and chrome
INK, INK2, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f3f2ef"
FS = dict(tick=6.2, label=6.2, legend=6.2, note=6.2, title=6.5, panel=8)   # PNAS: >= 6 pt at final size; figures
# are 183 mm wide and print at about 180 mm (scale 0.984), so 6.2 pt prints at 6.1 pt (5.5 / 6 until 2026-11-10)
LW = dict(axis=0.5, grid=0.4, ref=0.6, line=0.9, err=0.45, marker=0.6)

# ---- method families (FIGURE_PLAN colour table)
GREY_L, GREY_M, GREY_D = "#b5b3ac", "#7d7b74", "#52514e"      # standard conformal
ORANGE = "#eb6834"                                             # GeoCP and other local conformal methods
AQUA = "#1baf7a"                                               # model-based intervals
BLUE = "#2a78d6"                                               # JOMI (reference-set calibration; Jin & Ren)
VIOLET = "#4a3aa7"                                             # this paper: RegionCP and the selection-aware totals
VIOLET_BAND = "#ece9f7"                                        # light background behind this paper's results
RED = "#e34948"                                                # carbon-protocol rules (VCS, ART)
MISS = "#cb181d"      # status, not a family: an interval that misses the truth (the dark step of the maps' Reds ramp)
FAMILY = {"point": INK, "conformal": GREY_M, "local": ORANGE, "model": AQUA, "jomi": BLUE,
          "ours": VIOLET, "protocol": RED}

# results-file method name -> display label, colour, marker, filled
METHODS = {
    "point":                         ("Point prediction", INK, "*", True),
    "SCP |res|":                     ("Split CP", GREY_L, "x", True),
    "SCP stud":                      ("Normalized CP", GREY_M, "s", True),
    "CQR":                           ("CQR", GREY_D, "v", True),
    "BY stud":                       ("BY-adjusted CP", GREY_M, "<", False),
    "Mondrian CP":                   ("Mondrian CP", GREY_D, "h", False),        # bins of the predicted value (SI)
    "GeoCP":                         ("GeoCP", ORANGE, "D", True),
    "LSCP":                          ("LSCP", ORANGE, "^", False),
    "RLCP":                          ("RLCP", ORANGE, "v", False),
    "Regression kriging":            ("Regression kriging", AQUA, "^", True),
    "QRF":                           ("QRF", AQUA, "o", False),
    "Bootstrap ensemble + residual": ("Bootstrap", AQUA, "s", False),
    "JOMI stud":                     ("JOMI", BLUE, "o", True),
    "JOMI|state stud":               ("JOMI per state", BLUE, "D", False),
    "JOMI CQR-QRF":                  ("JOMI + QRF score", BLUE, "^", False),
    "JOMI norm-boot":                ("JOMI + bootstrap score", BLUE, "s", False),
    # RegionCP (renamed from SC-GeoCCP on 2026-11-08; the results files and code keep the key 'SC-GeoCCP')
    "SC-GeoCCP":                     ("RegionCP (ours)", VIOLET, "P", True),
    "SC-GeoCCP CQR-QRF":             ("RegionCP + QRF score", VIOLET, "^", False),
    "SC-GeoCCP norm-boot":           ("RegionCP + bootstrap score", VIOLET, "s", False),
    "cfBH":                          ("cfBH on the reference set", BLUE, "s", True),
    # carbon project totals (Fig. 7 VCS, SI figure ART; results/carbon_*.csv.gz columns): protocol rules are red,
    # never beside orange; the ART TREES twins share the style of their VCS role
    "VCS map (RF)":                  ("Protocol rule, map error layer", RED, "D", True),
    "VCS model-assisted":            ("Protocol rule, survey-calibrated estimate", AQUA, "s", True),
    "ours: VCS formula, sel.-aware": ("Protocol rule, selection-aware estimate (ours)", VIOLET, "o", True),
    "ours: exact, risk VCS":         ("Exact selection-aware bound (ours)", VIOLET, "o", False),
    "ART map (RF)":                  ("Protocol rule, map error layer", RED, "D", True),
    "ART model-assisted":            ("Protocol rule, survey-calibrated estimate", AQUA, "s", True),
    "ours: ART formula, sel.-aware": ("Protocol rule, selection-aware estimate (ours)", VIOLET, "o", True),
    "ours: exact, risk ART":         ("Exact selection-aware bound (ours)", VIOLET, "o", False),
}
# experiments/make_nc.py format: method -> (colour, marker, filled)
STY = {k: (c, m, f) for k, (_, c, m, f) in METHODS.items()}
SHORT = {k: v[0] for k, v in METHODS.items()}
# this paper's methods (2026-11-08), so that readers find them in every panel: violet, markers 35%
# larger with an ink edge and drawn on top, and every label that names them set in bold when the figure is saved
OURS = {k for k, v in METHODS.items() if k.startswith(("SC-GeoCCP", "ours:")) or "(ours)" in v[0]}
OURS_WORDS = ("RegionCP", "(ours)")

# ---- datasets: order, display label, nature of the ground truth (rows of Figs 4a and 8a)
FIA = os.environ.get("QW2_FIA", "fia_db")               # == experiments/data.py FIA (the FIA table reported)
DATASETS = [("poverty", "DHS wealth", "observed"), ("poverty_lab", "DHS wealth, no imagery", "observed"),
            (FIA, "FIA forest biomass", "observed"), ("camels", "CAMELS runoff", "observed"),
            ("voting", "County vote share", "observed"), ("seattle", "Seattle house price", "observed"),
            ("era5", "ERA5 temperature", "reanalysis"),
            ("health_stroke", "Stroke*", "model"), ("health_diabetes", "Diabetes*", "model"),
            ("health_bphigh", "High blood pressure*", "model"), ("health_obesity", "Obesity*", "model")]
TRUTH = {"observed": "Observed outcome", "reanalysis": "Reanalysis", "model": "*Model output (CDC PLACES)"}


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS["label"], "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
        "axes.linewidth": LW["axis"], "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": LW["axis"], "ytick.major.width": LW["axis"], "xtick.major.size": 2,
        "ytick.major.size": 2, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlepad": 3, "axes.labelpad": 2, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": LW["line"], "lines.markersize": 3.5, "lines.markeredgewidth": LW["marker"],
        "legend.frameon": False, "legend.handletextpad": 0.3, "legend.borderaxespad": 0.2,
        "legend.borderpad": 0.2, "legend.columnspacing": 0.9, "legend.handlelength": 1.0,
        "legend.labelspacing": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "figure.dpi": 150, "axes.unicode_minus": True, "hatch.linewidth": 0.4,
        # math in Arial too: italic variables, upright digits; symbols Arial lacks (e.g. the set operators) come
        # from STIX sans
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.default": "it",
        "mathtext.fallback": "stixsans",
    })
    # matplotlib sets sub- and superscripts at 70% of the text size (3.9 pt for 5.5 pt notes), below the 5 pt
    # minimum; 0.91 keeps them at 5.0 pt. A private constant, so guarded: if it disappears, audit_pdf_text says so.
    import matplotlib._mathtext as mathtext
    if hasattr(mathtext, "SHRINK_FACTOR"):
        mathtext.SHRINK_FACTOR = 0.91


def ramp(name, lo):
    """Matplotlib sequential ramp from lo to its dark end: the near-white start is dropped so that the lightest
    step stays visible next to no-data white (maps: Reds from 0.18 for coverage shortfall, Purples from 0.22 for
    guaranteed coverage)."""
    return mpl.colors.ListedColormap(mpl.colormaps[name](np.linspace(lo, 1.0, 256)))


def figure(w_mm, h_mm):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def axes_mm(fig, x0, y0, w, h, **kw):
    """Axes at an exact physical position (mm from the figure's lower-left corner)."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


def text_mm(fig, x, y, s, **kw):
    """Figure text at a physical position (mm from the lower-left corner)."""
    W, H = fig.get_size_inches() / MM
    return fig.text(x / W, y / H, s, **kw)


def letter_mm(fig, x, y, letter):
    """Panel letter whose top-left corner sits at (x, y) mm."""
    return text_mm(fig, x, y, letter, fontsize=FS["panel"], fontweight="bold", ha="left", va="top")


def panel_label(ax, letter, dx_pt=0.0, dy_pt=0.0, x=0.0, y=1.0):
    """Bold lowercase letter at a fixed physical offset from an axes corner (stays aligned across
    panels of different size; FIGURE_PLAN: 8 pt bold)."""
    off = ScaledTranslation(dx_pt / 72, dy_pt / 72, ax.figure.dpi_scale_trans)
    return ax.text(x, y, letter, transform=ax.transAxes + off, fontsize=FS["panel"], fontweight="bold",
                   ha="left", va="bottom", color=INK)


def style_of(key):
    lab, c, m, f = METHODS[key]
    return dict(color=c, marker=m, filled=f, label=lab)


def dot(ax, x, y, key, s=11, z=3, edge=None, alpha=1.0, hollow=None, emph=True, **kw):
    """Scatter in the method's family style; hollow markers get a white face (a surface ring). This paper's methods
    (OURS) are drawn 35% larger, on top, and filled ones with an ink edge; emph=False keeps the plain style where a
    panel shows nothing else (a dense cloud of one method needs no emphasis)."""
    lab, c, m, f = METHODS[key]
    f = f if hollow is None else not hollow
    if key in OURS and emph:
        s, z = s * 1.35, max(z, 6)
        edge = edge or (INK if f else None)
    if m in ("x", "+", "|", "_"):
        return ax.scatter(x, y, s=s, marker=m, color=c, linewidths=0.8, zorder=z, alpha=alpha, **kw)
    return ax.scatter(x, y, s=s, marker=m, facecolor=c if f else "white", edgecolor=edge or c,
                      linewidths=LW["marker"] if f else 0.75, zorder=z, alpha=alpha, **kw)


def handle(key, label=None, ms=3.6, hollow=None):
    lab, c, m, f = METHODS[key]
    f = f if hollow is None else not hollow
    ours = key in OURS
    ms = ms * 1.16 if ours else ms                     # area x 1.35, as in dot()
    if m in ("x", "+"):
        return Line2D([], [], ls="none", marker=m, ms=ms, color=c, mew=0.8, label=label or lab)
    return Line2D([], [], ls="none", marker=m, ms=ms, mfc=c if f else "white", mec=INK if (ours and f) else c,
                  mew=LW["marker"] if f else 0.75, label=label or lab)


def bold_ours(fig):
    """Set in bold the labels that name this paper's methods; called by save(). A label counts when it starts with
    'RegionCP' or contains '(ours)', is at most 60 characters and does not also name JOMI, so that sentences split over
    several lines, mixed annotations ('JOMI, RegionCP') and rules without a guarantee are left alone."""
    import matplotlib.text as mtext
    for t in fig.findobj(mtext.Text):
        s = (t.get_text() or "").strip()
        if s and len(s) <= 60 and "JOMI" not in s and (s.startswith("RegionCP") or "(ours)" in s):
            t.set_fontweight("bold")


def target(ax, v=1 - ALPHA, orient="v", **kw):
    """Nominal-level reference line (dashed so it never reads as a gridline)."""
    kw = dict(dict(color=INK2, lw=LW["ref"], ls=(0, (3, 2)), zorder=1), **kw)
    return ax.axvline(v, **kw) if orient == "v" else ax.axhline(v, **kw)


def hgrid(ax, ys, **kw):
    for yy in ys:
        ax.axhline(yy, color=GRID, lw=LW["grid"], zorder=0, **kw)


def clean(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)


def source_data(name, df):
    os.makedirs(SRC, exist_ok=True)
    df.to_csv(os.path.join(SRC, f"{name}.csv"), index=False, float_format="%.4f")


def save(fig, name, png_dpi=600, **alignment):
    """Blocking panel-alignment gate (render-time axes geometry, 1.5 pt tolerance), then vector PDF (the
    submission file for line art) and SVG (editable text), and a 600 dpi PNG preview, all at the exact figure
    size (no tight bounding box: the physical width is part of the spec). The alignment auditor is the
    figure-QA helper module; put its folder on PYTHONPATH to run the gate, or in FIGURE_QA_HELPERS for scripts
    run with python -I (which ignores PYTHONPATH; the map figures read downloaded files)."""
    os.makedirs(QA, exist_ok=True)
    bold_ours(fig)                                     # this paper's methods named in bold, in every figure
    helpers = os.environ.get("FIGURE_QA_HELPERS")
    if helpers and helpers not in sys.path:
        sys.path.append(helpers)
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        print(f"[{name}] panel-alignment audit NOT RUN (auditor not on PYTHONPATH or FIGURE_QA_HELPERS)")
    else:
        rep = require_matplotlib_panel_alignment(
            fig, json_out=os.path.join(QA, f"{name}.alignment.json"),
            overlay_svg=os.path.join(QA, f"{name}.alignment.svg"), tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            strict=True, **alignment)
        print(f"[{name}] panel alignment: {rep.get('verdict', rep.get('status', '?'))}")
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.svg"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=png_dpi)


# ---- schematic parts (method flow and concept figures). Each panel is a canvas whose data units are millimetres,
# so boxes, text pitch and marker sizes are physical. A map unit's data role is told by its fill; colour is used
# only where a method family owns the role (blue: the reference set of JOMI-type calibration).
LINE = 2.55                                                    # baseline pitch of 5.5 pt text rows (mm)
ROLE = {  # data role -> face, edge, marker area on a mini-map (pt^2), label
    "train":       (GREY_L, "none", 1.6, "Training fold"),
    "calibration": ("white", INK2, 3.0, "Calibration"),
    "test":        (GREY_M, "white", 3.8, "Test"),
    "selected":    (INK, "white", 5.8, "Selected"),
    "reference":   (BLUE, "white", 4.8, "Reference set"),
}


def card_panel(fig, x0, y0, w, h, letter, title, step=None, tag=None, tx=1.4):
    """A schematic panel: a millimetre canvas holding a card, the panel letter, an optional step badge, the title and
    an optional italic tag right-aligned on the title's baseline."""
    ax = canvas(fig, x0, y0, w, h)
    card(ax, 0, 0, w, h)
    yb = h - 3.4
    ax.text(tx, yb, letter, fontsize=FS["panel"], fontweight="bold", ha="left", va="baseline")
    x = tx + 3.6
    if step:
        badge(ax, x + 1.25, yb + 0.85, step); x += 3.3
    ax.text(x, yb, title, fontsize=FS["title"], fontweight="bold", ha="left", va="baseline")
    if tag:
        ax.text(w - tx, yb, tag, fontsize=FS["note"], color=INK2, fontstyle="italic", ha="right", va="baseline")
    return ax


def canvas(fig, x0, y0, w, h):
    """Frameless axes at (x0, y0) mm whose data coordinates are millimetres, 0..w by 0..h: one schematic panel."""
    ax = axes_mm(fig, x0, y0, w, h)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_autoscale_on(False); ax.axis("off")
    return ax


def card(ax, x, y, w, h, fc="white", ec=GRID, lw=0.6, r=1.2, z=0):
    """Rounded box with its lower-left corner at (x, y) mm (never clipped by the axes)."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec,
                       linewidth=lw, zorder=z, clip_on=False)
    ax.add_patch(p)
    return p


def arrow(ax, xy0, xy1, color=INK2, lw=0.7, head=(3.0, 2.4), z=4, **kw):
    """Flow arrow from xy0 to xy1 (mm) with a solid head of (length, width) in points."""
    a = FancyArrowPatch(xy0, xy1, arrowstyle=f"-|>,head_length={head[0]},head_width={head[1] / 2}",
                        mutation_scale=1, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z, clip_on=False,
                        joinstyle="miter", **kw)
    ax.add_patch(a)
    return a


def badge(ax, x, y, s, r=1.25, fc=INK, color="white", z=5):
    """Step number s in a filled circle centred at (x, y) mm."""
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor="none", zorder=z, clip_on=False))
    return ax.text(x, y - 0.03, s, ha="center", va="center", fontsize=FS["note"], fontweight="bold", color=color,
                   zorder=z + 1)


def lines(ax, x, y, rows, dy=LINE, **kw):
    """Text rows from baseline y downwards (mm); returns the baseline after the last row."""
    kw = dict(dict(fontsize=FS["note"], color=INK, ha="left", va="baseline"), **kw)
    for r in rows:
        ax.text(x, y, r, **kw); y -= dy
    return y


def run(ax, x, y, parts, gap=0.0, **kw):
    """One baseline of differently styled pieces [(text, style), ...] set side by side; widths are measured on the
    renderer at the final physical size. Returns the x (mm) after the last piece."""
    rd = ax.figure.canvas.get_renderer(); inv = ax.transData.inverted()
    for s, st in parts:
        t = ax.text(x, y, s, **dict(dict(fontsize=FS["note"], color=INK, ha="left", va="baseline"), **kw, **st))
        x = inv.transform((t.get_window_extent(rd).x1, 0))[0] + gap
    return x


def strip_height(n, pad=1.0, dy=LINE):
    """Height (mm) of a guarantee strip holding n rows of 5.5 pt text: cap height 1.4 mm, descender 0.4 mm."""
    return 2 * pad + 1.8 + (n - 1) * dy


def strip(ax, x, y, w, rows, head="Guarantee", n=None, fc=SHADE, pad=1.0, dy=LINE):
    """Guarantee strip: a shaded band (sized for n rows, default len(rows)) with a bold head word and a statement
    whose first row continues after the head. Returns the band's height (mm) and its texts, whose extents should
    stay inside (x + pad, y, x + w - pad, y + h) (check with overflow)."""
    h = strip_height(n or len(rows), pad, dy)
    card(ax, x, y, w, h, fc=fc, ec="none", lw=0, r=0.8)
    n0 = len(ax.texts); yb = y + h - pad - 1.4
    run(ax, x + pad, yb, [(head, dict(fontweight="bold")), (rows[0], {})], gap=0.9)
    lines(ax, x + pad, yb - dy, rows[1:], dy=dy)
    return h, list(ax.texts)[n0:]


def units(ax, xy, role, s=None, z=3, **kw):
    """Map units in their data-role style; filled roles get a white surface ring that separates neighbours."""
    fc, ec, s0, _ = ROLE[role]
    xy = np.asarray(xy, float).reshape(-1, 2)
    return ax.scatter(xy[:, 0], xy[:, 1], s=s0 if s is None else s, facecolor=fc, edgecolor=ec,
                      linewidths=0.45 if fc == "white" else 0.3, zorder=z, **kw)


def role_handle(role, label=None, ms=2.4):
    fc, _, _, lab = ROLE[role]
    return Line2D([], [], ls="none", marker="o", ms=ms, mfc=fc, mec=INK2 if fc == "white" else fc, mew=0.5,
                  label=label or lab)


def overflow(ax, box, texts=None, tol=0.05):
    """Layout guard for schematic panels: the texts of ax whose rendered extent leaves box = (x0, y0, x1, y1) mm."""
    rd = ax.figure.canvas.get_renderer(); inv = ax.transData.inverted(); bad = []
    for t in ax.texts if texts is None else texts:
        if not t.get_text():
            continue
        bb = t.get_window_extent(rd)
        (a, b), (c, d) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        if a < box[0] - tol or b < box[1] - tol or c > box[2] + tol or d > box[3] + tol:
            bad.append(t.get_text())
    return bad


def keyrow(ax, x, y, items, gap=2.0, **kw):
    """One row of map-unit keys [(role, label), ...] on baseline y (mm): the role's marker, then its label.
    Returns the x (mm) after the last label."""
    rd = ax.figure.canvas.get_renderer(); inv = ax.transData.inverted()
    for role, lab in items:
        units(ax, [x + 0.55, y + 0.55], role, clip_on=False)
        t = ax.text(x + 1.4, y, lab, **dict(dict(fontsize=FS["note"], color=INK, ha="left", va="baseline"), **kw))
        x = inv.transform((t.get_window_extent(rd).x1, 0))[0] + gap
    return x


# ---- map schematics (Figs 1, 2): a shaded surface, admin units, oblique layer stacks of basis functions
MAPRAMP = mpl.colors.LinearSegmentedColormap.from_list("map_surface", ["#f8f7f3", "#d1cdc1"])   # neutral, no family hue


def clip_line(p, box):
    """Pieces of the polyline p (n, 2) inside box = (x0, x1, y0, y1): Liang-Barsky on every segment. Cut lines before
    drawing them: a PDF keeps the full geometry behind a clip path, and the collision audit reads it."""
    x0, x1, y0, y1 = box
    out, cur = [], []
    for a, b in zip(p[:-1], p[1:]):
        d, t0, t1, ok = b - a, 0.0, 1.0, True
        for pp, qq in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
            if pp == 0:
                ok &= qq >= 0
            elif pp < 0:
                t0 = max(t0, qq / pp)
            else:
                t1 = min(t1, qq / pp)
        if not ok or t0 > t1:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = []
            continue
        if t0 > 0 and cur:                              # re-enters the window: start a new piece
            out.append(np.array(cur)); cur = []
        if not cur:
            cur = [a + t0 * d]
        cur.append(a + t1 * d)
        if t1 < 1:                                      # leaves the window
            out.append(np.array(cur)); cur = []
    if len(cur) > 1:
        out.append(np.array(cur))
    return out


def shaded_map(ax, x0, y0, w, h, field=None, levels=8, lines=None, n=(161, 97), z=1):
    """A w x h mm map at (x0, y0) mm: field (a function of (n, 2) map-mm coordinates) as filled MAPRAMP bands, with
    contour lines in colour `lines` if given, or a plain SHADE background without a field; and a thin frame."""
    if field is not None:
        gx, gy = np.meshgrid(np.linspace(0, w, n[0]), np.linspace(0, h, n[1]))
        Z = field(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        lv = np.linspace(Z.min(), Z.max(), levels)
        ax.contourf(gx + x0, gy + y0, Z, levels=lv, cmap=MAPRAMP, zorder=z, antialiased=True)
        if lines:
            ax.contour(gx + x0, gy + y0, Z, levels=lv[1:-1], colors=lines, linewidths=0.3, zorder=z + 0.1)
    else:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=SHADE, edgecolor="none", zorder=z))
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=GRID, lw=0.4, zorder=6))


def voronoi(seeds, box):
    """Voronoi cells of the seeds inside box = (x0, y0, x1, y1), by half-plane clipping: admin units on a map."""
    x0, y0, x1, y1 = box
    out = []
    for i, p in enumerate(seeds):
        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        for k, q in enumerate(seeds):
            if k == i:
                continue
            n, c = q - p, (q @ q - p @ p) / 2                     # keep {z : n.z <= c}
            new = []
            for a, b in zip(poly, np.roll(poly, -1, 0)):
                fa, fb = n @ a - c, n @ b - c
                if fa <= 0:
                    new.append(a)
                if fa * fb < 0:
                    new.append(a + fa / (fa - fb) * (b - a))
            poly = np.array(new)
        out.append(poly)
    return out


def oblique(x0, y0, sx=0.52, kx=0.24, sy=0.19):
    """Oblique view of a map for layer stacks: (x, y) mm on the map -> canvas mm."""
    return lambda p: np.c_[x0 + sx * p[:, 0] + kx * p[:, 1], y0 + sy * p[:, 1]]


def layer_stack(ax, x0, top, size, layers, pitch=4.95, view=None, label_dx=0.8):
    """Oblique map layers stacked downwards from `top` (mm). layers = [(label, draw), ...], top layer first; draw(T, z)
    paints one layer through the map -> canvas transform T above zorder z (None: an empty layer). The bottom layer is
    drawn first, so each layer sits on the one below it. Returns the y (mm) of the stack's bottom edge."""
    view = dict(sx=0.52, kx=0.24, sy=0.19) if view is None else view
    w, h = size; hl = view["sy"] * h; n = len(layers)
    for k in range(n - 1, -1, -1):
        label, draw = layers[k]
        y0 = top - hl - pitch * k
        T = oblique(x0, y0, **view)
        pg = Polygon(T(np.array([[0, 0], [w, 0], [w, h], [0, h]], float)), closed=True, facecolor="white",
                     edgecolor=GREY_L, lw=0.4, zorder=10 + 3 * (n - 1 - k))
        ax.add_patch(pg)
        if draw is not None:
            draw(T, pg.get_zorder())
        corner = T(np.array([[w, h]]))[0]
        ax.text(corner[0] + label_dx, y0 + hl / 2, label, fontsize=FS["note"], color=INK2, ha="left", va="center",
                zorder=30)
    return top - hl - pitch * (n - 1)


def bump_layer(ax, T, centres, radii, size, z, color=VIOLET):
    """Gaussian bumps on one layer: a faint disc and a ring of radius h round each centre, both cut to the map before
    drawing, and the centres."""
    w, h = size; th = np.linspace(0, 2 * np.pi, 97)
    for c, r in zip(centres, radii):
        ring = c + r * np.c_[np.cos(th), np.sin(th)]
        disc = Path(ring, closed=True).clip_to_bbox(Bbox([[0, 0], [w, h]]))
        if len(disc.vertices) > 2:
            ax.add_patch(Polygon(T(disc.vertices), closed=True, facecolor=color, edgecolor="none", alpha=0.10,
                                 zorder=z + 0.5))
        for piece in clip_line(ring, (0, w, 0, h)):
            q = T(piece)
            ax.plot(q[:, 0], q[:, 1], color=color, lw=0.3, alpha=0.7, zorder=z + 0.6)
    cxy = T(np.asarray(centres, float))
    ax.scatter(cxy[:, 0], cxy[:, 1], s=0.9, color=color, linewidths=0, zorder=z + 0.7)


def cell_layer(ax, T, cells, z, fill=None, color=VIOLET):
    """Admin units on one layer: their outlines, with cell number `fill` tinted (one indicator of the basis)."""
    for k, cell in enumerate(cells):
        if k == fill:
            ax.add_patch(Polygon(T(cell), closed=True, facecolor=color, alpha=0.22, edgecolor="none", zorder=z + 0.4))
        ax.add_patch(Polygon(T(cell), closed=True, facecolor="none", edgecolor=INK2, lw=0.35, zorder=z + 0.5))
