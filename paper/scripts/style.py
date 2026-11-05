"""
style.py -- one look for every figure of the paper.

Sizes follow Physical Review E: one column 86 mm, two columns 178 mm; figures are saved
at their final size (no scaling in LaTeX), lettering 6.5-7.5 pt in TeX Gyre Heros
(a Helvetica clone), panel labels (a), (b), ... placed after layout so that they align.

Colours keep one meaning each across the paper (a relaxed version of the talk's contract):
    INDIGO  genuine memory: data, and estimators that a correct analysis uses
    CORAL   artefacts: what a flawed analysis reports, spurious memory
    TEAL    fields: individual and time fields, walls, hidden states, common drives
    PLUM    social coupling: the others' past
    greys   chance, null models, raw data
Species are marked by hue families: larvae indigo, termites ochre, T cells slate.
"""
from __future__ import annotations

import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_rgb

MM = 1 / 25.4
COL, FULL = 86.0, 178.0

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")

_GYRE = os.environ.get("TEX_GYRE_DIR", os.path.expanduser(r"~/AppData/Local/Programs/MiKTeX/fonts/opentype/public/tex-gyre"))
for _f in glob.glob(os.path.join(_GYRE, "texgyreheros-*.otf")):
    fm.fontManager.addfont(_f)
FONT = "Arial"

INDIGO, CORAL, TEAL, PLUM = "#3B5BA5", "#E4774E", "#17A589", "#9B6BB0"
OCHRE, SLATE = "#C8922A", "#5F6B7A"
INK, INK2, MUTED = "#151515", "#484846", "#8C8B86"
GRID, AXIS = "#E7E6E1", "#B9B8B0"
BAND, BAND2 = "#ECEBE6", "#DAD9D2"
WHITE = "#FFFFFF"


def mix(c, w, base="#FFFFFF"):
    """w = 1 gives c, w = 0 gives base."""
    a, b = np.array(to_rgb(c)), np.array(to_rgb(base))
    return tuple(w * a + (1 - w) * b)


def ramp(c, n, lo=0.35, hi=1.0, dark=False):
    """n tints of one colour, light to full (or full to dark)."""
    if dark:
        return [mix(c, 1 - t, "#000000") for t in np.linspace(0, 0.55, n)]
    return [mix(c, t) for t in np.linspace(lo, hi, n)]


# sequential map for order parameters: paper white -> indigo -> deep plum-black
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#F7F6F2", "#D9DEEC", "#A9B6DA", "#6F86C4", INDIGO, "#2A3E78", "#1B2347"])
# raw tracks: short flights light, long ones dark (log colour scale)
RAW = LinearSegmentedColormap.from_list("raw", ["#DAD8D0", "#A9B3CC", "#6F84BE", INDIGO, "#1B2550"])

RC = {
    "font.family": FONT, "font.size": 7.0,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.cal": "STIXGeneral:italic",
    "mathtext.fallback": "stixsans",
    "axes.labelsize": 7.2, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.titlesize": 7.2, "axes.titlecolor": INK, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.titlepad": 3.0, "axes.labelpad": 2.5,
    "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 1.3, "ytick.minor.size": 1.3, "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.45, "axes.axisbelow": True,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "lines.solid_capstyle": "round",
    "lines.markeredgewidth": 0.7, "patch.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 6.3, "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3, "legend.columnspacing": 1.0,
    "errorbar.capsize": 0,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "pdf.fonttype": 42, "figure.dpi": 220, "savefig.dpi": 600,
}


def setup():
    sns.set_theme(style="ticks", rc=RC)
    mpl.rcParams.update(RC)


def fig(w_mm, h_mm, **kw):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM), **kw)


def despine(ax, left=False, bottom=False):
    sns.despine(ax=ax, left=left, bottom=bottom)


def text(ax, x, y, s, c=INK2, **kw):
    kw.setdefault("fontsize", 6.4)
    return ax.text(x, y, s, color=c, **kw)


def label_panels(figure, axes, letters=None, dx_pt=0.0, dy_pt=3.0, fontsize=8.2, align_rows=True):
    """Put (a), (b), ... above the top-left corner of each axes' tight box (tick labels
    included), aligned on a common baseline for axes whose tops coincide."""
    letters = letters or [f"({chr(97 + i)})" for i in range(len(axes))]
    figure.canvas.draw()
    r = figure.canvas.get_renderer()
    inv = figure.transFigure.inverted()
    boxes = []
    for ax in axes:
        tb = ax.get_tightbbox(r)
        x0 = inv.transform((tb.x0, tb.y0))[0]
        ytop = inv.transform(ax.get_window_extent(r).corners()[1])[1]   # top of the axes frame
        boxes.append([x0, ytop])
    if align_rows:
        for i, (x0, yt) in enumerate(boxes):
            same = [b[1] for b in boxes if abs(b[1] - yt) < 0.02]
            boxes[i][1] = max(same)
    w, h = figure.get_size_inches()
    for (x0, yt), s in zip(boxes, letters):
        figure.text(x0 + dx_pt / 72 / w, yt + dy_pt / 72 / h, s, fontsize=fontsize, fontweight="bold",
                    color=INK, ha="left", va="bottom")


def heads(figure, items, dy_pt=3.0, fontsize=8.2, title_size=6.6, gap_pt=3.5, align_rows=True, one_row=False):
    """Panel heads: a bold letter at the top-left of each axes' tight box, followed on the same
    baseline by a short descriptive title.  items: list of (ax, letter, title or None)."""
    figure.canvas.draw()
    r = figure.canvas.get_renderer()
    inv = figure.transFigure.inverted()
    w, h = figure.get_size_inches()
    pos = []
    for ax, _, _ in items:
        tb = ax.get_tightbbox(r)
        x0 = inv.transform((tb.x0, tb.y0))[0]
        if ax.axison:
            ytop = inv.transform(ax.get_window_extent(r).corners()[1])[1]
        else:
            ytop = inv.transform((tb.x1, tb.y1))[1]
        pos.append([x0, ytop])
    if one_row:
        top = max(b[1] for b in pos)
        for p in pos:
            p[1] = top
    elif align_rows:
        for i, (_, yt) in enumerate(pos):
            pos[i][1] = max(b[1] for b in pos if abs(b[1] - yt) < 0.03)
    for (x0, yt), (ax, lab, ttl) in zip(pos, items):
        y = yt + dy_pt / 72 / h
        t = figure.text(x0, y, lab, fontsize=fontsize, fontweight="bold", color=INK, ha="left", va="baseline")
        if ttl:
            bb = t.get_window_extent(r)
            xt = inv.transform((bb.x1, bb.y0))[0] + gap_pt / 72 / w
            figure.text(xt, y, ttl, fontsize=title_size, color=INK2, ha="left", va="baseline")


def save(figure, name):
    os.makedirs(FIGS, exist_ok=True)
    figure.savefig(os.path.join(FIGS, name + ".pdf"))
    figure.savefig(os.path.join(FIGS, name + ".png"), dpi=300)
    plt.close(figure)
    print("   saved", name)
