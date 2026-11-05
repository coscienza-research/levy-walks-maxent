"""
common.py -- shared imports, paths, labels and theory/drawing helpers for the figure scripts.

Each figure lives in its own module (fig1.py, fig2.py, ...) with a function make(); run
`python make_figs.py [fig1 fig2 ...]` to rebuild them.  Numbers are read from
../../results/*.json (experiment scripts in ../../code) and ../data/*.json (compute.py).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code")
RES = os.path.join(HERE, "..", "..", "results")
DATA = os.path.join(HERE, "..", "data")
for _p in (CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import safe  # noqa: E402,F401  (single-threaded BLAS, below-normal priority)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401
from matplotlib.colors import LogNorm, LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch  # noqa: E402

import style as S  # noqa: E402

S.setup()

_DATA_ROOT = os.environ.get("LEVY_DATA_DIR", r"Z:/Datasets/Levy-walks")
DROS = os.path.join(_DATA_ROOT, "drosophila_larvae_sims2019")
TERM = os.path.join(_DATA_ROOT, "termites_castiblanco_zenodo3628061")
COND_EX = "shi/+ 33 Deg"
SHORT = {"BL/+ 22 Deg": "BL/+ 22 °C", "shi/+ 22 Deg": "shi/+ 22 °C", "New Blsens-rprhid_control": "rpr,hid control",
         "BL/+ 33 Deg": "BL/+ 33 °C", "shi/+ 33 Deg": "shi/+ 33 °C", "MB247/+": "MB247/+",
         "MB247-shi": "MB247>shi", "BL-shi": "BL>shi", "BLsens-shi": "BLsens>shi",
         "New BLsens-rprhid": "BLsens>rpr,hid"}
GROUP = {"BL/+ 22 Deg": "control", "shi/+ 22 Deg": "control", "New Blsens-rprhid_control": "control",
         "BL/+ 33 Deg": "control", "shi/+ 33 Deg": "control", "MB247/+": "control",
         "MB247-shi": "block", "BL-shi": "block", "BLsens-shi": "block", "New BLsens-rprhid": "ablation"}
GMARK = {"control": "o", "block": "s", "ablation": "D"}

# grey ramp for the order parameter of the memory chain (Pi = 0 Pareto phase, 1 condensed)
PHASE = LinearSegmentedColormap.from_list(
    "phase", ["#FBFAF7", "#EFEEE9", "#DCDAD3", "#BDBBB3", "#93918A", "#5E5D58", "#33322F"])


def jload(path):
    with open(path) as f:
        return json.load(f)


def res(name):
    return jload(os.path.join(RES, name))


def dat(name):
    return jload(os.path.join(DATA, name))


# ----------------------------------------------------------------------------- theory
def stationary(mu, beta, L, n=4001):
    """Exact stationary density of the K = 1 causal chain in x = ln(l/lmin) in [0, L]:
    pi(x) ~ exp(h' x) Z(h' + J x), with h' = 1 - mu and J = beta / L."""
    from maxent_levy import pl_logZ
    hp, J = 1.0 - mu, beta / L
    x = np.linspace(0.0, L, n)
    lp = hp * x + pl_logZ(hp + J * x, L)
    lp -= lp.max()
    p = np.exp(lp)
    p /= np.trapezoid(p, x)
    return x, p


def order_parameter(mu, beta, L, n=1201):
    """Pi = fraction of stationary flights longer than sqrt(lmin lmax)."""
    x, p = stationary(mu, beta, L, n)
    return float(np.trapezoid(p * (x > L / 2), x))


def apparent_mu(mu, beta, L, n=4001):
    """Exponent returned by an i.i.d. truncated-Pareto fit to the stationary law."""
    from maxent_levy import _invert_mean
    x, p = stationary(mu, beta, L, n)
    return 1.0 - _invert_mean(np.trapezoid(p * x, x), L)


def ccdf_of(x, p):
    cdf = np.cumsum(p) * (x[1] - x[0])
    return np.clip(1 - cdf / cdf[-1], 1e-12, None)


# ----------------------------------------------------------------------------- drawing
def draw_track(ax, xy, starts, lens, vmin=1.0, vmax=30.0, lw=0.9, cmap=None, under="#E4E3DE"):
    cmap = cmap or S.RAW
    norm = LogNorm(vmin=vmin, vmax=vmax)
    st = list(starts) + [len(xy) - 1]
    ax.plot(xy[:, 0], xy[:, 1], color=under, lw=0.45, zorder=1)
    for i in np.argsort(lens):                       # long flights on top
        a, b = xy[st[i]], xy[st[i + 1]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(norm(np.clip(lens[i], vmin, vmax))), lw=lw,
                zorder=2, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")
    return norm


def draw_spin(ax, cx, cy, r, u, phi=0.55, col=S.INDIGO, lw=0.9, proj=True, fill="#F3F4F8"):
    """A classical Heisenberg spin on a sphere of radius r; u = cos(polar angle)."""
    ax.add_patch(Circle((cx, cy), r, facecolor=fill, edgecolor=S.AXIS, lw=0.5, zorder=2))
    ax.add_patch(Ellipse((cx, cy), 2 * r, 0.62 * r, facecolor="none", edgecolor=S.AXIS, lw=0.4,
                         ls=(0, (2, 1.2)), zorder=2))
    s = np.sqrt(max(0.0, 1 - u * u))
    tx, ty = cx + r * s * np.cos(phi) * 0.95, cy + r * u * 0.95 + r * s * np.sin(phi) * 0.16
    if proj:
        ax.plot([cx, cx], [cy - r, cy + r], color=S.MUTED, lw=0.45, zorder=2)
        ax.plot([tx, cx], [ty, ty], color=col, lw=0.5, ls=(0, (1.2, 1.0)), zorder=3)
        ax.plot([cx - 0.12 * r, cx + 0.12 * r], [ty, ty], color=col, lw=1.0, zorder=4, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((cx, cy), (tx, ty), arrowstyle="-|>", mutation_scale=4.5, color=col,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=4))


def larva_track():
    """The example larva (shi/+ at 33 C) used in Figs. 1 and 5: positions and flights."""
    from exp_drosophila import get_flights
    key, flights, starts, _ = get_flights(0.5)
    fname = key.csv[key.Treatment == COND_EX].iloc[3]
    xy = pd.read_csv(os.path.join(DROS, fname))[["X", "Y"]].to_numpy()
    return xy, np.asarray(flights[fname]), starts[fname]
