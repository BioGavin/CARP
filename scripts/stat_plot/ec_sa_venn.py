#!/usr/bin/env python3
"""Area-proportional two-set Venn diagrams, drawn so a count reads as an area.

Circle areas are proportional to set size and the centre distance is solved so the lens
area equals the true intersection, so every region is read as area, not as a schematic.

Library only: figure/ec_sa_overlap_venn.pdf is drawn by plot_ec_sa_overlap.py.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    sys.path.insert(0, p)

import matplotlib.pyplot as plt  # noqa: E402  (after _style picks the Agg backend)

from _style import C, INK2  # noqa: E402

DATA = os.path.join(ROOT, "dataset", "bertampep")

EC_COLOR, SA_COLOR = C[0], C[1]


def pool(fname):
    """-> set of unique, normalised sequences in a BERT-AmPEP60 split."""
    df = pd.read_csv(os.path.join(DATA, fname))
    return set(df["SEQUENCE"].astype(str).str.strip().str.upper())


def lens_area(d, r1, r2):
    """Area of the intersection of two circles, radii r1/r2, centre distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    tri = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - tri


def solve_distance(r1, r2, target):
    """Centre distance whose lens area equals ``target`` (0 < target < min circle area)."""
    lo, hi = abs(r1 - r2) + 1e-9, r1 + r2 - 1e-9
    return brentq(lambda d: lens_area(d, r1, r2) - target, lo, hi, xtol=1e-12)


def draw_venn(ax, n_left, n_right, n_both, labels, unit, note=None):
    """One area-proportional 2-set Venn. ``unit`` converts a count to an area, so
    several panels drawn with the same unit share one scale."""
    r1, r2 = np.sqrt(n_left * unit / np.pi), np.sqrt(n_right * unit / np.pi)
    d = solve_distance(r1, r2, n_both * unit)
    x1, x2 = -d / 2, d / 2

    for x, r, color in ((x1, r1, EC_COLOR), (x2, r2, SA_COLOR)):
        ax.add_patch(plt.Circle((x, 0), r, facecolor=color, alpha=0.42, lw=0, zorder=2))
        ax.add_patch(plt.Circle((x, 0), r, facecolor="none", edgecolor=color,
                                lw=1.4, zorder=3))

    # region counts along the x-axis; a region too thin for its label gets a leader line
    lens_l, lens_r = max(x1 - r1, x2 - r2), min(x1 + r1, x2 + r2)
    span = max(r1, r2)

    def region_label(lo, hi, text, out_x):
        """Label the region spanning [lo, hi]; if it is too thin for the digits,
        put the label at ``out_x`` outside the circles and draw a leader to it."""
        if hi - lo >= 0.30 * span:
            ax.text((lo + hi) / 2, 0, text, ha="center", va="center", zorder=4)
        else:
            ax.annotate(text, xy=((lo + hi) / 2, 0), xytext=(out_x, span * 0.55),
                        ha="center", va="bottom", zorder=5,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=INK2,
                                        shrinkA=1, shrinkB=1.5))

    region_label(x1 - r1, lens_l, f"{n_left - n_both:,}", x1 - r1 - 0.12 * span)
    region_label(lens_r, x2 + r2, f"{n_right - n_both:,}", x2 + r2 + 0.14 * span)
    ax.text((lens_l + lens_r) / 2, 0, f"{n_both:,}", ha="center", va="center",
            fontweight="bold", zorder=4)

    top = max(r1, r2)
    ax.text(x1, -top - 0.14 * top, labels[0], ha="center", va="top", color=EC_COLOR)
    ax.text(x2, top + 0.14 * top, labels[1], ha="center", va="bottom", color=SA_COLOR)
    if note:
        ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", va="top",
                color=INK2, fontsize=plt.rcParams["xtick.labelsize"])

    ax.set_aspect("equal")
    ax.set_xlim(x1 - r1 - 0.30 * top, x2 + r2 + 0.32 * top)
    ax.set_ylim(-top * 1.5, top * 1.5)
    ax.axis("off")


def match_venn_scale(axes):
    """Give several Venn panels one data-per-inch scale, so one area unit is one area.

    draw_venn leaves each axes with limits that hug its own circles, and an equal aspect
    then fits each panel to its own span: a pool whose span is narrower -- the test pool,
    whose S. aureus circle is tiny -- gets drawn LARGER, so the E. coli circle comes out a
    different size in each panel and the shared unit is silently broken. Widening every
    panel to the largest span, about its own centre, restores it."""
    hx = max(ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes) / 2
    hy = max(ax.get_ylim()[1] - ax.get_ylim()[0] for ax in axes) / 2
    for ax in axes:
        for lim, setter, half in ((ax.get_xlim(), ax.set_xlim, hx),
                                  (ax.get_ylim(), ax.set_ylim, hy)):
            mid = (lim[0] + lim[1]) / 2
            setter(mid - half, mid + half)
