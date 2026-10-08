#!/usr/bin/env python3
"""How the S. aureus pools overlap the E. coli training pool, as two fixed-size pages.

  figure/ec_sa_overlap_venn.pdf     (12.2 x 7.8 cm) area-proportional set diagrams of
                                    E. coli train against each S. aureus pool
  figure/ec_sa_overlap_scatter.pdf  (7.8 x 7.8 cm)  E. coli vs S. aureus pMIC on the
                                    peptides measured in both, and the map fitted to them

Together: most S. aureus peptides are not new molecules -- 81% of the training pool and
84% of the test pool are already in E. coli training -- and for those that are shared, the
E. coli label already predicts much of the S. aureus label.

Everything is counted per **unique peptide** (duplicate rows averaged first), so the 2,368
shared training peptides here are exactly the set ec_sa_baselines.ec_label_copy() fits the
EC->SA map on and the "shared" ridge probe targets: the line drawn on the scatter *is* that
map. The raw-row counts differ (2,395 / 273) because the S. aureus files repeat peptides;
never mix the two bases inside one figure.

Usage:
    python3 scripts/stat_plot/plot_ec_sa_overlap.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    sys.path.insert(0, p)

import matplotlib.pyplot as plt  # noqa: E402  (after _style picks the Agg backend)

from _style import apply, save, save_csv  # noqa: E402
from carp.dataset import read_split  # noqa: E402
from ec_sa_baselines import dedup, panel_a  # noqa: E402
from ec_sa_venn import DATA, draw_venn, match_venn_scale, pool as venn_pool  # noqa: E402

POOLS = ["train-SA.csv", "test-SA.csv"]
# Page sizes in cm. The left page holds two set diagrams side by side; the right one is
# square, because the scatter is an equal-aspect, equal-limit panel and a square page
# spends no margin on empty sides.
PAGE_CM = {"venn": (12.2, 7.8), "scatter": (7.8, 7.8)}
# Margins in inches. The Venns label themselves inside axes that have no spines or ticks,
# so they need almost none; the scatter needs an ordinary y label and tick column.
PADS = {"venn": {"left": 0.06, "right": 0.06, "top": 0.06, "bottom": 0.06},
        "scatter": {"left": 0.68, "right": 0.12, "top": 0.10, "bottom": 0.55}}


def shared_peptides(fname, ec):
    """-> number of unique peptides in ``fname`` that are also in the E. coli train pool.

    Duplicates are averaged first, so a peptide counts once however often it was measured
    -- the same basis dedup() puts the scatter on."""
    sa = pd.read_csv(os.path.join(DATA, fname))
    sa["S"] = sa["SEQUENCE"].astype(str).str.strip().str.upper()
    return int(sa.groupby("S", as_index=False)["SA_MIC"].mean()["S"].isin(ec).sum())


def page(kind, ncol=1, wspace=0.0):
    """-> (figure, [axes]) on the exact page PAGE_CM asks for, margins already applied."""
    w, h = (c / 2.54 for c in PAGE_CM[kind])
    pads = PADS[kind]
    fig = plt.figure(figsize=(w, h))
    span = 1 - (pads["left"] + pads["right"]) / w
    cell = span / (ncol + (ncol - 1) * wspace)
    axes = [fig.add_axes((pads["left"] / w + i * cell * (1 + wspace), pads["bottom"] / h,
                          cell, 1 - (pads["top"] + pads["bottom"]) / h))
            for i in range(ncol)]
    return fig, axes


def fixed_size_save(fig, figdir, stem):
    """Save at the figure's own size. The shared rcParams crop to a tight bbox, and
    bbox_inches=None does NOT turn that off -- matplotlib reads None as 'use rcParams' --
    so an explicit Bbox is the only way to get the page PAGE_CM asks for."""
    save(fig, os.path.join(figdir, stem), bbox_inches=fig.bbox_inches, pad_inches=0)


def figure_venn(ec, counts, figdir):
    """Both pools against E. coli train, on one shared area unit so the E. coli circle is
    the same size in each and the two S. aureus pools are directly comparable."""
    fig, axes = page("venn", ncol=2, wspace=0.03)
    unit = 1.0 / len(ec)
    for ax, fname in zip(axes, POOLS):
        sa = venn_pool(fname)
        which = "train" if "train" in fname else "test"
        draw_venn(ax, len(ec), len(sa), counts[fname],
                  (r"$\it{E.}$ $\it{coli}$ train" + f"\n{len(ec):,}",
                   r"$\it{S.}$ $\it{aureus}$ " + which + f"\n{len(sa):,}"), unit)
    match_venn_scale(axes)       # or each panel is scaled to its own span; see the helper
    fixed_size_save(fig, figdir, "ec_sa_overlap_venn")


def figure_scatter(shared, slope, intercept, figdir):
    """E. coli vs S. aureus pMIC on the shared peptides, with the fitted map."""
    fig, (ax,) = page("scatter")
    row = panel_a(ax, shared, slope, intercept, square=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    fixed_size_save(fig, figdir, "ec_sa_overlap_scatter")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figure"))
    args = ap.parse_args()

    ec = venn_pool("train-EC.csv")
    counts = {f: shared_peptides(f, ec) for f in POOLS}

    ec_tr = dedup(*read_split("E. coli", "train"))
    shared = dedup(*read_split("S. aureus", "train")).merge(ec_tr, on="seq",
                                                            suffixes=("_sa", "_ec"))
    assert len(shared) == counts[POOLS[0]], "the scatter must plot the Venn's intersection"
    slope, intercept = np.polyfit(shared.pmic_ec, shared.pmic_sa, 1)

    apply(font_scale=1.5)
    os.makedirs(args.figdir, exist_ok=True)
    figure_venn(ec, counts, args.figdir)
    row = figure_scatter(shared, slope, intercept, args.figdir)

    out = pd.DataFrame([{"pool": f, "n_unique": len(venn_pool(f)), "n_shared": counts[f]}
                        for f in POOLS] + [{**row, "pool": "scatter"}])
    print(out.to_string(index=False, float_format="%.4f"))
    save_csv(out, args.figdir, "ec_sa_overlap.csv")


if __name__ == "__main__":
    main()
