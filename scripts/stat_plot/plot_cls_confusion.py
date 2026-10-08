#!/usr/bin/env python3
"""Confusion-matrix panel (paper figure) for the ESM-2 classifier comparison.

Per-example predictions are not stored, but the confusion matrix at the 0.5
threshold is exactly recoverable from precision, recall, n, and prevalence
(prevalence 0.5 => P = N = n/2). For a chosen seed this renders a 2x2 panel of
the four backbones (native/fine-tuned x 150M/650M), each a row-normalised
confusion matrix with integer counts and row percentages, and writes PDF+PNG.

Usage:
    python3 scripts/stat_plot/plot_cls_confusion.py --seed 0
    python3 scripts/stat_plot/plot_cls_confusion.py --seed 0 --results <dir> --figdir <dir>
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from _style import C, FULL_W, apply, save

# single-hue sequential ramp (light -> the categorical blue -> deep blue)
CMAP = LinearSegmentedColormap.from_list("carp_blue", ["#f4f8fd", C[0], "#0d3a73"])

# (file stem, panel title) in reading order
CONFIGS = [
    ("native_backbone_150M", "Native (150M)"),
    ("native_backbone_650M", "Native (650M)"),
    ("finetuned_backbone_150M", "Fine-tuned (150M)"),
    ("finetuned_backbone_650M", "Fine-tuned (650M)"),
]


def confusion(res):
    """-> (cm 2x2, P, N). Rows = true [active, inactive], cols = pred [active, inactive]."""
    n = res["n"]
    P = round(res["prevalence"] * n)
    N = n - P
    rec, prec = res["recall"], res["precision"]
    tp = rec * P
    fp = tp * (1 - prec) / prec if prec > 0 else 0.0
    fn = P - tp
    tn = N - fp
    cm = np.array([[tp, fn], [fp, tn]], dtype=float)
    return np.rint(cm).astype(int), P, N


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    default_results = os.path.join(root, "results", "02-esm-classifier")
    default_figdir = os.path.join(root, "figure")

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--organism", default="Ecoli")
    p.add_argument("--r", default="7", help="train neg:pos ratio tag in the filename")
    p.add_argument("--results", default=default_results, help="dir of result JSONs")
    p.add_argument("--figdir", default=default_figdir, help="output dir for the PDF")
    args = p.parse_args()
    figdir = args.figdir
    os.makedirs(figdir, exist_ok=True)

    apply()
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.1))
    tick_lab = ["Active", "Inactive"]

    for col, (ax, (stem, title)) in enumerate(zip(axes.ravel(), CONFIGS)):
        fn = f"{stem}_{args.organism}_r{args.r}_seed{args.seed}.json"
        res = json.load(open(os.path.join(args.results, fn)))["result"]["test_at_best_val"]
        cm, P, N = confusion(res)
        row_tot = cm.sum(axis=1, keepdims=True)
        frac = cm / row_tot  # row-normalised => diagonal of row 0 is recall

        ax.imshow(frac, cmap=CMAP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if frac[i, j] > 0.55 else "#0b0b0b"
                ax.text(j, i - 0.08, f"{cm[i, j]}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")
                ax.text(j, i + 0.22, f"{frac[i, j]:.1%}", ha="center", va="center",
                        color=color, fontsize=7)
        ax.set_title(title)
        ax.set_xticks([0, 1], tick_lab)
        ax.set_xlabel("Predicted")
        if col == 0:
            ax.set_yticks([0, 1], tick_lab, rotation=90, va="center")
            ax.set_ylabel("True")
        else:
            ax.set_yticks([0, 1], ["", ""])
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.tight_layout(w_pad=0.6)
    save(fig, os.path.join(figdir, f"cls_confusion_{args.organism}_seed{args.seed}"))


if __name__ == "__main__":
    main()
