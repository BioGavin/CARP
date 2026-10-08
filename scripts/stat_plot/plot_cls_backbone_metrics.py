#!/usr/bin/env python3
"""Native vs regression-fine-tuned ESM-2 classifier: metric bars (paper Table B.1 as a figure).

Reads the per-run classifier JSONs (written by scripts/train_cls.py; the same runs as
stat_cls_esm.py / plot_cls_confusion.py) and renders two grouped-bar panels, mean +/- std
over the 5 seeds of ``result.test_at_best_val``, as horizontal bars with the mean at
each bar end:

  top:    threshold-free metrics: AUROC, AUPR
  bottom: fixed 0.5-threshold metrics: Precision, Recall, F1, MCC, Accuracy

P@25% is omitted: it is ~1.0 for every backbone and carries no contrast.
The differences are ~0.01, so each panel's value axis is zoomed onto its data (it does
not start at 0); read values from the bar labels, not bar lengths.
Hue = backbone kind (native / fine-tuned), shade = size (light 150M, dark 650M).

Usage:
    python3 scripts/stat_plot/plot_cls_backbone_metrics.py
    python3 scripts/stat_plot/plot_cls_backbone_metrics.py --results <dir> --figdir <dir>
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from _style import apply, fit_size, group_layout, save, save_csv
from stat_cls_esm import load, per_seed_frame, summary_frame

# (backbone, size) in display order -> (legend label, color)
CONFIGS = [
    (("native", "150M"), "Native (150M)", "#86b4ec"),
    (("native", "650M"), "Native (650M)", "#1f5fb0"),
    (("finetuned", "150M"), "Fine-tuned (150M)", "#f5a27f"),
    (("finetuned", "650M"), "Fine-tuned (650M)", "#c4471a"),
]
THRESH_FREE = [("auc", "AUROC"), ("aupr", "AUPR")]
THRESH_05 = [("precision", "Prec"), ("recall", "Rec"), ("f1", "F1"), ("mcc", "MCC"),
             ("acc", "Acc")]


def _bar_panel(ax, sm, metrics, group_frac=None, horizontal=True, value_scale=0.75):
    """Grouped bars per metric (one bar per backbone config), mean +/- std, value at the
    bar end. The value axis is zoomed onto the data, with room past the bars for the
    labels. ``group_frac``: see ``_style.group_layout``."""
    pos, centers, span_y, bar_w = group_layout(len(metrics), len(CONFIGS), group_frac)
    lo, hi = np.inf, -np.inf
    for k, ((bk, size), label, color) in enumerate(CONFIGS):
        row = sm[(sm["backbone"] == bk) & (sm["size"] == size)]
        if row.empty:
            continue
        row = row.iloc[0]
        p = pos[k]
        mu = np.array([row[f"{m}_mean"] for m, _ in metrics])
        sd = np.array([row[f"{m}_std"] for m, _ in metrics])
        err = dict(capsize=1.5, error_kw=dict(lw=0.7, ecolor="#333", capthick=0.7), zorder=2)
        fs = plt.rcParams["xtick.labelsize"] * value_scale
        if horizontal:
            ax.barh(p, mu, bar_w, xerr=sd, color=color, linewidth=0, label=label, **err)
            for y, v, e in zip(p, mu, sd):
                ax.annotate(f"{v:.3f}", (v + e, y), xytext=(2, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=fs)
        else:
            ax.bar(p, mu, bar_w, yerr=sd, color=color, linewidth=0, label=label, **err)
            for x, v, e in zip(p, mu, sd):
                ax.annotate(f"{v:.3f}", (x, v + e), xytext=(0, 1.5), textcoords="offset points",
                            ha="center", va="bottom", rotation=90, fontsize=fs)
        lo, hi = min(lo, (mu - sd).min()), max(hi, (mu + sd).max())
    span = hi - lo
    # room past the bars for the value labels: horizontal text needs more than rotated
    vlim = (lo - span * 0.3, hi + span * (0.5 if horizontal else 0.35))
    set_vlim, get_vticks, set_vticks = ((ax.set_xlim, ax.get_xticks, ax.set_xticks) if horizontal
                                        else (ax.set_ylim, ax.get_yticks, ax.set_yticks))
    set_vlim(vlim)
    # scores cap at 1: room past it is only for labels, so no tick past 1.0
    set_vticks([t for t in get_vticks() if t <= 1.0 + 1e-9])
    set_vlim(vlim)
    names = [lbl for _, lbl in metrics]
    if horizontal:
        ax.set_yticks(centers, names)
        ax.set_ylim(span_y[1], span_y[0])  # inverted: first metric / config on top
        ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(centers, names)
        ax.set_xlim(span_y)
        ax.grid(axis="x", visible=False)


def draw(axes, sm):
    """Threshold-free metrics into ``axes[0]``, 0.5-threshold metrics into ``axes[1]``.
    -> legend (handles, labels)."""
    _bar_panel(axes[0], sm, THRESH_FREE)
    _bar_panel(axes[1], sm, THRESH_05)
    return axes[1].get_legend_handles_labels()


def draw_single(ax, sm, group_frac=None, value_scale=0.75):
    """All seven metrics in one panel of vertical bars on a shared value axis.
    -> legend (handles, labels)."""
    _bar_panel(ax, sm, THRESH_FREE + THRESH_05, group_frac, horizontal=False,
               value_scale=value_scale)
    return ax.get_legend_handles_labels()


def plot(sm, figpath):
    # stacked (the panels hold 2 and 5 metric groups); height ratios = group counts so
    # both panels share one bar scale
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 7),
                             gridspec_kw={"height_ratios": [len(THRESH_FREE), len(THRESH_05)]})
    handles, labels = draw(axes, sm)
    fit_size(fig, axes[0], lambda: fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.93), h_pad=1.2),
             axis="y")
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    save(fig, figpath)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(root, "results", "02-esm-classifier"))
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--fig-name", default="cls_backbone_metrics")
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        raise SystemExit(f"no usable result JSONs in {args.results}")
    sm = summary_frame(per_seed_frame(rows))
    keep = ["backbone_label", "size", "n_seeds"] + [
        f"{m}_{s}" for m, _ in THRESH_FREE + THRESH_05 for s in ("mean", "std")]
    save_csv(sm[keep], args.figdir, f"{args.fig_name}.csv", float_format="%.4f")

    apply(font_scale=1.2)
    plot(sm, os.path.join(args.figdir, args.fig_name))


if __name__ == "__main__":
    main()
