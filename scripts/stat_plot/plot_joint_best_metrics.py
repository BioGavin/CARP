#!/usr/bin/env python3
"""Joint reg+cls model at one lambda (default 0.25, the paper's best): metric bars.

Uses the same runs and the same checkpoint choice as plot_joint_lambda_sweep.py /
paper Table C.1 (150M backbone, test metrics at the best *regression* checkpoint),
mean +/- std over the 5 seeds, and renders two bar panels:

  Regression:     MSE, PCC, KTC
  Classification: AUROC, AUPR, Prec, Rec, F1, MCC, Acc

twice: side by side as vertical bars (<name>.pdf) and stacked as horizontal bars
(<name>_hbar.pdf). Regression bars start at 0; the classification value axis is zoomed
to 0.7-1 (its scores all sit in ~0.87-0.99), so read those values from the bar labels.

Usage:
    python3 scripts/stat_plot/plot_joint_best_metrics.py
    python3 scripts/stat_plot/plot_joint_best_metrics.py --lam 0.5
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import C, apply, save, save_csv
from plot_joint_lambda_sweep import load

BAR_FRAC = 0.6  # bar width as a fraction of the metric slot (this figure keeps its own
                # fixed size instead of the shared physical bar width)
REG = [("mse", "MSE"), ("pcc", "PCC"), ("ktc", "KTC")]
CLS = [("auc", "AUROC"), ("aupr", "AUPR"), ("precision", "Prec"), ("recall", "Rec"),
       ("f1", "F1"), ("mcc", "MCC"), ("acc", "Acc")]


def _bar_panel(ax, sub, metrics, color, vmin=0.0, vstep=0.2, horizontal=False):
    """One bar per metric, mean +/- std, value label at the bar end. The value axis runs
    from ``vmin``; ticks stop at 1 (scores cap there), the space past it holds labels."""
    p = np.arange(len(metrics))
    span = (-0.5, len(metrics) - 0.5)
    mu = np.array([sub[m].mean() for m, _ in metrics])
    sd = np.array([sub[m].std(ddof=1) for m, _ in metrics])
    err = dict(lw=0.8, ecolor="#333", capthick=0.8)
    fs = plt.rcParams["xtick.labelsize"] * 0.85
    ticks = np.arange(vmin, 1 + 1e-9, vstep)
    names = [lbl for _, lbl in metrics]
    if horizontal:
        ax.barh(p, mu, BAR_FRAC, xerr=sd, color=color, linewidth=0, capsize=2.5,
                error_kw=err, zorder=2)
        for y, v, e in zip(p, mu, sd):
            ax.annotate(f"{v:.3f}", (v + e, y), xytext=(2, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=fs, zorder=4)
        ax.set_yticks(p, names)
        ax.set_ylim(span[1], span[0])  # inverted: first metric on top
        ax.set_xlim(vmin, 1 + (1 - vmin) * 0.15)
        ax.set_xticks(ticks)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(p, mu, BAR_FRAC, yerr=sd, color=color, linewidth=0, capsize=2.5,
               error_kw=err, zorder=2)
        for x, v, e in zip(p, mu, sd):
            ax.annotate(f"{v:.3f}", (x, v + e), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=fs, zorder=4)
        ax.set_xticks(p, names)
        ax.set_xlim(span)
        ax.set_ylim(vmin, 1 + (1 - vmin) * 0.1)
        ax.set_yticks(ticks)
        ax.grid(axis="x", visible=False)
    return mu, sd


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(root, "results", "03-joint-classreg"))
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--esm", default="esm2_t30_150M",
                    help="keep only runs whose esm_name contains this")
    ap.add_argument("--lam", type=float, default=0.25, help="MSE weight to plot")
    args = ap.parse_args()

    df = load(args.results, args.esm)
    sub = df[np.isclose(df["lam"], args.lam)]
    if sub.empty:
        raise SystemExit(f"no joint runs with lambda={args.lam:g} for --esm {args.esm}")
    print(f"lambda={args.lam:g}: {len(sub)} seeds {sorted(sub['seed'])}")

    apply(font_scale=1.2)
    name = f"joint_lam{args.lam:g}_metrics"
    ratios = [len(REG), len(CLS)]
    for horizontal in (False, True):
        # panel ratios = metric counts, so both panels share one slot size
        if horizontal:  # stacked panels
            fig, axes = plt.subplots(2, 1, figsize=(3.8, 5),
                                     gridspec_kw={"height_ratios": ratios})
        else:  # side by side
            fig, axes = plt.subplots(1, 2, figsize=(6, 2.6),
                                     gridspec_kw={"width_ratios": ratios})
        reg = _bar_panel(axes[0], sub, REG, C[0], horizontal=horizontal)
        axes[0].set_title("Regression")
        # classification scores all sit in 0.87-0.99: zoom to 0.7-1 (axis not from 0)
        cls = _bar_panel(axes[1], sub, CLS, C[1], vmin=0.7, vstep=0.1, horizontal=horizontal)
        axes[1].set_title("Classification")
        fig.tight_layout(**({"h_pad": 1.2} if horizontal else {"w_pad": 1.5}))
        save(fig, os.path.join(args.figdir, name + ("_hbar" if horizontal else "")))

    rows = [{"task": t, "metric": lbl, "mean": m, "std": s}
            for t, mets, (mu, sd) in [("regression", REG, reg), ("classification", CLS, cls)]
            for (_, lbl), m, s in zip(mets, mu, sd)]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.4f"))
    save_csv(table, args.figdir, f"{name}.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
