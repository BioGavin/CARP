#!/usr/bin/env python3
"""Combined joint-training figure: lambda sweep on top, lambda=0.25 metric bars below.

Top:    plot_joint_lambda_sweep.draw -- AUROC (left axis) and MSE (right axis) vs lambda.
Bottom: plot_joint_best_metrics._bar_panel as horizontal bars -- Regression (MSE, PCC,
        KTC) above Classification (AUROC ... Acc) at --lam (default 0.25).

Drawn into one figure from the same data (not pasted PDFs), so fonts and line widths
match exactly. Output: figure/joint_combined.pdf.

Usage:
    python3 scripts/stat_plot/plot_joint_combined.py
    python3 scripts/stat_plot/plot_joint_combined.py --lam 0.5
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from _style import C, apply, save
from plot_joint_best_metrics import CLS, REG, _bar_panel
from plot_joint_lambda_sweep import draw, load, summarize


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(root, "results", "03-joint-classreg"))
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--esm", default="esm2_t30_150M",
                    help="keep only runs whose esm_name contains this")
    ap.add_argument("--lam", type=float, default=0.25, help="MSE weight for the bar panels")
    ap.add_argument("--fig-name", default="joint_combined")
    args = ap.parse_args()

    df = load(args.results, args.esm)
    if df.empty:
        raise SystemExit(f"no joint runs for --esm {args.esm} in {args.results}")
    sub = df[np.isclose(df["lam"], args.lam)]
    if sub.empty:
        raise SystemExit(f"no joint runs with lambda={args.lam:g} for --esm {args.esm}")

    apply(font_scale=1.2)
    # the two source figures are 4.6 x 3.2 in (sweep) and 3.8 x 5 in (bars)
    fig = plt.figure(figsize=(4.6, 8.6), layout="constrained")
    fig.get_layout_engine().set(hspace=0.04)  # room between stacked bar panels
    top, bottom = fig.subfigures(2, 1, height_ratios=[3.2, 5.2], hspace=0.04)
    draw(top.subplots(), summarize(df))
    ax_reg, ax_cls = bottom.subplots(2, 1, gridspec_kw={"height_ratios": [len(REG), len(CLS)]})
    _bar_panel(ax_reg, sub, REG, C[0], horizontal=True)
    ax_reg.set_title("Regression")
    _bar_panel(ax_cls, sub, CLS, C[1], vmin=0.7, vstep=0.1, horizontal=True)
    ax_cls.set_title("Classification")
    save(fig, os.path.join(args.figdir, args.fig_name))


if __name__ == "__main__":
    main()
