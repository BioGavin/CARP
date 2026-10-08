#!/usr/bin/env python3
"""Paper Figure 2 (figure/Fig2.pdf), drawn from the raw results into one figure.

Three full-width rows, all bars vertical, every axes boxed (top/right spines without
ticks), rows tagged A / B / C:

  A: ESM-2 size x frozen/fine-tune regression (MSE, PCC, KTC)
     -- plot_esm_regression.draw_vbar
  B: native vs regression-fine-tuned classifier, all seven metrics in one panel
     -- plot_cls_backbone_metrics.draw_single
  C: joint training: lambda sweep, then the lambda=0.25 regression / classification bars
     -- plot_joint_lambda_sweep.draw + plot_joint_best_metrics._bar_panel

Same data, checkpoint choices and styling as the standalone figures; bars use a plain
proportional layout (``group_frac``), not the shared physical bar width.

Usage:
    python3 scripts/stat_plot/plot_overview_combined.py
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import plot_cls_backbone_metrics as cls_fig
import plot_esm_regression as reg_fig
from _style import C, apply, save
from plot_joint_best_metrics import CLS, REG, _bar_panel
from plot_joint_lambda_sweep import draw as draw_sweep
from plot_joint_lambda_sweep import load as load_joint
from plot_joint_lambda_sweep import summarize as summarize_joint
from stat_cls_esm import load as load_cls
from stat_cls_esm import per_seed_frame, summary_frame


def build(path, reg_df, cls_sm, joint, joint_lam):
    """A / B / C stacked as full-width rows, all bars vertical, every axes boxed."""
    fig = plt.figure(figsize=(13.5, 12.5), layout="constrained")
    fig.get_layout_engine().set(hspace=0.03, wspace=0.03)
    row_a, row_b, row_c = fig.subfigures(3, 1, height_ratios=[1, 1.05, 1], hspace=0.03)
    for sf, tag in zip((row_a, row_b, row_c), "ABC"):
        sf.suptitle(tag, x=0.005, ha="left", fontweight="bold", fontsize=20)

    # A: regression across ESM-2 sizes, one panel per metric
    # value labels at 0.95x the tick size (standalone figures use 0.75x)
    handles, labels = reg_fig.draw_vbar(row_a.subplots(1, 3), reg_df, reg_fig.PLOT_METRICS_3,
                                        group_frac=0.7, value_scale=0.95)
    row_a.legend(handles, labels, ncol=2, loc="outside upper center")

    # B: classifier on native vs fine-tuned backbones, all metrics in one panel
    handles, labels = cls_fig.draw_single(row_b.subplots(), cls_sm, group_frac=0.75,
                                          value_scale=0.95)
    row_b.legend(handles, labels, ncol=4, loc="outside upper center")

    # C: joint training -- lambda sweep, then the one-lambda regression / classification bars
    ax_sweep, ax_reg, ax_cls = row_c.subplots(
        1, 3, gridspec_kw={"width_ratios": [7.5, len(REG), len(CLS)]})
    draw_sweep(ax_sweep, summarize_joint(joint))
    _bar_panel(ax_reg, joint_lam, REG, C[0])
    ax_reg.set_title("Regression")
    _bar_panel(ax_cls, joint_lam, CLS, C[1], vmin=0.7, vstep=0.1)
    ax_cls.set_title("Classification")

    # close every axes: top/right spines as plain lines, no extra ticks
    for ax in fig.get_axes():
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    save(fig, path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    res = os.path.join(root, "results")
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--esm", default="esm2_t30_150M", help="backbone of the joint runs")
    ap.add_argument("--lam", type=float, default=0.25, help="MSE weight for the joint bars")
    ap.add_argument("--fig-name", default="Fig2")
    args = ap.parse_args()

    reg_df = reg_fig.summarize(reg_fig.load(os.path.join(res, "01-esm-regressor")))
    cls_sm = summary_frame(per_seed_frame(load_cls(os.path.join(res, "02-esm-classifier"))))
    joint = load_joint(os.path.join(res, "03-joint-classreg"), args.esm)
    joint_lam = joint[np.isclose(joint["lam"], args.lam)]
    if reg_df.empty or cls_sm.empty or joint_lam.empty:
        raise SystemExit("missing results for one of the three columns")

    apply(font_scale=1.8)
    build(os.path.join(args.figdir, args.fig_name), reg_df, cls_sm, joint, joint_lam)

if __name__ == "__main__":
    main()
