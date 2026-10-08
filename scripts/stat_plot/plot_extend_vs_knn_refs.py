#!/usr/bin/env python3
"""S. aureus data efficiency against the two label-copy baselines, MSE over PCC.

The frozen-head sweep from results/04-extend (joint E. coli backbone vs native ESM-2)
plotted against what pure memorisation would have achieved on the same 328 test rows.
The legend only names the baselines, so the caption has to carry their definitions:

  EC-1NN  For each S. aureus test peptide, find its nearest neighbour among the 3638
          E. coli *training* peptides (identity = LCS / max length, from a global
          alignment), take that peptide's E. coli pMIC, and map it onto the S. aureus
          scale with SA = a*EC + b, where a and b are least-squares fitted on the 2368
          peptides measured in both organisms -- training pools only, two parameters, no
          test label. The rescaling matters for MSE (0.489 mapped vs 0.577 raw) but not
          for PCC, which is invariant under a positive affine map, so the PCC line sits
          at 0.578 either way. No S. aureus training label enters at all, so the line is
          flat in n. 83% of the test peptides have an exact match in the E. coli training
          pool, which is what makes this a memorisation reference rather than a
          homology-based prediction.

A trained head only earns its keep where it sits below (MSE) / above (PCC) the line.

Both panels share one x axis; MSE is on top because the error axis is what the models are
trained on, and PCC below because it is the metric the paper ranks on.

Usage:
    python3 scripts/stat_plot/plot_extend_vs_knn_refs.py
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

from _style import INK2, apply, save, save_csv  # noqa: E402
from plot_extend_rawbase_vs_joint import ARMS, _log_ticks, load, summarize, thin_log_ticks  # noqa: E402
from ec_sa_baselines import ec_label_copy  # noqa: E402

PANELS = [("mse", "MSE"), ("pcc", "PCC")]
PAGE_CM = (10.0, 16.0)       # the page this figure is laid out for
# A panel's width:height (3.418 x 2.391 in), and the gap between the two as a fraction of
# panel height. Fixing the shape rather than letting the page dictate it keeps the curves
# comparable across page sizes. PADS are the furniture: the y label
# and its ticks on the left, and below, the two-line x ticks (~0.45 in), the x label
# (~0.32) and the three-row key (~0.65), which take whatever height is left.
PANEL_ASPECT = 3.418 / 2.391
PANEL_GAP = 0.10
PADS = {"left": 0.70, "right": 0.14, "top": 0.10}
# A bare name: what it copies, from which pool, and that it is rescaled all live in the
# module docstring for the caption to pick up.
REF = {"color": INK2, "ls": ":", "label": "EC-1NN"}


def draw(ax, by_kind, ext, metric, ec_ref):
    """One metric: both trained arms (mean, +/-1 std band, per-seed dots) + the baseline."""
    rng = np.random.default_rng(0)                   # fixed jitter, identical on re-runs
    for kind, style in ARMS.items():
        sub = ext[ext["kind"] == kind].sort_values("n_train")
        x, m, s = sub["n_train"].values, sub[f"{metric}_mean"].values, sub[f"{metric}_std"].values
        ax.fill_between(x, m - s, m + s, color=style["color"], alpha=0.15, lw=0, zorder=2)
        for per_seed in by_kind[kind].values():
            ntr = np.mean([v["n_train"] for v in per_seed.values()])
            ys = [v[metric] for v in per_seed.values()]
            ax.scatter(ntr * np.exp(rng.uniform(-0.03, 0.03, len(ys))), ys,
                       color=style["color"], alpha=0.35, s=6, lw=0, zorder=3)
        ax.plot(x, m, style["ls"], color=style["color"], marker=style["marker"],
                mec="white", mew=0.7, zorder=4, label=style["label"])

    ax.axhline(ec_ref, color=REF["color"], ls=REF["ls"], lw=1.2, zorder=1,
               label=REF["label"])
    ax.set_xscale("log")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "04-extend"))
    ap.add_argument("--esm", default="esm2_t30_150M",
                    help="keep only runs whose esm_name contains this")
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figure"))
    args = ap.parse_args()

    by_kind = load(args.results, args.esm)
    missing = [k for k in ARMS if not by_kind[k]]
    if missing:
        raise SystemExit(f"no 04-extend runs for arm(s) {missing} (--esm {args.esm})")
    ext = summarize(by_kind)

    print("computing the label-copy baseline (sequence alignment) ...", flush=True)
    ec_test, ec_pcc, slope, intercept = ec_label_copy()
    ec_ref = {"pcc": ec_pcc, "mse": ec_test.attrs["mse"]}
    print(f"EC 1-NN copy (flat): PCC {ec_ref['pcc']:.3f}  MSE {ec_ref['mse']:.3f} "
          f"[map SA = {slope:.3f}*EC {intercept:+.3f}]")

    apply(font_scale=1.5)
    w, h = (c / 2.54 for c in PAGE_CM)
    panel_w = w - PADS["left"] - PADS["right"]
    plot_h = (2 + PANEL_GAP) * panel_w / PANEL_ASPECT
    fig, axes = plt.subplots(2, 1, figsize=(w, h), sharex=True)
    for ax, (metric, label) in zip(axes, PANELS):
        draw(ax, by_kind, ext, metric, ec_ref[metric])
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    _log_ticks(axes[-1], thin_log_ticks(sorted(ext[ext["kind"] == "joint"]["n_train"].unique()),
                                        min_ratio=2.0))
    axes[-1].set_xlabel(r"Number of $\it{S.}$ $\it{aureus}$ training peptides")
    # the panels are sized in inches, so place them by fraction rather than letting
    # tight_layout negotiate -- it would trade panel area against the labels
    fig.subplots_adjust(left=PADS["left"] / w, right=1 - PADS["right"] / w,
                        top=1 - PADS["top"] / h, bottom=1 - (PADS["top"] + plot_h) / h,
                        hspace=PANEL_GAP)
    # stacked, as in the composite: three entries side by side are far wider than a panel
    # this narrow, and they do not divide into two columns without leaving a hole
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=1, loc="lower center",
               bbox_to_anchor=((PADS["left"] + panel_w / 2) / w, 0.015),
               handlelength=1.9, labelspacing=0.45)

    os.makedirs(args.figdir, exist_ok=True)
    save(fig, os.path.join(args.figdir, "extend_vs_knn_refs"),
         bbox_inches=fig.bbox_inches, pad_inches=0)
    out = pd.concat([ext.assign(source="model"),
                     pd.DataFrame([{"kind": "ec_1nn", "source": "baseline",
                                    "pcc_mean": ec_ref["pcc"], "mse_mean": ec_ref["mse"]}])],
                    ignore_index=True)
    save_csv(out, args.figdir, "extend_vs_knn_refs.csv")


if __name__ == "__main__":
    main()
