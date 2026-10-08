#!/usr/bin/env python3
"""Does the 650M backbone beat the 150M one? Paired differences across the pipeline.

Compares the two backbone sizes at three stages, always on E. coli-trained models with
the same 5 seeds (a seed is the same train/val split in both arms, so the comparison is
paired and the per-seed difference cancels the split-to-split noise):

  01 single-task fine-tune  -- results/01-esm-regressor (``mode=finetune``),
                               ``result.test_at_best_val``
  03 joint (lambda=0.25)    -- results/03-joint-classreg, test metrics at the best
                               *regression* checkpoint
  04 extend -> S. aureus    -- results/04-extend, full SA train pool, joint base,
                               ``result.new_organism.test``

Every difference is sign-normalised so **positive means 650M is better**, whatever the
metric's own direction. The figure stacks one panel per
stage: a point-range per metric (mean +/- std of the paired difference) with a zero line. The 650M lead at stage 01
is gone by stage 03 and reversed at stage 04.

Paired t-tests over the 5 seeds go to the CSV; with n=5 they have little power, so read
the figure as a trend rather than per-metric significance.

Usage:
    python3 scripts/stat_plot/plot_650m_vs_150m.py
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _style import C, MUTED, apply, save, save_csv

TITLE_PAD, TITLE_GAP = 0.03, 0.05  # inches: left margin of the stage titles, and their gap to the frame
PAGE_CM = (14.5, 9.0)  # the page this figure is laid out for (width, height in cm)
SMALL, LARGE = "esm2_t30_150M", "esm2_t33_650M"
# metric key -> (pretty label, lower_is_better)
METRICS = {"mse": ("MSE", True), "pcc": ("PCC", False), "ktc": ("KTC", False),
           "r2": ("R²", False), "spearman": ("Spearman", False),
           "auc": ("AUROC", False), "aupr": ("AUPR", False), "mcc": ("MCC", False),
           "f1": ("F1", False), "acc": ("Acc", False)}
# (stage key, plain name for the CSV, panel label, colour, metrics shown) in reading order
STAGES = [
    ("single", "01 single-task fine-tune", "CARP-S1", C[1], ["mse", "pcc"]),
    ("joint", "03 joint (lambda=0.25)", "CARP-S2\n($\\lambda$ = 0.25)", C[0],
     ["mse", "pcc", "auc", "mcc"]),
    ("extend", "04 extend to S. aureus", "CARP-S2\ntransfer", C[2],
     ["mse", "pcc"]),
]


def _collect(pattern, keep, get):
    """-> dict[esm substring][seed] = metrics. Newest mtime per (size, seed) wins."""
    out = {SMALL: {}, LARGE: {}}
    for f in sorted(glob.glob(pattern), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = d.get("args", {})
        esm = a.get("esm_name", "")
        size = next((s for s in (SMALL, LARGE) if s in esm), None)
        if size is None or not keep(a, d):
            continue
        r = get(d)
        if r:
            out[size][a["seed"]] = r
    return out


def load(results_root, lam):
    return {
        "single": _collect(
            os.path.join(results_root, "01-esm-regressor", "*.json"),
            lambda a, d: a.get("mode") == "finetune",
            lambda d: d.get("result", {}).get("test_at_best_val")),
        "joint": _collect(
            os.path.join(results_root, "03-joint-classreg", "*.json"),
            lambda a, d: np.isclose(a.get("mse_weight", -1), lam),
            lambda d: d.get("result", {}).get("reg", {}).get("all_test_at_best_val")),
        "extend": _collect(
            os.path.join(results_root, "04-extend", "*.json"),
            lambda a, d: (a.get("n_sub") == 0
                          and d.get("result", {}).get("base_load", {}).get("kind") == "joint"),
            lambda d: d.get("result", {}).get("new_organism", {}).get("test")),
    }


def summarize(by_stage):
    """One row per (stage, metric): paired difference, sign-normalised so + = 650M better."""
    rows = []
    for key, plain, _, _, metrics in STAGES:
        small, large = by_stage[key][SMALL], by_stage[key][LARGE]
        seeds = sorted(set(small) & set(large))  # paired: same seed = same split
        for m in metrics:
            if not all(m in small[s] and m in large[s] for s in seeds):
                continue
            x = np.array([small[s][m] for s in seeds], float)
            y = np.array([large[s][m] for s in seeds], float)
            d = (x - y) if METRICS[m][1] else (y - x)  # + = 650M better either way
            t, p = stats.ttest_rel(y, x)
            rows.append({"stage": plain, "metric": METRICS[m][0],
                         "n_seeds": len(seeds), "mean_150M": x.mean(), "mean_650M": y.mean(),
                         "delta_mean": d.mean(), "delta_std": d.std(ddof=1),
                         "t": t, "p": p})
    return pd.DataFrame(rows)


def plot(df, figpath):
    """One stacked panel per stage (shared x): point-range of the paired difference per
    metric, with a zero line. Panel heights follow the metric counts, so every row is
    the same height."""
    blocks = [(lbl, color, df[df["stage"] == plain]) for _, plain, lbl, color, _ in STAGES]
    blocks = [b for b in blocks if not b[2].empty]
    fig, axes = plt.subplots(len(blocks), 1, figsize=tuple(c / 2.54 for c in PAGE_CM), sharex=True,
                             gridspec_kw={"height_ratios": [len(b[2]) for b in blocks]})
    axes = np.atleast_1d(axes)
    for ax, (lbl, color, sub) in zip(axes, blocks):
        y = np.arange(len(sub))
        ax.axvline(0, ls=(0, (3, 3)), color=MUTED, lw=0.8, zorder=1)
        ax.errorbar(sub["delta_mean"], y, xerr=sub["delta_std"], ls="none", color=color,
                    marker="o", ms=5, mec="white", mew=0.8, capsize=2.5, elinewidth=1.0,
                    zorder=3)
        ax.set_yticks(y, sub["metric"])
        ax.set_ylim(len(sub) - 0.4, -0.6)  # inverted: first metric on top
        ax.yaxis.tick_right()  # metric names on the right, stage titles on the left
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        for side in ("top", "right"):  # close the frame, no extra ticks
            ax.spines[side].set_visible(True)
    axes[-1].set_xlabel("Paired difference (650M − 150M)")
    # The stage titles are figure text, not y labels, so they hug the left frame at one x: a y label
    # sits beside its own tick labels and ends up at a different x in every panel. The strip they
    # need is measured, and the axes get the rest.
    titles = [fig.text(0, 0.5, lbl, color=color, rotation=90, ha="right", va="center",
                       multialignment="center", linespacing=1.3) for lbl, color, _ in blocks]
    fig_w = fig.get_figwidth()
    strip = max(t.get_window_extent(fig.canvas.get_renderer()).width for t in titles) / fig.dpi
    fig.tight_layout(rect=((TITLE_PAD + strip + TITLE_GAP) / fig_w, 0, 1, 1), h_pad=0.6)
    for t, ax in zip(titles, axes):
        box = ax.get_position()
        t.set_position((box.x0 - TITLE_GAP / fig_w, (box.y0 + box.y1) / 2))
    # the shared rcParams crop to a tight bbox; pass the figure's own so the page stays PAGE_CM
    save(fig, figpath, bbox_inches=fig.bbox_inches, pad_inches=0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--lam", type=float, default=0.25, help="joint MSE weight")
    ap.add_argument("--fig-name", default="scale_650m_vs_150m")
    args = ap.parse_args()

    df = summarize(load(args.results, args.lam))
    if df.empty:
        raise SystemExit(f"no stage has both backbone sizes in {args.results}")
    print(df.to_string(index=False, float_format="%.4f"))

    apply(font_scale=1.5)
    save_csv(df, args.figdir, f"{args.fig_name}.csv", float_format="%.4f")
    plot(df, os.path.join(args.figdir, args.fig_name))


if __name__ == "__main__":
    main()
