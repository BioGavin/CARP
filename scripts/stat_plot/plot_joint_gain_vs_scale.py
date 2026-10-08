#!/usr/bin/env python3
"""The joint-training gain vanishes with backbone scale.

For each ESM-2 size, compares two E. coli regression models trained under otherwise
identical conditions:

  single-task fine-tune  -- results/01-esm-regressor (``mode=finetune``),
                            ``result.test_at_best_val``
  joint (lambda=0.25)    -- results/03-joint-classreg, test metrics at the best
                            *regression* checkpoint (``result.reg.all_test_at_best_val``)

Seeds are paired across the two arms and across sizes (the same seed is the same
train/val split), so the gain is computed per seed and then averaged:

    gain = single-task - joint   (MSE; positive = joint is better)

The figure plots the two arms' test MSE vs backbone size; the paired gain per size is
written to the CSV. The arms converge as the backbone grows -- at 650M the gain is ~0,
i.e. joint training buys at 150M roughly what 4x the parameters buy at single-task.

Usage:
    python3 scripts/stat_plot/plot_joint_gain_vs_scale.py
    python3 scripts/stat_plot/plot_joint_gain_vs_scale.py --metric pcc
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _style import C, apply, save, save_csv

# esm_name substring -> (short label, #params in M); also fixes plotting order
SIZE_MAP = [
    ("esm2_t6_8M", "8M", 8),
    ("esm2_t12_35M", "35M", 35),
    ("esm2_t30_150M", "150M", 150),
    ("esm2_t33_650M", "650M", 650),
]
# metric key -> (pretty label, lower_is_better)
METRICS = {"mse": ("MSE", True), "pcc": ("PCC", False), "ktc": ("KTC", False),
           "r2": ("R²", False), "spearman": ("Spearman", False)}
PAGE_CM = (13.0, 9.0)  # the page this figure is laid out for (width, height in cm)
ARMS = {"single": ("CARP-S1", C[1], "s", "--"),
        "joint": (r"CARP-S2 ($\lambda$ = 0.25)", C[0], "o", "-")}


def size_of(esm_name):
    base = str(esm_name).split("/")[-1]
    for sub, label, params in SIZE_MAP:
        if base.startswith(sub):
            return label, params
    return None, None


def load(results_root, lam):
    """-> dict[(params, arm)][seed] = metrics dict. Newest mtime per key wins."""
    out = {}
    single = os.path.join(results_root, "01-esm-regressor", "*.json")
    for f in sorted(glob.glob(single), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a, r = d.get("args", {}), d.get("result", {}).get("test_at_best_val", {})
        if a.get("mode") != "finetune" or not r or "esm_name" not in a:
            continue
        _, params = size_of(a["esm_name"])
        if params:
            out.setdefault((params, "single"), {})[a["seed"]] = r

    joint = os.path.join(results_root, "03-joint-classreg", "*.json")
    for f in sorted(glob.glob(joint), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = d.get("args", {})
        r = d.get("result", {}).get("reg", {}).get("all_test_at_best_val")
        if not r or "esm_name" not in a or not np.isclose(a.get("mse_weight", -1), lam):
            continue
        _, params = size_of(a["esm_name"])
        if params:
            out.setdefault((params, "joint"), {})[a["seed"]] = r
    return out


def summarize(by_key, metric):
    """One row per size: each arm's mean/std and the paired gain over shared seeds."""
    rows = []
    for _, label, params in SIZE_MAP:
        single, joint = by_key.get((params, "single")), by_key.get((params, "joint"))
        if not single or not joint:
            continue
        row = {"model": label, "params_M": params}
        for arm, per_seed in (("single", single), ("joint", joint)):
            v = np.array([s[metric] for s in per_seed.values() if metric in s], float)
            row[f"{arm}_mean"], row[f"{arm}_std"] = v.mean(), v.std(ddof=1)
        seeds = sorted(set(single) & set(joint))  # paired: same seed = same split
        d = np.array([single[s][metric] - joint[s][metric] for s in seeds], float)
        if not METRICS[metric][1]:  # higher-is-better: gain is joint - single
            d = -d
        row.update(n_seeds=len(seeds), gain_mean=d.mean(), gain_std=d.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot(df, metric, figpath):
    pretty = METRICS[metric][0]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=tuple(c / 2.54 for c in PAGE_CM))
    for arm, (label, color, marker, ls) in ARMS.items():
        ax.errorbar(x, df[f"{arm}_mean"], yerr=df[f"{arm}_std"], color=color, ls=ls,
                    marker=marker, ms=5, mec="white", mew=0.8, capsize=2.5,
                    elinewidth=1.0, label=label, zorder=3)
    ax.set_ylabel(pretty)
    ax.set_xticks(x, df["model"])
    ax.set_xlim(-0.35, len(df) - 0.65)
    # a little room past the error bars, so they do not touch the frame
    lo = (df[["single_mean", "joint_mean"]].values
          - df[["single_std", "joint_std"]].values).min()
    hi = (df[["single_mean", "joint_mean"]].values
          + df[["single_std", "joint_std"]].values).max()
    pad = (hi - lo) * 0.12
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):  # close the frame, no extra ticks
        ax.spines[side].set_visible(True)
    ax.set_xlabel("ESM-2 model size")
    # errorbar containers would put the whiskers in the legend keys; the keys show line + marker only
    handles = [Line2D([], [], color=color, ls=ls, marker=marker, ms=5, mec="white", mew=0.8)
               for label, color, marker, ls in ARMS.values()]
    labels = [label for label, *_ in ARMS.values()]
    ax.legend(handles, labels, loc="upper right", borderaxespad=0.5)
    fig.tight_layout()
    # the shared rcParams crop to a tight bbox; pass the figure's own so the page stays PAGE_CM
    save(fig, figpath, bbox_inches=fig.bbox_inches, pad_inches=0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--metric", default="mse", choices=list(METRICS))
    ap.add_argument("--lam", type=float, default=0.25, help="joint MSE weight")
    ap.add_argument("--fig-name", default="joint_gain_vs_scale")
    args = ap.parse_args()

    df = summarize(load(args.results, args.lam), args.metric)
    if df.empty:
        raise SystemExit(f"no size has both arms in {args.results}")
    print(df.to_string(index=False, float_format="%.4f"))

    apply(font_scale=1.5)
    name = f"{args.fig_name}_{args.metric}"
    save_csv(df, args.figdir, f"{name}.csv", float_format="%.4f")
    plot(df, args.metric, os.path.join(args.figdir, name))


if __name__ == "__main__":
    main()
