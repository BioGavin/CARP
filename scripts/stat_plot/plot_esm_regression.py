#!/usr/bin/env python3
"""Aggregate ESM-2 size x (frozen/finetune) E. coli pMIC regression results and plot.

Reads every result JSON in the results dir (written by scripts/train_reg.py), groups
by (model size, mode), computes mean and sample std (ddof=1) across seeds for each
metric, writes a summary CSV to <figdir>/data/, and renders grouped-bar figures (mean +/- std error bars) as PDFs:
<fig-name>_1x3 (MSE, PCC, KTC in a row), <fig-name>_3x1 (the same stacked, as
horizontal bars) and <fig-name>_5metric.

Usage:
    python3 scripts/stat_plot/plot_esm_regression.py
    python3 scripts/stat_plot/plot_esm_regression.py --results <dir> --figdir <dir>
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import C, apply, fit_size, group_layout, save, save_csv

# esm_name substring -> (short label, #params in M) ; also fixes plotting order
SIZE_MAP = [
    ("esm2_t6_8M", "8M", 8),
    ("esm2_t12_35M", "35M", 35),
    ("esm2_t30_150M", "150M", 150),
    ("esm2_t33_650M", "650M", 650),
]
MODES = ["frozen", "finetune"]
MODE_LABELS = {"frozen": "frozen", "finetune": "fine-tune"}
# metric key -> pretty label
METRICS = {
    "mse": "MSE",
    "pcc": "PCC",
    "ktc": "KTC",
    "spearman": "Spearman",
    "r2": "R²",
}
MODE_COLORS = {"frozen": C[0], "finetune": C[1]}
# benchmark-aligned 3-metric main figure, and the full 5-metric figure
PLOT_METRICS_3 = ["mse", "pcc", "ktc"]
PLOT_METRICS_5 = ["mse", "r2", "pcc", "ktc", "spearman"]


def size_of(esm_name):
    base = esm_name.split("/")[-1]
    for sub, label, params in SIZE_MAP:
        if base.startswith(sub):
            return label, params
    return base, -1


def load(results_dir):
    """-> dict[(label, params, mode)][seed] = {metric: value}. Deduped by newest mtime."""
    by_key = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json")), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a, r = d.get("args", {}), d.get("result", {}).get("test_at_best_val", {})
        if "esm_name" not in a or "mode" not in a or not r:
            continue
        label, params = size_of(a["esm_name"])
        key = (label, params, a["mode"])
        # newest file per (size, mode, seed) wins (handles legacy-name overlaps)
        by_key.setdefault(key, {})[a["seed"]] = {m: r[m] for m in METRICS if m in r}
    return by_key


def summarize(by_key):
    rows = []
    for sub, label, params in SIZE_MAP:
        for mode in MODES:
            key = (label, params, mode)
            if key not in by_key:
                continue
            per_seed = by_key[key]
            row = {"model": label, "params_M": params, "mode": mode,
                   "n_seeds": len(per_seed), "seeds": ",".join(map(str, sorted(per_seed)))}
            for m in METRICS:
                vals = [s[m] for s in per_seed.values() if m in s]
                if vals:
                    row[f"{m}_mean"] = float(np.mean(vals))
                    row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# (nrows, ncols) -> figure height in inches (the width follows from the bar layout)
LAYOUT_HEIGHT = {(1, 3): 2.7, (1, 5): 2.5}
HBAR_WIDTH = 3.6  # figure width (in) of the stacked horizontal-bar layout


def _means_stds(df, m, mode, labels):
    means, stds = [], []
    for lab in labels:
        sub = df[(df["model"] == lab) & (df["mode"] == mode)]
        means.append(sub[f"{m}_mean"].values[0] if len(sub) else np.nan)
        stds.append(sub[f"{m}_std"].values[0] if len(sub) else 0.0)
    return np.array(means), np.array(stds)


def draw_hbar(axes, df, metrics):
    """Horizontal grouped bars into stacked ``axes`` (one per metric): model sizes top to
    bottom, frozen / fine-tune per size, mean +/- std, value at the bar end. The value
    axis starts at 0; bars use the shared physical width. -> legend (handles, labels)."""
    labels = [s[1] for s in SIZE_MAP if s[1] in set(df["model"])]
    pos, centers, span, bar_w = group_layout(len(labels), len(MODES))
    for ax, m in zip(axes, metrics):
        right = 0.0
        for i, mode in enumerate(MODES):
            means, stds = _means_stds(df, m, mode, labels)
            ax.barh(pos[i], means, bar_w, xerr=stds, color=MODE_COLORS[mode], linewidth=0,
                    label=MODE_LABELS[mode], capsize=2,
                    error_kw=dict(lw=0.8, ecolor="#333", capthick=0.8), zorder=2)
            for y, mu, sd in zip(pos[i], means, stds):
                if not np.isnan(mu):
                    ax.annotate(f"{mu:.3f}", (mu + sd, y), xytext=(2, 0),
                                textcoords="offset points", ha="left", va="center",
                                fontsize=plt.rcParams["xtick.labelsize"] * 0.75, zorder=4)
            right = max(right, np.nanmax(means + stds))
        ax.set_xlim(0, right * 1.25)  # room for the value labels
        ax.set_xlabel(METRICS[m])
        ax.set_yticks(centers, labels)
        ax.set_ylim(span[1], span[0])  # inverted: 8M on top, frozen above fine-tune
        ax.grid(axis="y", visible=False)
    axes[len(axes) // 2].set_ylabel("ESM-2 model size")
    return axes[0].get_legend_handles_labels()


def plot_hbar(df, figpath, metrics):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(HBAR_WIDTH, 7))
    handles, leg_labels = draw_hbar(axes, df, metrics)
    fit_size(fig, axes[0], lambda: fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.96), h_pad=0.8),
             axis="y")
    fig.legend(handles, leg_labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    save(fig, figpath)


def draw_vbar(axes, df, metrics, group_frac=None, value_scale=0.75):
    """Vertical grouped bars into ``axes`` (one per metric, side by side): mean across
    seeds per (backbone size, mode), +/- std error bars, and the mean printed above each
    error bar. The y-axis starts at 0 so bar heights stay proportional to the values.
    ``group_frac``: see ``_style.group_layout``; ``value_scale``: value-label font size
    relative to the tick labels. -> legend (handles, labels)."""
    labels = [s[1] for s in SIZE_MAP if s[1] in set(df["model"])]
    pos, centers, xlim, bar_w = group_layout(len(labels), len(MODES), group_frac)
    for ax, m in zip(axes, metrics):
        top = 0.0
        for i, mode in enumerate(MODES):
            xs = pos[i]
            color = MODE_COLORS[mode]
            means, stds = _means_stds(df, m, mode, labels)
            ax.bar(xs, means, bar_w, yerr=stds, color=color, linewidth=0,
                   label=MODE_LABELS[mode], capsize=2,
                   error_kw=dict(lw=0.8, ecolor="#333", capthick=0.8), zorder=2)
            # value labels sit just above the error bar, rotated to fit the narrow bars
            for xi, mu, sd in zip(xs, means, stds):
                if np.isnan(mu):
                    continue
                ax.annotate(f"{mu:.3f}", (xi, mu + sd), xytext=(0, 1.5),
                            textcoords="offset points", ha="center", va="bottom",
                            rotation=90,
                            fontsize=plt.rcParams["xtick.labelsize"] * value_scale,
                            zorder=4)
            top = max(top, np.nanmax(np.add(means, stds)))
        ax.set_ylim(0, top * 1.25)  # headroom for the labels
        ax.set_title(METRICS[m])
        ax.set_xticks(centers)
        ax.set_xticklabels(labels)  # "8M", "35M", ...
        ax.set_xlim(xlim)
        ax.grid(axis="x", visible=False)
    axes[len(axes) // 2].set_xlabel("ESM-2 model size")  # one label under the middle panel
    return axes[0].get_legend_handles_labels()


def plot(df, figpath, metrics, layout):
    nrows, ncols = layout
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, LAYOUT_HEIGHT[layout]))
    axes = np.atleast_1d(axes).ravel()
    handles, leg_labels = draw_vbar(axes, df, metrics)
    fit_size(fig, axes[0], lambda: fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.9), w_pad=1.2))
    fig.legend(handles, leg_labels, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.99))
    save(fig, figpath)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(root, "results", "01-esm-regressor"))
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--csv-name", default="esm_size_mode_summary.csv")
    ap.add_argument("--fig-name", default="esm_size_mode_regression")
    args = ap.parse_args()

    by_key = load(args.results)
    df = summarize(by_key)
    if df.empty:
        raise SystemExit(f"no result JSONs parsed in {args.results}")

    save_csv(df, args.figdir, args.csv_name, float_format="%.4f")
    print(df.to_string(index=False))

    apply(font_scale=1.2)
    base = os.path.join(args.figdir, args.fig_name)
    plot(df, base + "_1x3", PLOT_METRICS_3, (1, 3))
    plot_hbar(df, base + "_3x1", PLOT_METRICS_3)
    plot(df, base + "_5metric", PLOT_METRICS_5, (1, 5))


if __name__ == "__main__":
    main()
