#!/usr/bin/env python3
"""Compare S. aureus MIC-head transfer from a raw vs a joint-fine-tuned ESM-2 backbone.

Reads every per-run JSON in ``results/04-extend/`` and splits the runs by
``result.base_load.kind``:

  - "joint": frozen backbone taken from the step-3 joint (lam=0.25) E. coli
    checkpoint, then a fresh S. aureus head (experiment-4 main arm).
  - "raw":   frozen *native* pretrained ESM-2 backbone, then a fresh S. aureus head
    (the control arm; ``forgetting`` is null, no base organism). Displayed as
    "Native" -- the paper's term for the untuned pretrained backbone (cf. the
    classification "native vs. fine-tuned" comparison).

Both arms add the SA head under the **identical** frozen ``fit_frozen`` recipe and
the same ``n_sub`` x seed grid, so the only variable is the origin of the frozen
backbone weights. For each arm this groups by ``n_sub`` (0 == full SA train pool),
computes mean and sample std (ddof=1) over the 5 seeds of the SA *test* metrics,
writes a combined summary CSV (to figure/data/), and renders:

  figure/extend_sa_rawbase_vs_joint.pdf -- two panels vs the number of S. aureus
  training peptides the new head saw (log x-axis), with +/-1 std bands and per-seed
  scatter:
    left:  SA test MSE (lower is better) -- joint vs native
    right: SA test PCC (higher is better) -- joint vs native

  The transfer benefit of joint fine-tuning is the vertical PCC gap between the two
  arms in the PCC panel: largest in the low-data regime and shrinking as SA data grows (the raw
  numbers are in the CSV). MSE (not RMSE) is the error axis to match the
  paper's regression reporting. PCC anchors the right panel because it is the primary
  regression metric the paper ranks models on; Spearman and R^2 tell the same monotone
  story (see the CSV) and are omitted to keep the two-arm comparison legible.

Usage:
    python3 scripts/stat_plot/plot_extend_rawbase_vs_joint.py
    python3 scripts/stat_plot/plot_extend_rawbase_vs_joint.py --results <dir> --figdir <dir>
    python3 scripts/stat_plot/plot_extend_rawbase_vs_joint.py --esm esm2_t33_650M   # other backbone

Only runs whose ``args.esm_name`` contains ``--esm`` are used (default: the primary
150M backbone), so extension runs on other backbone sizes never mix into one arm.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import C, apply, save, save_csv

# The two backbone arms, keyed by result.base_load.kind. One color per arm, held
# consistent across every panel so the eye tracks an arm, not a metric.
ARMS = {
    "joint": {"label": r"Joint $\it{E.}$ $\it{coli}$ backbone", "color": C[0],
              "marker": "o", "ls": "-"},
    "raw":   {"label": "Native ESM-2", "color": C[1],
              "marker": "s", "ls": "--"},
}
METRICS = ["mse", "pcc", "spearman", "r2"]


def load(results_dir, esm):
    """-> dict[kind][n_sub][seed] = {n_train, rmse, pcc, spearman, r2}.

    Only kinds in ARMS and runs whose esm_name contains ``esm`` are kept. Newest file per (kind, n_sub, seed) wins (mtime),
    so re-runs override cleanly."""
    by_key = {k: defaultdict(dict) for k in ARMS}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json")), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = d.get("args", {})
        r = d.get("result", {})
        kind = r.get("base_load", {}).get("kind")
        test = r.get("new_organism", {}).get("test", {})
        if kind not in ARMS or "n_sub" not in a or not test:
            continue
        if esm not in a.get("esm_name", ""):
            continue
        rec = {"n_train": r["new_organism"].get("n_train")}
        for m in METRICS:
            rec[m] = test.get(m)
        by_key[kind][a["n_sub"]][a["seed"]] = rec
    return by_key


def summarize(by_key):
    """One row per (kind, n_sub), mean/std over seeds. Sorted by n_train ascending."""
    rows = []
    for kind, per_nsub in by_key.items():
        for nsub, per_seed in per_nsub.items():
            recs = list(per_seed.values())
            row = {"kind": kind, "n_sub": nsub,
                   "label": "full" if nsub == 0 else str(nsub),
                   "n_train": float(np.mean([x["n_train"] for x in recs])),
                   "n_seeds": len(recs)}
            for m in METRICS:
                vals = [x[m] for x in recs if x[m] is not None]
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["kind", "n_train"]).reset_index(drop=True)
    return df


def _plot_metric(ax, by_key, df, metric, ylabel):
    """One metric, both arms: mean line + std band + per-seed scatter."""
    rng = np.random.default_rng(0)  # fixed jitter so reruns are identical
    for kind, style in ARMS.items():
        sub = df[df["kind"] == kind].sort_values("n_train")
        x = sub["n_train"].values
        m, s = sub[f"{metric}_mean"].values, sub[f"{metric}_std"].values
        ax.fill_between(x, m - s, m + s, color=style["color"], alpha=0.15, lw=0, zorder=1)
        for nsub, per_seed in by_key[kind].items():
            ntr = np.mean([v["n_train"] for v in per_seed.values()])
            ys = [v[metric] for v in per_seed.values()]
            jit = np.exp(rng.uniform(-0.03, 0.03, len(ys)))  # multiplicative: log x-axis
            ax.scatter(ntr * jit, ys, color=style["color"], alpha=0.35, s=6, lw=0, zorder=2)
        ax.plot(x, m, style["ls"], color=style["color"], marker=style["marker"],
                mec="white", mew=0.7, zorder=3, label=style["label"])
    ax.set_xscale("log")
    ax.set_ylabel(ylabel)


def thin_log_ticks(ticks, min_ratio=2.0):
    """Keep ticks at least ``min_ratio`` apart on a log axis; always keep the last."""
    shown = [ticks[-1]]
    for t in reversed(ticks[:-1]):
        if shown[-1] / t >= min_ratio:
            shown.append(t)
    return shown[::-1]


def _log_ticks(ax, ticks):
    ax.set_xticks(ticks)
    # the largest size is the full S. aureus training pool: say so on its tick
    ax.set_xticklabels([f"{int(round(t))}" for t in ticks[:-1]]
                       + [f"{int(round(ticks[-1]))}\n(full)"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis="x", visible=False)


def plot_compare(by_key, df, figpath):
    """MSE (left) | PCC (right), joint vs native; the vertical PCC gap is the
    transfer benefit of joint fine-tuning."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))

    _plot_metric(axes[0], by_key, df, "mse", "MSE")
    _plot_metric(axes[1], by_key, df, "pcc", "PCC")

    # shared x ticks at the actual n_train values; drop labels that would collide
    ticks = sorted(df[df["kind"] == "joint"]["n_train"].unique())
    for a in axes:
        _log_ticks(a, thin_log_ticks(ticks))
        # close the frame: top/right spines as plain lines, no extra ticks
        a.spines["top"].set_visible(True)
        a.spines["right"].set_visible(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(w_pad=2.0, rect=(0, 0.07, 1, 1))  # bottom strip for the shared title
    # one x-axis title shared by both panels, just under the tick labels
    fig.text(0.5, 0.0, r"Number of $\it{S.}$ $\it{aureus}$ training peptides",
             ha="center", va="bottom", fontsize=plt.rcParams["axes.labelsize"])
    save(fig, figpath)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(root, "results", "04-extend"))
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--esm", default="esm2_t30_150M",
                    help="keep only runs whose esm_name contains this")
    args = ap.parse_args()

    by_key = load(args.results, args.esm)
    missing = [k for k in ARMS if not by_key[k]]
    if missing:
        raise SystemExit(f"no runs found for arm(s) {missing} in {args.results} "
                         f"(need both native and joint results for --esm {args.esm})")
    df = summarize(by_key)
    os.makedirs(args.figdir, exist_ok=True)
    print(df.to_string(index=False))
    save_csv(df, args.figdir, "extend_sa_rawbase_vs_joint.csv")

    apply(font_scale=1.5)
    plot_compare(by_key, df, os.path.join(args.figdir, "extend_sa_rawbase_vs_joint"))


if __name__ == "__main__":
    main()
