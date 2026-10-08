#!/usr/bin/env python3
"""BERT-AmPEP60 vs CARP regression, one panel per organism.

Both models are scored on the same fixed test set, with five seeds that re-split the
training pool 90/10 and the epoch picked by validation MSE.

  BERT-AmPEP60  -- results/05-repro-bertampep60 (ProtBERT, its published recipe),
                   ``result.test_at_best_val``
  CARP, E. coli -- results/03-joint-classreg, lambda = 0.25, 150M backbone, test metrics at
                   the best *regression* checkpoint
  CARP, S. aureus -- results/04-extend, the same lambda = 0.25 backbone frozen with a new
                   S. aureus head on the full training pool

The S. aureus CARP model was fine-tuned on E. coli labels (84% of the S. aureus test peptides
are in the E. coli training pool), so that panel compares the two systems, not the backbones.

Usage:
    python3 scripts/stat_plot/plot_bertampep60_vs_carp.py
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from _style import ANNOT_FS, C, INK, INK2, apply, group_layout, save, save_csv
import matplotlib.pyplot as plt  # noqa: E402  (after _style picks the Agg backend)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ESM = "esm2_t30_150M"
LAM = 0.25
# metric key -> tick label
METRICS = [("mse", "MSE"), ("r2", "R²"), ("pcc", "PCC"), ("ktc", "KTC"),
           ("spearman", "Spearman")]
MODELS = [("bert", "BERT-AmPEP60", C[3]), ("carp", "CARP", C[0])]
ORGS = [("E. coli", r"$\it{E.}$ $\it{coli}$"), ("S. aureus", r"$\it{S.}$ $\it{aureus}$")]
PAGE_CM = (28.0, 8.6)
# Margins in inches: the y label and ticks on the left, the panel headlines above, one tick
# row and a one-row key below.
PADS = {"left": 0.72, "right": 0.10, "top": 0.36, "bottom": 0.78, "gap": 0.30}
GROUP_FRAC = 0.78
LABEL_FS = ANNOT_FS * 0.92


def _newest(pattern, keep, get):
    """-> dict[seed] = metrics over the files that pass ``keep``; the newest file per seed wins."""
    out = {}
    for f in sorted(glob.glob(pattern), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = d.get("args", {})
        r = get(d)
        if r and "seed" in a and keep(a, d):
            out[a["seed"]] = r
    return out


def load(results_root):
    """-> dict[(organism, model)][seed] = metrics."""
    def lam_ok(a):
        return np.isclose(a.get("mse_weight", -1), LAM)

    ec_joint = _newest(
        os.path.join(results_root, "03-joint-classreg", "*.json"),
        lambda a, d: ESM in a.get("esm_name", "") and lam_ok(a),
        lambda d: d.get("result", {}).get("reg", {}).get("all_test_at_best_val"))
    sa_joint = _newest(
        os.path.join(results_root, "04-extend", "*.json"),
        lambda a, d: (ESM in a.get("esm_name", "") and a.get("n_sub") == 0
                      and d.get("result", {}).get("base_load", {}).get("kind") == "joint"
                      and f"lam{LAM}" in os.path.basename(a.get("base_ckpt") or "")),
        lambda d: d.get("result", {}).get("new_organism", {}).get("test"))
    bert = {org: _newest(
        os.path.join(results_root, "05-repro-bertampep60", f"protbert_{code}_seed*.json"),
        lambda a, d: True, lambda d: d.get("result", {}).get("test_at_best_val"))
        for org, code in (("E. coli", "EC"), ("S. aureus", "SA"))}
    return {("E. coli", "bert"): bert["E. coli"], ("E. coli", "carp"): ec_joint,
            ("S. aureus", "bert"): bert["S. aureus"], ("S. aureus", "carp"): sa_joint}


def summarize(by_key):
    rows = []
    for org, _ in ORGS:
        for model, mlabel, _ in MODELS:
            per_seed = by_key[(org, model)]
            if not per_seed:
                raise SystemExit(f"no runs for {org} / {mlabel}")
            for m, _ in METRICS:
                v = np.array([s[m] for s in per_seed.values()], float)
                rows.append({"organism": org, "model": mlabel, "metric": m, "n_seeds": len(v),
                             "mean": v.mean(), "std": v.std(ddof=1)})
    return pd.DataFrame(rows)


def panel(ax, df, org):
    pos, centers, span, w = group_layout(len(METRICS), len(MODELS), group_frac=GROUP_FRAC)
    for i, (model, label, color) in enumerate(MODELS):
        sub = df[(df.organism == org) & (df.model == label)].set_index("metric").loc[
            [m for m, _ in METRICS]]
        means, stds = sub["mean"].to_numpy(), sub["std"].to_numpy()
        ax.bar(pos[i], means, width=w, color=color, zorder=2, label=label)
        ax.errorbar(pos[i], means, yerr=stds, fmt="none", ecolor=INK2, elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=3)
        for x, m, s in zip(pos[i], means, stds):
            ax.annotate(f"{m:.3f}", xy=(x, m + s), xytext=(0, 3.5), textcoords="offset points",
                        ha="center", va="bottom", color=INK, zorder=4,
                        fontsize=plt.rcParams["xtick.labelsize"] * LABEL_FS)
    ax.set_xticks(centers, [t for _, t in METRICS])
    ax.set_xlim(*span)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", length=0)  # the gridlines already locate the values
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figure"))
    ap.add_argument("--fig-name", default="bertampep60_vs_carp")
    args = ap.parse_args()

    df = summarize(load(args.results))
    print(df.to_string(index=False, float_format="%.4f"))

    apply(font_scale=1.5)
    w, h = (c / 2.54 for c in PAGE_CM)
    pw = (w - PADS["left"] - PADS["right"] - PADS["gap"]) / 2
    bottom, height = PADS["bottom"] / h, 1 - (PADS["top"] + PADS["bottom"]) / h
    fig = plt.figure(figsize=(w, h))
    axes = []
    for i in range(2):
        left = PADS["left"] + i * (pw + PADS["gap"])
        axes.append(fig.add_axes((left / w, bottom, pw / w, height),
                                 sharey=axes[0] if axes else None))
    axes[1].tick_params(labelleft=False)
    for ax, (org, title) in zip(axes, ORGS):
        panel(ax, df, org)
        ax.set_title(title, pad=6)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), columnspacing=2.2, handlelength=1.1)

    save_csv(df, args.figdir, f"{args.fig_name}.csv", float_format="%.4f")
    # the shared rcParams crop to a tight bbox; pass the figure's own so the page stays
    # exactly PAGE_CM
    save(fig, os.path.join(args.figdir, args.fig_name),
         bbox_inches=fig.bbox_inches, pad_inches=0)


if __name__ == "__main__":
    main()
