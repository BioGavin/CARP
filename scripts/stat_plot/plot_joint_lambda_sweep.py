#!/usr/bin/env python3
"""Joint reg+cls training across the loss weight lambda (paper Figure C.1).

Reads the step-3 per-run JSONs (written by scripts/train_joint.py) for the 150M
backbone and, per lambda (= --mse-weight), takes the test metrics at the best
*regression* checkpoint (``result.reg.all_test_at_best_val``, as in paper Table C.1),
mean +/- std over the 5 seeds. Renders one plot vs lambda with two y-axes:

  left  axis: classification AUROC  -- lambda=1 omitted (classification head untrained)
  right axis: regression test MSE   -- lambda=0 omitted (regression head untrained)

The shaded band marks the lambdas where both heads are trained.

Usage:
    python3 scripts/stat_plot/plot_joint_lambda_sweep.py
    python3 scripts/stat_plot/plot_joint_lambda_sweep.py --results <dir> --figdir <dir>
"""
import argparse
import glob
import json
import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import C, FULL_W, INK2, apply, save, save_csv

METRICS = ["mse", "r2", "pcc", "ktc", "spearman", "auc", "aupr", "f1", "precision",
           "recall", "mcc", "acc"]


def load(results_dir, esm):
    """-> long frame: lam, seed, <metrics>. Newest mtime per (lam, seed) wins."""
    by_key = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json")), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a, r = d.get("args", {}), d.get("result", {})
        test = r.get("reg", {}).get("all_test_at_best_val")
        if not test or "mse_weight" not in a or esm not in a.get("esm_name", ""):
            continue
        lam = float(a["mse_weight"])
        by_key[(lam, a["seed"])] = {"lam": lam, "seed": a["seed"],
                                    **{m: test.get(m) for m in METRICS}}
    return pd.DataFrame(by_key.values())


def summarize(df):
    g = df.groupby("lam")[METRICS]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    out.insert(0, "n_seeds", df.groupby("lam").size())
    # degenerate endpoints: that head never trained, its metrics are meaningless
    out.loc[out.index == 0.0, [c for c in out if c.split("_")[0] in
                               ("mse", "r2", "pcc", "ktc", "spearman")]] = np.nan
    out.loc[out.index == 1.0, [c for c in out if c.split("_")[0] in
                               ("auc", "aupr", "f1", "precision", "recall", "mcc", "acc")]] = np.nan
    return out.reset_index()


def draw(ax_l, sm):
    """Draw the dual-axis sweep into ``ax_l`` (AUROC, left); adds the twin MSE axis."""
    lams = sm["lam"].to_numpy()
    ax_r = ax_l.twinx()
    # ax_l is the bottom layer (band, label, grid); the twin ax_r is drawn above it,
    # so neither the band nor the grid can cover the MSE line or its markers
    ax_l.axvspan(0.25, 0.75, color="#f1f0ec", lw=0, zorder=0)
    ax_l.text(0.5, 0.03, "both heads trained", transform=ax_l.get_xaxis_transform(),
              ha="center", va="bottom", color=INK2, style="italic",
              fontsize=plt.rcParams["legend.fontsize"])
    handles = []
    # (below, above) padding in units of the data span: MSE is kept in the lower half
    # of the plot and AUROC in the upper half so the two lines never cross
    for ax, (m, label, color, marker, (below, above)) in [
        (ax_l, ("auc", "Classification test AUROC", C[1], "s", (1.5, 0.2))),
        (ax_r, ("mse", "Regression test MSE", C[0], "o", (0.45, 1.5))),
    ]:
        mu, sd = sm[f"{m}_mean"].to_numpy(), sm[f"{m}_std"].to_numpy()
        ok = ~np.isnan(mu)
        h = ax.errorbar(lams[ok], mu[ok], yerr=sd[ok], color=color, marker=marker, ms=5,
                        mec="white", mew=0.8, capsize=2.5, elinewidth=1.0, zorder=3,
                        label=label)
        handles.append(h)
        # value labels sit beside each point, clear of the vertical error bar: AUROC to
        # the upper right, MSE to the upper left (the last AUROC-side point would hit the
        # right spine, so any point at the max lambda also goes left)
        for lam, v in zip(lams[ok], mu[ok]):
            right = m != "mse" and lam < lams.max()
            ax.annotate(f"{v:.3f}", (lam, v), xytext=(4 if right else -4, 4),
                        textcoords="offset points", ha="left" if right else "right",
                        va="bottom", fontsize=plt.rcParams["legend.fontsize"] * 0.9,
                        zorder=4, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        lo, hi = (mu - sd)[ok].min(), (mu + sd)[ok].max()
        ax.set_ylim(lo - (hi - lo) * below, hi + (hi - lo) * above)
        # the axis label and ticks wear the series color so each axis maps to its line
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis="y", colors=color)
    # close the frame: top spine drawn as a plain line (no ticks) alongside the right axis
    ax_r.spines["right"].set_visible(True)
    ax_l.spines["top"].set_visible(True)
    ax_r.grid(False)
    ax_l.set_xticks(lams)
    ax_l.set_xticklabels([f"{v:g}" for v in lams])
    ax_l.set_xlim(-0.08, 1.08)
    ax_l.set_xlabel(r"$\lambda$ (MSE weight)")
    ax_l.grid(axis="x", visible=False)


def plot(sm, figpath):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    draw(ax, sm)
    fig.tight_layout()
    save(fig, figpath)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(root, "results", "03-joint-classreg"))
    ap.add_argument("--figdir", default=os.path.join(root, "figure"))
    ap.add_argument("--esm", default="esm2_t30_150M",
                    help="keep only runs whose esm_name contains this")
    ap.add_argument("--fig-name", default="joint_lambda_sweep")
    args = ap.parse_args()

    df = load(args.results, args.esm)
    if df.empty:
        raise SystemExit(f"no joint runs for --esm {args.esm} in {args.results}")
    sm = summarize(df)
    print(sm[["lam", "n_seeds", "mse_mean", "mse_std", "auc_mean", "auc_std"]].to_string(index=False))
    save_csv(sm, args.figdir, f"{args.fig_name}.csv", float_format="%.4f")

    apply(font_scale=1.2)
    plot(sm, os.path.join(args.figdir, args.fig_name))


if __name__ == "__main__":
    main()
