"""Grouped BAR chart of the strategy x train-ratio sweep.

Run from the CARP/ root:
  python3 scripts/stat_plot/bar_plot_cls_ft_esm.py --seed 0 --test-r 1 --prefix sweep_ft_esm_cls

Bars are mean over the 4 test strategies; error bars are the std across them.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from _style import BAR_W, C, apply, bar_x, fit_size, panel_label, save


_HERE = os.path.dirname(__file__)  # CARP/scripts/stat_plot
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # CARP
RESULTS_DIR = os.path.join(_ROOT, "results", "02-esm-classifier")
FIG_DIR = os.path.join(_ROOT, "figure")

STRAT_ORDER = ["random", "length_matched", "composition_matched", "dual"]
PRETTY = {
    "random": "Random",
    "length_matched": "Length-matched",
    "composition_matched": "Composition-matched",
    "dual": "Dual",
}
COLORS = dict(zip(STRAT_ORDER, C))


# ---- 1-2. read runs -> long table -------------------------------------------
def load_long(seed, test_r, prefix):
    pat = os.path.join(RESULTS_DIR, f"{prefix}*test{test_r:g}_seed{seed}.json")
    rows = []
    for path in sorted(glob.glob(pat)):
        with open(path) as f:
            d = json.load(f)
        ebs = d.get("eval_by_strategy")
        if not ebs:
            print(f"  ! no eval_by_strategy in {os.path.basename(path)} — skipped")
            continue
        a = d["args"]
        for test_strat, m in ebs.items():
            if test_strat not in STRAT_ORDER:  # skip a stray summary key
                continue
            rows.append(
                {
                    "train_strat": a["strategy"],
                    "train_ratio": float(a["neg_pos_ratio"]),
                    "test_strat": test_strat,
                    "auc": m["auc"],
                    "aupr": m["aupr"],
                }
            )
    if not rows:
        raise SystemExit(f"no runs matched {pat}")
    return pd.DataFrame(rows)


# ---- 3. summarise across the 4 test strategies ------------------------------
def summarise(df, metric):
    """mean/std of `metric` over the 4 test strategies, per (train_strat, ratio)."""
    g = df.groupby(["train_strat", "train_ratio"])[metric]
    return g.agg(["mean", "std"]).reset_index()


# ---- 4. grouped bar chart ---------------------------------------------------
def plot_bars(df, out_base, ymin, ymax):
    ratios = sorted(df["train_ratio"].unique())
    pos, centers, xlim = bar_x(len(ratios), len(STRAT_ORDER))
    # stacked: 4 strategies x 5 ratios per panel is too wide for two panels side by side
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
    for ax, tag, (metric, label) in zip(axes, ["(a)", "(b)"], [("auc", "AUC"), ("aupr", "AUPR")]):
        summ = summarise(df, metric)
        for k, s in enumerate(STRAT_ORDER):
            sub = (
                summ[summ["train_strat"] == s].set_index("train_ratio").reindex(ratios)
            )
            means = sub["mean"].to_numpy()
            stds = np.nan_to_num(sub["std"].to_numpy())
            ax.bar(
                pos[k],
                means,
                BAR_W,
                yerr=stds,
                capsize=1.5,
                color=COLORS[s],
                label=PRETTY[s],
                linewidth=0,
                error_kw=dict(lw=0.6, ecolor="#333", capthick=0.6),
            )
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(centers)
        ax.set_xticklabels([f"1:{r:g}" for r in ratios])
        ax.set_xlim(xlim)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        panel_label(ax, tag)
    axes[-1].set_xlabel("Train pos:neg ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(STRAT_ORDER), loc="lower center",
               bbox_to_anchor=(0.5, 0.97), title="Negative sampling (train)",
               title_fontsize=7.5)
    fit_size(fig, axes[0], lambda: fig.tight_layout(h_pad=1.0))
    save(fig, out_base)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="seed")
    p.add_argument("--test-r", type=float, default=1, help="fixed test neg:pos ratio")
    p.add_argument("--prefix", default="sweep_ft_esm_cls", help="result-file prefix")
    p.add_argument("--ymin", type=float, default=0.90, help="y-axis lower bound for plots")
    p.add_argument("--ymax", type=float, default=1.00, help="y-axis upper bound for plots")
    p.add_argument("--figdir", default=FIG_DIR, help="output dir for the PDF")
    args = p.parse_args()

    df = load_long(args.seed, args.test_r, args.prefix)
    print(
        f"loaded {len(df)} rows "
        f"({df['train_strat'].nunique()} train strat x "
        f"{df['train_ratio'].nunique()} ratio x {df['test_strat'].nunique()} test strat)"
    )
    apply()
    out_base = os.path.join(args.figdir, f"{args.prefix}_bar_seed{args.seed}")
    plot_bars(df, out_base, args.ymin, args.ymax)


if __name__ == "__main__":
    main()
