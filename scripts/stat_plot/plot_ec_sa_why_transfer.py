#!/usr/bin/env python3
"""Why does an E. coli backbone transfer to S. aureus? -- the answer in one figure.

Both panels are ridge probes on the same frozen embeddings, scored on the same S. aureus
test peptides, for two backbones (native ESM-2 vs the joint E. coli model), so one y axis
carries the whole argument. They decontaminate the transfer claim in the two independent
directions the data can be contaminated.

  left   Label space. The S. aureus target is split into the part a linear map of E. coli
         pMIC accounts for ("shared with E. coli", a*EC + b fitted on the 2368 peptides
         measured in both organisms -- the line in figure/ec_sa_overlap_combined.pdf) and
         the remainder ("S. aureus-specific"). The split is exhaustive, and the two
         backbones differ in exactly one place: E. coli fine-tuning drives the shared part
         to the ceiling and leaves the S. aureus-specific part where the native backbone
         already had it.

  right  Sequence space. The plain S. aureus probe re-scored inside each overlap bucket:
         peptides whose sequence is already in the E. coli training pool ("seen", 84% of
         the test set), near-neighbours (identity >= --near), and peptides E. coli training
         never saw. The gain is concentrated where E. coli training overlaps the test set
         (+0.13 on seen, +0.09 on near) and is absent on unseen peptides, where the point
         estimate in fact favours the native backbone (0.45 vs 0.30, and all five joint
         seeds fall below the native run). With n = 24 the bootstrap CIs there span most
         of the axis and overlap, so this panel supports "no evidence of a gain on unseen
         peptides" and NOT "the joint backbone is worse" -- separating those two needs a
         decontaminated re-split, not a re-scoring of this test set.

Together: the transfer gain is E. coli potency on molecules E. coli training already
contained, not organism-general antimicrobial chemistry.

Error bars differ by panel, because the dominant uncertainty does: +/-1 sd over the five
joint seeds on the left, where every bar is scored on all 323 peptides; a percentile
bootstrap 95% CI over peptides on the right, where the unseen bucket holds 24 of them and
seed spread would badly understate the error. The native arm is one deterministic ridge
fit, so it has no seed spread and shows an error bar only on the right.

The panels carry no (a)/(b) tags, so a caption refers to them by position.

Needs results/ec-sa-transfer/probe/ (scripts/shell/ec_sa_transfer.sh probe) and costs a
~10 s sequence-alignment pass to bucket the test peptides.

Usage:
    python3 scripts/stat_plot/plot_ec_sa_why_transfer.py
    python3 scripts/stat_plot/plot_ec_sa_why_transfer.py --near 0.9 --n-boot 5000
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    sys.path.insert(0, p)

import matplotlib.pyplot as plt  # noqa: E402  (after _style picks the Agg backend)

from _style import ANNOT_FS, INK, INK2, apply, group_layout, save, save_csv  # noqa: E402
from plot_extend_rawbase_vs_joint import ARMS  # noqa: E402
from ec_sa_baselines import boot_ci, ec_label_copy  # noqa: E402

# the two backbone arms, coloured and labelled as in plot_extend_rawbase_vs_joint
RAW, JOINT = ARMS["raw"]["color"], ARMS["joint"]["color"]
ARMS_HERE = (("raw", RAW), ("joint", JOINT))
# left panel: probe targets, in narrative order (the whole, the explained part, the rest)
TARGETS = [("sa", r"$\it{S.}$ $\it{aureus}$" + "\npMIC"),
           ("shared", "shared with\n" + r"$\it{E.}$ $\it{coli}$"),
           ("residual", r"$\it{S.}$ $\it{aureus}$-" + "\nspecific")]
# right panel: the plain S. aureus probe, bucketed by overlap with the E. coli train pool
SUBSETS = [("all", "All"), ("seen", "Seen"), ("near", "Near"), ("unseen", "Unseen")]
PAGE_CM = (20.0, 7.8)        # the page this figure is laid out for
RATIOS = (3, 4)              # targets : subsets
# Margins in inches: the y label and its ticks on the left; above, the panel headlines;
# below, the two-line x ticks and a one-row key.
PADS = {"left": 0.70, "right": 0.10, "top": 0.34, "bottom": 0.83, "gap": 0.30}
TITLES = ("Label space", "Sequence space")
# Share of a group's unit filled by its two bars, which is what sets how far apart their value
# labels sit. The labels carry three decimals to match the text, ~25% wider than two, and the
# pair runs its numbers together wherever the two bars are nearly equal in height, which they
# are in two groups here. Both knobs give a little: at 0.82 and 0.92 the narrowest gap is
# ~1.6 mm, where widening the pitch alone crowds the groups and shrinking the label alone
# would take it under 6 pt.
GROUP_FRAC = 0.82
LABEL_FS = ANNOT_FS * 0.92


def arm_stat(df, arm, col):
    """-> (mean, std or 0) of ``col`` over an arm's runs."""
    s = df.loc[df.arm == arm, col].dropna()
    return (float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0) if len(s) else (np.nan, 0.0)


def load_probes(probe_dir):
    """-> DataFrame[arm, tag, <metric>_pcc/_r2/_n] over every probe_*.json on disk."""
    rows = []
    for f in sorted(glob.glob(os.path.join(probe_dir, "probe_*.json"))):
        d = json.load(open(f))
        rec = {"arm": "raw" if d["tag"] == "raw" else "joint", "tag": d["tag"]}
        for block in ("zero_shot", "probe"):
            for k, v in d.get(block, {}).items():
                rec[f"{k}_pcc"], rec[f"{k}_r2"], rec[f"{k}_n"] = v["pcc"], v["r2"], v["n"]
        rows.append(rec)
    if not rows:
        raise SystemExit(f"no probe_*.json in {probe_dir} -- run scripts/shell/ec_sa_transfer.sh probe")
    return pd.DataFrame(rows)


def load_preds(probe_dir):
    """-> DataFrame[arm, run, seq, sa_pmic, probe_sa] over every probe run on disk."""
    frames = []
    for f in sorted(glob.glob(os.path.join(probe_dir, "probe_*_sa_test_preds.csv"))):
        run = os.path.basename(f)[len("probe_"):-len("_sa_test_preds.csv")]
        d = pd.read_csv(f)[["seq", "sa_pmic", "probe_sa"]]
        d["run"], d["arm"] = run, ("raw" if run == "raw" else "joint")
        frames.append(d)
    if not frames:
        raise SystemExit(f"no probe_*_sa_test_preds.csv in {probe_dir}")
    return pd.concat(frames, ignore_index=True)


def overlap_subsets(near, test=None):
    """-> DataFrame[seq, subset] tagging each S. aureus test peptide seen/near/unseen.

    "seen" is identity 1.0 against the E. coli training pool, i.e. the exact sequence was
    in it. Identity is LCS / max(len) from a global alignment; ec_label_copy() already
    computes it per test row.
    Pass its ``test`` frame to reuse an alignment pass a caller has already paid for."""
    if test is None:
        test, *_ = ec_label_copy(near=near)
    t = test.groupby("seq", as_index=False).agg(nn=("nn_ec_identity", "mean"))
    t["subset"] = np.where(t.nn >= 1.0, "seen", np.where(t.nn >= near, "near", "unseen"))
    return t[["seq", "subset"]]


def _value_labels(ax, xs, means, tops):
    """Value above each bar, clear of whatever the error bar reaches."""
    for x, m, top in zip(xs, means, tops):
        ax.annotate(f"{m:.3f}", xy=(x, top), xytext=(0, 3.5), textcoords="offset points",
                    ha="center", va="bottom", color=INK, zorder=4,
                    fontsize=plt.rcParams["xtick.labelsize"] * LABEL_FS)


def panel_targets(ax, probes):
    """Left: ridge-probe PCC per target, native vs joint. Error bars are +/-1 sd over seeds,
    and are omitted where there is no seed spread (the native arm is a single deterministic fit)."""
    pos, centers, span, w = group_layout(len(TARGETS), 2, group_frac=GROUP_FRAC)
    rows = []
    for i, (arm, color) in enumerate(ARMS_HERE):
        stats = [arm_stat(probes, arm, f"{t}_pcc") for t, _ in TARGETS]
        means, stds = [m for m, _ in stats], [s for _, s in stats]
        ax.bar(pos[i], means, width=w, color=color, zorder=2, label=ARMS[arm]["label"])
        # The native arm is one deterministic RidgeCV fit on one frozen backbone, so its sd is
        # exactly 0; a zero-length whisker still draws its caps and reads as a real error bar.
        err = np.nan_to_num(np.asarray(stds, float))
        sel = err > 0
        if sel.any():
            ax.errorbar(np.asarray(pos[i])[sel], np.asarray(means)[sel], yerr=err[sel],
                        fmt="none", ecolor=INK2, elinewidth=0.9,
                        capsize=2.5, capthick=0.9, zorder=3)
        _value_labels(ax, pos[i], means, np.add(means, stds))
        rows += [{"panel": "targets", "arm": arm, "group": t, "n": 323, "pcc": m, "err_lo": s,
                  "err_hi": s} for (t, _), m, s in zip(TARGETS, means, stds)]
    ax.set_xticks(centers)
    ax.set_xticklabels([lab for _, lab in TARGETS])
    ax.set_xlim(*span)
    ax.set_xlabel("Probe target")
    return rows


def panel_subsets(ax, preds, n_boot):
    """Right: the S. aureus probe per overlap bucket. Error bars are bootstrap 95% CIs."""
    pos, centers, span, w = group_layout(len(SUBSETS), 2, group_frac=GROUP_FRAC)
    rows, ns = [], {}
    for i, (arm, color) in enumerate(ARMS_HERE):
        means, los, his = [], [], []
        for key, _ in SUBSETS:
            # per run: the point estimate and its CI; then average across seeds, so the
            # bar is the typical run and the whisker the sampling error of one run
            per_run = []
            for _, d in preds[preds.arm == arm].groupby("run"):
                g = d if key == "all" else d[d.subset == key]
                y, yhat = g.sa_pmic.values, g.probe_sa.values
                per_run.append((pearsonr(y, yhat)[0], *boot_ci(y, yhat, n_boot)))
                ns[key] = len(g)
            p, lo, hi = np.mean(per_run, axis=0)
            means.append(p)
            los.append(max(p - lo, 0.0))
            his.append(max(hi - p, 0.0))
            rows.append({"panel": "subsets", "arm": arm, "group": key, "n": ns[key],
                         "pcc": p, "err_lo": p - lo, "err_hi": hi - p})
        ax.bar(pos[i], means, width=w, color=color, zorder=2)
        ax.errorbar(pos[i], means, yerr=[los, his], fmt="none", ecolor=INK2, elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=3)
        _value_labels(ax, pos[i], means, np.add(means, his))
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{lab}\n$n$ = {ns[k]}" for k, lab in SUBSETS])
    ax.set_xlim(*span)
    ax.set_xlabel(r"Overlap with $\it{E.}$ $\it{coli}$ train")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe-dir", default=os.path.join(ROOT, "results", "ec-sa-transfer", "probe"))
    ap.add_argument("--near", type=float, default=0.8,
                    help="identity threshold separating 'near' from 'unseen'")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figure"))
    args = ap.parse_args()

    probes = load_probes(args.probe_dir)
    print("bucketing the test peptides (sequence alignment) ...", flush=True)
    preds = load_preds(args.probe_dir).merge(overlap_subsets(args.near), on="seq")

    apply(font_scale=1.5)
    w, h = (c / 2.54 for c in PAGE_CM)
    span = w - PADS["left"] - PADS["right"] - PADS["gap"]
    widths = [span * r / sum(RATIOS) for r in RATIOS]
    bottom, height = PADS["bottom"] / h, 1 - (PADS["top"] + PADS["bottom"]) / h
    fig = plt.figure(figsize=(w, h))
    axes = []
    for i, pw in enumerate(widths):
        left = PADS["left"] + sum(widths[:i]) + i * PADS["gap"]
        axes.append(fig.add_axes((left / w, bottom, pw / w, height),
                                 sharey=axes[0] if axes else None))
    axes[1].tick_params(labelleft=False)

    rows = panel_targets(axes[0], probes)
    rows += panel_subsets(axes[1], preds, args.n_boot)
    axes[0].set_ylabel(r"$\it{S.}$ $\it{aureus}$ test PCC")
    # headlines rather than x labels: neither x axis carries a quantity, both are
    # categories, and the tick names below already say which
    for ax, title in zip(axes, TITLES):
        ax.set_xlabel("")
        ax.set_title(title, pad=6)
        # the unseen bucket's bootstrap CI reaches below zero on both arms -- clipping it
        # at 0 would hide exactly the fact the panel exists to show
        ax.set_ylim(-0.38, 1.22)
        # pinned: the headroom for the 0.98 bar's value label pushes the auto locator to
        # 0.5 steps
        ax.set_yticks([-0.25, 0.0, 0.25, 0.50, 0.75, 1.00])
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", length=0)    # the gridlines already locate the values
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    # one row at the foot: the headlines have the top, and the 0.98 bar leaves no in-panel
    # corner free once the values are labelled
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), columnspacing=2.2, handlelength=1.1)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format="%.3f"))
    os.makedirs(args.figdir, exist_ok=True)
    # the shared rcParams crop to a tight bbox, which would make the page smaller than
    # PAGE_CM asks for; bbox_inches=None does not turn that off (matplotlib reads None as
    # "use rcParams"), so pass the figure's own bbox
    save(fig, os.path.join(args.figdir, "ec_sa_why_transfer"),
         bbox_inches=fig.bbox_inches, pad_inches=0)
    save_csv(out, args.figdir, "ec_sa_why_transfer.csv")


if __name__ == "__main__":
    main()
