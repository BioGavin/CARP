#!/usr/bin/env python3
"""Aggregate ESM-2 (native/frozen vs fine-tuned) x size classifier results into CSV tables.

Reads every result JSON in the classifier results dir (written by scripts/train_cls.py),
identifies each run's backbone (native = frozen raw ESM, no --init-from; fine-tuned =
regression-checkpoint backbone) and size from `args`, pulls the selected test metrics
(`result.test_at_best_val`, i.e. the epoch with best validation AUC), and writes:

  1. cls_per_seed_metrics.csv   one row per run (backbone x size x seed), raw values
  2. cls_summary_mean_std.csv   mean and sample std (ddof=1) across seeds per (backbone, size)

P@25% is reported as the p@200 cutoff: the test set has n=808 (prevalence 0.5), so
top-200 covers 200/808 = 24.75% ~= 25%. The exact top-202 is not recoverable because
per-example scores are not stored; p@200 is the closest saved cutoff. The other stored
count cutoffs (p@50, p@100, p@400) are carried through as-is for reference.

Usage:
    python3 scripts/stat_plot/stat_cls_esm.py
    python3 scripts/stat_plot/stat_cls_esm.py --results <dir> --outdir <dir>
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# esm_name substring -> (short label, #params in M) ; also fixes row order
SIZE_MAP = [
    ("esm2_t6_8M", "8M", 8),
    ("esm2_t12_35M", "35M", 35),
    ("esm2_t30_150M", "150M", 150),
    ("esm2_t33_650M", "650M", 650),
]
# backbone key -> pretty label ; also fixes row order (native before fine-tuned)
BACKBONE_LABELS = {"native": "Native", "finetuned": "Fine-tuned"}

# metric key in test_at_best_val -> output column name
METRICS = {
    "auc": "AUC",
    "aupr": "AUPR",
    "p@200": "P@25%",   # top-200/808 = 24.75% ~= 25%
    "f1": "F1",
    "precision": "Prec",
    "recall": "Rec",
    "mcc": "MCC",
    "acc": "Acc",
}
# extra count-cutoff columns carried through untouched (reference only)
EXTRA_PATK = {"p@50": "P@50", "p@100": "P@100", "p@400": "P@400"}

STRATS = ["random", "length_matched", "composition_matched", "dual"]


def size_of(esm_name):
    base = str(esm_name).split("/")[-1]
    for sub, label, params in SIZE_MAP:
        if base.startswith(sub):
            return label, params
    return base, -1


def backbone_of(args):
    """native = frozen raw ESM (no regression init); finetuned = init from reg checkpoint."""
    return "finetuned" if args.get("init_from") else "native"


def load(results_dir):
    """-> list of per-run dicts. On duplicate (backbone,size,seed), newest mtime wins."""
    by_key = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json")), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  ! skip {os.path.basename(f)}: {e}")
            continue
        a = d.get("args", {})
        r = d.get("result", {}).get("test_at_best_val", {})
        if "esm_name" not in a or not r:
            print(f"  ! skip {os.path.basename(f)}: missing args.esm_name or test_at_best_val")
            continue
        label, params = size_of(a["esm_name"])
        bk = backbone_of(a)
        seed = a.get("seed")
        row = {
            "backbone": bk,
            "backbone_label": BACKBONE_LABELS.get(bk, bk),
            "size": label,
            "params_M": params,
            "seed": seed,
            "organism": a.get("organism"),
            "train_r": a.get("neg_pos_ratio"),
            "test_r": a.get("test_neg_pos_ratio"),
            "epoch": r.get("epoch"),
            "n_test": r.get("n"),
            "prevalence": r.get("prevalence"),
            "file": os.path.basename(f),
        }
        for mk in list(METRICS) + list(EXTRA_PATK):
            row[mk] = r.get(mk)
        # mean cross-strategy AUC/AUPR (checks for a negative-sampling shortcut)
        ebs = d.get("eval_by_strategy", {})
        for m in ("auc", "aupr"):
            vals = [ebs[s][m] for s in STRATS if s in ebs and m in ebs[s]]
            row[f"xstrat_{m}"] = float(np.mean(vals)) if vals else np.nan
        by_key[(bk, params, seed)] = row  # newest mtime overwrites older dupes
    return list(by_key.values())


def per_seed_frame(rows):
    df = pd.DataFrame(rows)
    order_bk = {"native": 0, "finetuned": 1}
    df = df.sort_values(
        ["backbone", "params_M", "seed"],
        key=lambda s: s.map(order_bk) if s.name == "backbone" else s,
    ).reset_index(drop=True)
    return df


def summary_frame(df):
    metric_keys = list(METRICS) + list(EXTRA_PATK) + ["xstrat_auc", "xstrat_aupr"]
    out = []
    for (bk, params), g in df.groupby(["backbone", "params_M"], sort=False):
        rec = {
            "backbone": bk,
            "backbone_label": BACKBONE_LABELS.get(bk, bk),
            "size": g["size"].iloc[0],
            "params_M": params,
            "n_seeds": len(g),
            "seeds": ",".join(str(s) for s in sorted(g["seed"].dropna().tolist())),
        }
        for mk in metric_keys:
            v = g[mk].to_numpy(dtype=float)
            rec[f"{mk}_mean"] = np.nanmean(v)
            rec[f"{mk}_std"] = np.nanstd(v, ddof=1) if np.sum(~np.isnan(v)) > 1 else 0.0
        out.append(rec)
    sm = pd.DataFrame(out).sort_values(
        ["backbone", "params_M"],
        key=lambda s: s.map({"native": 0, "finetuned": 1}) if s.name == "backbone" else s,
    ).reset_index(drop=True)
    return sm


def print_table(sm):
    """Compact console preview in the paper's column order (mean_std, 3 dp)."""
    cols = list(METRICS.items())
    header = f"{'Backbone':18s} " + " ".join(f"{lbl:>13s}" for _, lbl in cols)
    print(header)
    for _, r in sm.iterrows():
        name = f"{r['backbone_label']} ({r['size']})"
        cells = []
        for mk, _ in cols:
            cells.append(f"{r[mk + '_mean']:.3f}_{r[mk + '_std']:.3f}")
        print(f"{name:18s} " + " ".join(f"{c:>13s}" for c in cells))


def main():
    here = os.path.dirname(os.path.abspath(__file__))       # .../scripts/stat_plot
    root = os.path.abspath(os.path.join(here, "..", ".."))  # CARP-scratch
    default_results = os.path.join(root, "results", "02-esm-classifier")
    default_outdir = os.path.join(root, "figure", "data")

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results", default=default_results, help="dir of classifier result JSONs")
    p.add_argument("--outdir", default=default_outdir, help="output dir for CSVs")
    args = p.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    rows = load(args.results)
    if not rows:
        raise SystemExit(f"no usable result JSONs in {args.results}")
    df = per_seed_frame(rows)
    sm = summary_frame(df)

    per_seed_csv = os.path.join(outdir, "cls_per_seed_metrics.csv")
    summary_csv = os.path.join(outdir, "cls_summary_mean_std.csv")
    df.to_csv(per_seed_csv, index=False)
    sm.to_csv(summary_csv, index=False)

    n_cfg = sm.shape[0]
    print(f"loaded {len(df)} runs across {n_cfg} configs "
          f"({df['seed'].nunique()} seeds: {sorted(df['seed'].dropna().unique().tolist())})")
    print(f"per-seed  -> {per_seed_csv}")
    print(f"summary   -> {summary_csv}")
    print()
    print_table(sm)


if __name__ == "__main__":
    main()
