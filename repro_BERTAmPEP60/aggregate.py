"""Aggregate repro_BERTAmPEP60 per-run JSONs into a mean±std table.

For each organism, summarize the 'test_at_best_val' metrics (primary, selection="val")
and the 'best_test' metrics (reference == original leaky reporting) over seeds.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(_ROOT, "results", "05-repro-bertampep60")
METRICS = ["mse", "rmse", "r2", "pcc", "ktc", "spearman"]
_BLOCKS = {"val": "test_at_best_val", "best_test": "best_test"}


def summarize(results_dir):
    records = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        d = json.load(open(f))
        records.append((d.get("args", {}).get("organism"), d.get("result", {})))
    rows = []
    organisms = sorted({org for org, _ in records if org})
    for org in organisms:
        for sel, block in _BLOCKS.items():
            vals = {m: [] for m in METRICS}
            for org2, res in records:
                if org2 != org:
                    continue
                blk = res.get(block, {})
                for m in METRICS:
                    if blk.get(m) is not None:
                        vals[m].append(blk[m])
            n = max((len(vals[m]) for m in METRICS), default=0)
            if n == 0:
                continue
            row = {"organism": org, "selection": sel, "n_seeds": n}
            for m in METRICS:
                row[f"{m}_mean"] = float(np.mean(vals[m]))
                row[f"{m}_std"] = float(np.std(vals[m], ddof=1)) if len(vals[m]) > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_DIR)
    args = ap.parse_args()
    df = summarize(args.results)
    print(df.to_string(index=False))
    csv_path = os.path.join(args.results, "protbert_baseline_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
