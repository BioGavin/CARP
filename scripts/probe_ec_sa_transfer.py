#!/usr/bin/env python3
"""Probe *what* a frozen backbone carries over from E. coli to S. aureus.

For one frozen backbone (a raw pretrained ESM-2, or a step-1/step-3 E. coli checkpoint)
this embeds the EC-train / SA-train / SA-test pools once (mean-pooled, max_len 64, the
same features train_extend's heads see) and reports, on the SA test set:

  zero-shot (checkpoint arms only; no S. aureus label is used)
    - ec_head  : the base E. coli regression head applied to SA test peptides
    - cls_head : the AMP-classification head's logit (joint bases only)

  ridge linear probes on the frozen embeddings (standardised, RidgeCV)
    - sa        : SA-train -> SA pMIC                  (linear analogue of the SA head)
    - shared    : SA∩EC-train -> a*EC_pMIC + b          (the part of SA activity that
                                                          E. coli activity predicts)
    - residual  : SA∩EC-train -> SA_pMIC - (a*EC_pMIC+b) (the S. aureus-specific part)
    - ec        : EC-train -> EC pMIC                   (how linearly EC potency is encoded)
  a, b are fitted on SA∩EC-train peptides only; shared/residual are scored on the SA
  test peptides that also have an EC label.

Every per-peptide SA-test prediction is written alongside the metrics so the
seen / near / unseen split can be applied afterwards -- that is what
scripts/stat_plot/plot_ec_sa_why_transfer.py's overlap_subsets() does with them.

Usage:
    python3 scripts/probe_ec_sa_transfer.py --raw-base --esm-name facebook/esm2_t30_150M_UR50D
    python3 scripts/probe_ec_sa_transfer.py --base-ckpt <joint_..._seed0_reg_bestval.pt> --tag joint_seed0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from carp.dataset import read_split  # noqa: E402
from carp.esmfeature import ESM_NAME, embed_sequences, get_tokenizer  # noqa: E402
from carp.models import ESMMultiOrganismRegressor, _key  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "results", "ec-sa-transfer", "probe")
ALPHAS = np.logspace(-2, 5, 15)


def pool(organism, split):
    """-> DataFrame[seq, pmic], duplicate sequences averaged (as plot_ec_sa_label_overlap.dedup)."""
    seqs, pmic = read_split(organism, split)
    df = pd.DataFrame({"seq": seqs, "pmic": pmic.numpy()})
    return df.groupby("seq", as_index=False)["pmic"].mean()


def reg_metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res, ss_tot = np.sum((y - yhat) ** 2), np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "pcc": float(pearsonr(y, yhat)[0]),
            "spearman": float(spearmanr(y, yhat)[0]),
            "mse": float(np.mean((y - yhat) ** 2)), "r2": float(1 - ss_res / ss_tot)}


def ridge(X_tr, y_tr, X_te):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X_tr, y_tr)
    return m.predict(X_te), float(m[-1].alpha_)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-ckpt", default=None, help="step-1 / step-3 E. coli checkpoint")
    ap.add_argument("--raw-base", action="store_true", help="raw pretrained ESM-2 backbone")
    ap.add_argument("--esm-name", default=ESM_NAME, help="HF ESM-2 id (MUST match the ckpt)")
    ap.add_argument("--tag", default=None, help="output stem (default: raw / ckpt basename)")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    if args.raw_base == bool(args.base_ckpt):
        raise SystemExit("pass exactly one of --base-ckpt or --raw-base")
    tag = args.tag or ("raw" if args.raw_base else os.path.splitext(os.path.basename(args.base_ckpt))[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ESMMultiOrganismRegressor(args.esm_name, frozen=True)
    info = {"kind": "raw"} if args.raw_base else model.load_phase1_base(args.base_ckpt, "E. coli")
    model.to(device).eval()
    print(f"[{tag}] base: {info}  device={device}", flush=True)

    tok = get_tokenizer(args.esm_name)
    ec_tr, sa_tr, sa_te = pool("E. coli", "train"), pool("S. aureus", "train"), pool("S. aureus", "test")
    ec_te = pool("E. coli", "test")   # held-out E. coli, to ask what each head encodes at home
    emb = {name: embed_sequences(df.seq.tolist(), tok, model.backbone, max_len=args.max_len,
                                 batch_size=128, device=device)
           for name, df in [("ec_tr", ec_tr), ("sa_tr", sa_tr), ("sa_te", sa_te), ("ec_te", ec_te)]}

    preds = pd.DataFrame({"seq": sa_te.seq, "sa_pmic": sa_te.pmic})
    # base_ckpt is recorded because the checkpoint *variant* matters for reading the
    # results: a reg_bestval snapshot carries whatever cls head happened to exist at the
    # best regression epoch, which is not the best classification head.
    res = {"tag": tag, "base_ckpt": args.base_ckpt, "base_load": info,
           "esm_name": args.esm_name, "zero_shot": {}, "probe": {}, "alpha": {}}

    # --- zero-shot: E. coli heads applied to S. aureus test peptides ------------------
    with torch.no_grad():
        z = emb["sa_te"].to(device)
        if _key("E. coli") in model.reg_heads:
            preds["zs_ec_head"] = model.reg_heads[_key("E. coli")](z).cpu().numpy()
        if model.cls_head is not None:
            preds["zs_cls_logit"] = model.cls_head(z).cpu().numpy()
    for col in ("zs_ec_head", "zs_cls_logit"):
        if col in preds:
            res["zero_shot"][col] = reg_metrics(preds.sa_pmic, preds[col])

    # --- the same two heads on held-out E. coli, against E. coli pMIC -------------------
    # The regression head is at home here, so it bounds what the backbone can express; the
    # classification head answers a different question -- does "is this an AMP?" also rank
    # potency among peptides that are all active?
    ec_preds = pd.DataFrame({"seq": ec_te.seq, "ec_pmic": ec_te.pmic})
    with torch.no_grad():
        z = emb["ec_te"].to(device)
        if _key("E. coli") in model.reg_heads:
            ec_preds["ec_head"] = model.reg_heads[_key("E. coli")](z).cpu().numpy()
        if model.cls_head is not None:
            ec_preds["cls_logit"] = model.cls_head(z).cpu().numpy()
    res["ec_test"] = {f"{c}_vs_ec_pmic": reg_metrics(ec_preds.ec_pmic, ec_preds[c])
                      for c in ("ec_head", "cls_logit") if c in ec_preds}

    # --- ridge probes ------------------------------------------------------------------
    X = {k: v.numpy() for k, v in emb.items()}
    preds["probe_sa"], res["alpha"]["sa"] = ridge(X["sa_tr"], sa_tr.pmic.values, X["sa_te"])
    res["probe"]["sa"] = reg_metrics(preds.sa_pmic, preds.probe_sa)

    ec_map = dict(zip(ec_tr.seq, ec_tr.pmic))
    both_tr = sa_tr.seq.map(ec_map).notna().values
    ec_of_tr = sa_tr.seq[both_tr].map(ec_map).values
    a, b = np.polyfit(ec_of_tr, sa_tr.pmic.values[both_tr], 1)
    res["ec_to_sa_map"] = {"slope": float(a), "intercept": float(b), "n": int(both_tr.sum())}
    targets_tr = {"shared": a * ec_of_tr + b}
    targets_tr["residual"] = sa_tr.pmic.values[both_tr] - targets_tr["shared"]
    ec_of_te = sa_te.seq.map(ec_map).values.astype(float)
    preds["shared_true"] = a * ec_of_te + b                      # NaN where no EC label
    preds["residual_true"] = preds.sa_pmic - preds.shared_true
    has_ec = ~np.isnan(ec_of_te)
    for t in ("shared", "residual"):
        p, res["alpha"][t] = ridge(X["sa_tr"][both_tr], targets_tr[t], X["sa_te"])
        preds[f"probe_{t}"] = p
        res["probe"][t] = reg_metrics(preds[f"{t}_true"][has_ec], p[has_ec])

    # EC potency encoding: 5-fold CV on EC train (no SA involvement)
    from sklearn.model_selection import cross_val_predict
    ec_cv = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
                              X["ec_tr"], ec_tr.pmic.values, cv=5)
    res["probe"]["ec_cv"] = reg_metrics(ec_tr.pmic.values, ec_cv)

    for k, v in {**res["zero_shot"], **res["probe"], **res.get("ec_test", {})}.items():
        print(f"  {k:22s} n={v['n']:4d}  pcc={v['pcc']:.3f}  spearman={v['spearman']:.3f}  r2={v['r2']:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"probe_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    preds.to_csv(os.path.join(args.out_dir, f"probe_{tag}_sa_test_preds.csv"), index=False)
    if len(ec_preds.columns) > 2:
        ec_preds.to_csv(os.path.join(args.out_dir, f"probe_{tag}_ec_test_preds.csv"), index=False)
    # embeddings are git-ignored (cache/) but kept for later geometry plots
    cache = os.path.join(_ROOT, "cache", "ec-sa-transfer")
    os.makedirs(cache, exist_ok=True)
    torch.save({k: v for k, v in emb.items()}, os.path.join(cache, f"emb_{tag}.pt"))
    print(f"saved -> {args.out_dir}/probe_{tag}.json")


if __name__ == "__main__":
    main()
