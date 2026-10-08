"""Extend a frozen CARP base model with a new-organism regression head.

Loads a step-1 regression fine-tune (ESMRegressor) or a step-3 joint
(ESMClassRegressor) checkpoint, freezes the ESM-2 backbone and every existing
head, adds a fresh regression head for a new organism, and trains ONLY that
head. Because the trunk never changes, forgetting on the base organism is 0 by
construction (asserted)."""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carp.dataset import TokenDataset, read_classification_split, read_split
from carp.esmfeature import ESM_NAME, get_tokenizer
from carp.metrics import classification_metrics, ranking_metrics, regression_metrics
from carp.models import ESMMultiOrganismRegressor, _key

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
RESULTS_DIR = os.path.join(_ROOT, "results", "04-extend")
CKPT_DIR = os.path.join(_ROOT, "checkpoints", "04-extend")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def split_train_val(n, val_frac, seed):
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def subsample(n, k, seed):
    """Indices of a size-k random subset of range(n) (k<=0 or k>=n -> all)."""
    if k is None or k <= 0 or k >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def resolve_base_mode(args):
    """Return "raw" or "ckpt". Exactly one of --raw-base / --base-ckpt required."""
    if args.raw_base and args.base_ckpt:
        raise SystemExit("pass exactly one of --base-ckpt or --raw-base, not both")
    if not args.raw_base and not args.base_ckpt:
        raise SystemExit("pass one of --base-ckpt <path> or --raw-base")
    return "raw" if args.raw_base else "ckpt"


def _reg_metric(p, t):
    return {**regression_metrics(p, t), **ranking_metrics(p, t, (10, 50, 100))}


@torch.no_grad()
def eval_head(model, loader, device, organism, head):
    """head='pmic' -> regression+ranking metrics; head='logit' -> classification
    metrics on sigmoid probs (organism-independent shared head)."""
    preds, trues = [], []
    for b in loader:
        out = model(b["input_ids"].to(device), b["attention_mask"].to(device), organism)[head]
        if head == "logit":
            out = torch.sigmoid(out)
        preds.append(out.detach().cpu().numpy())
        trues.append(b["target"].numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return _reg_metric(p, t) if head == "pmic" else classification_metrics(p, t)


def load_base(model, ckpt, base_org):
    """Populate ``model`` from the base checkpoint. A saved multi-organism ckpt
    (reg_heads.*) -> add base organism + non-strict load; otherwise defer to
    load_phase1_base (auto-detects step-1 regression vs step-3 joint)."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    if any(k.startswith("reg_heads.") for k in sd):
        model.add_organism(base_org)
        if any(k.startswith("cls_head.") for k in sd):
            model._ensure_cls_head()      # so a saved classification head is restored, not dropped
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return {"kind": "multiorg", "missing": list(missing), "unexpected": list(unexpected)}
    return model.load_phase1_base(ckpt, organism=base_org)


def init_wandb(args):
    if not args.wandb:
        return None
    import wandb
    return wandb.init(project=args.wandb_project, name=args.wandb_name or ckpt_name(args),
                      config=vars(args), mode=args.wandb_mode)


def fit_frozen(model, new_org, loaders, optimizer, epochs, device, grad_clip, wandb_run=None):
    """Train ONLY the new organism's head. Returns (best_val_metrics, best_head_state)."""
    mse_fn = nn.MSELoss()
    new_key = _key(new_org)
    best = {"mse": float("inf"), "epoch": -1}
    best_head = None
    for epoch in tqdm(range(epochs), desc="frozen", dynamic_ncols=True):
        model.train()                                   # trunk stays eval (frozen)
        for b in loaders["new_train"]:
            pmic = model(b["input_ids"].to(device), b["attention_mask"].to(device), new_org)["pmic"]
            loss = mse_fn(pmic.view(-1), b["target"].float().view(-1).to(device))
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), grad_clip)
            optimizer.step()
        model.eval()
        v = eval_head(model, loaders["new_val"], device, new_org, "pmic")
        if v["mse"] < best["mse"]:
            best = {**v, "epoch": epoch}
            best_head = copy.deepcopy(
                {k: t.detach().cpu() for k, t in model.reg_heads[new_key].state_dict().items()})
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "new_val/best_mse": best["mse"],
                           **{f"new_val/{k}": val for k, val in v.items() if k != "n"}})
    return best, best_head


def build_loaders(args, tok, need_cls, has_base):
    def loader(seqs, targets, bs, shuffle=False, idx=None):
        ds = TokenDataset(seqs, targets, tok, max_len=args.max_len)
        if idx is not None:
            ds = Subset(ds, idx)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    # NEW organism regression stream (subsample train -> the data-efficiency knob)
    ns_seqs, ns_pmic = read_split(args.new_organism, "train")
    keep = subsample(len(ns_seqs), args.n_sub, args.seed)
    tr_idx, va_idx = split_train_val(len(keep), args.val_frac, args.seed)
    new_train = loader(ns_seqs, ns_pmic, args.batch_size, shuffle=True, idx=keep[tr_idx])
    new_val = loader(ns_seqs, ns_pmic, 128, idx=keep[va_idx])
    nt_seqs, nt_pmic = read_split(args.new_organism, "test")
    new_test = loader(nt_seqs, nt_pmic, 128)

    loaders = {"new_train": new_train, "new_val": new_val, "new_test": new_test}
    print(f"new({args.new_organism}) reg: train {len(new_train.dataset)} "
          f"(of {len(ns_seqs)}, n_sub={args.n_sub}) / val {len(new_val.dataset)} / test {len(new_test.dataset)}")

    if has_base:  # OLD organism regression test set (regression forgetting probe)
        oe_seqs, oe_pmic = read_split(args.base_organism, "test")
        loaders["old_reg_test"] = loader(oe_seqs, oe_pmic, 128)
        print(f"old({args.base_organism}) reg_test {len(loaders['old_reg_test'].dataset)}")

    if need_cls:  # classification forgetting probe (only when the base has a cls head)
        oc_seqs, oc_lab = read_classification_split(
            args.base_organism, args.strategy, args.neg_pos_ratio, "test")
        loaders["old_cls_test"] = loader(oc_seqs, oc_lab, 128)
        print(f"old({args.base_organism}) cls_test {len(loaders['old_cls_test'].dataset)}")
    return loaders


def ckpt_name(args):
    new = args.new_organism.replace(" ", "").replace(".", "")
    n = "full" if args.n_sub <= 0 else f"n{args.n_sub}"
    base = "rawbase" if args.raw_base else "frozen"
    return f"extend_{new}_{base}_{n}_seed{args.seed}"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", default=None,
                   help="base checkpoint (step-1 reg or step-3 joint); omit when using --raw-base")
    p.add_argument("--raw-base", action="store_true",
                   help="use a raw pretrained ESM-2 backbone (no base ckpt, no base organism)")
    p.add_argument("--base-organism", default="E. coli",
                   help="organism already in the base (ignored with --raw-base)")
    p.add_argument("--new-organism", default="S. aureus", help="organism to add")
    p.add_argument("--n-sub", type=int, default=0,
                   help="subsample new-organism train to N (<=0 = full)")
    p.add_argument("--esm-name", default=ESM_NAME, help="HF ESM-2 id (MUST match base)")
    p.add_argument("--strategy", default="length_matched", help="cls dataset (forgetting probe)")
    p.add_argument("--neg-pos-ratio", type=float, default=1.0, help="cls neg:pos ratio")
    p.add_argument("--max-len", type=int, default=64, help="token truncation length")
    p.add_argument("--batch-size", type=int, default=16, help="batch size")
    p.add_argument("--epochs", type=int, default=60, help="epochs")
    p.add_argument("--head-lr", type=float, default=1e-4, help="new-head LR")
    p.add_argument("--weight-decay", type=float, default=3e-3, help="weight decay")
    p.add_argument("--dropout", type=float, default=0.2, help="head dropout")
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    p.add_argument("--val-frac", type=float, default=0.1, help="new-train fraction held as val")
    p.add_argument("--seed", type=int, default=0, help="seed (subsample + val split)")
    p.add_argument("--out", default=None, help="output JSON (relative name -> results/04-extend/)")
    p.add_argument("--save-ckpt", action=argparse.BooleanOptionalAction, default=True,
                   help="save the extended multi-organism model")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="CARP", help="wandb project")
    p.add_argument("--wandb-name", default=None, help="wandb run name (auto if unset)")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"], help="wandb mode")
    args = p.parse_args()
    mode = resolve_base_mode(args)          # "raw" | "ckpt" (fails fast on both/neither)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_desc = "raw ESM-2" if mode == "raw" else args.base_organism
    print(f"extend  base={base_desc} -> new={args.new_organism}  frozen  "
          f"n_sub={args.n_sub}  esm={args.esm_name}  device={device}", flush=True)

    wandb_run = init_wandb(args)
    tok = get_tokenizer(args.esm_name)
    model = ESMMultiOrganismRegressor(args.esm_name, frozen=True, head_dropout=args.dropout)
    if mode == "raw":
        info = {"kind": "raw", "esm_name": args.esm_name}   # keep the raw HF-pretrained backbone; no base head
    else:
        info = load_base(model, args.base_ckpt, args.base_organism)
    print("base load:", info)

    model.freeze_trunk()
    new_key = model.add_organism(args.new_organism)
    model.to(device)          # move the fully-assembled model (incl. the new head) as a unit
    need_cls = model.cls_head is not None
    has_base = (mode == "ckpt")             # is there a base-organism head to probe for forgetting?
    loaders = build_loaders(args, tok, need_cls, has_base)

    # forgetting baseline (old-organism metrics BEFORE training the new head)
    if has_base:
        model.eval()
        old_reg_before = eval_head(model, loaders["old_reg_test"], device, args.base_organism, "pmic")
        old_cls_before = (eval_head(model, loaders["old_cls_test"], device, args.base_organism, "logit")
                          if need_cls else None)
    else:
        old_reg_before = old_cls_before = None

    optimizer = torch.optim.AdamW(model.param_groups(args.head_lr, args.weight_decay))
    n_trainable = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print(f"trainable params: {n_trainable}")

    best_val, best_head = fit_frozen(model, args.new_organism, loaders, optimizer,
                                     args.epochs, device, args.grad_clip, wandb_run)
    if best_head is not None:
        model.reg_heads[new_key].load_state_dict(best_head)

    model.eval()
    new_test = eval_head(model, loaders["new_test"], device, args.new_organism, "pmic")

    if has_base:
        old_reg_after = eval_head(model, loaders["old_reg_test"], device, args.base_organism, "pmic")
        old_cls_after = (eval_head(model, loaders["old_cls_test"], device, args.base_organism, "logit")
                         if need_cls else None)
        d_rmse = old_reg_after["rmse"] - old_reg_before["rmse"]
        d_spear = old_reg_after["spearman"] - old_reg_before["spearman"]
        # trunk + old head never changed -> regression forgetting must be exactly 0
        assert abs(d_rmse) < 1e-6 and abs(d_spear) < 1e-6, \
            f"frozen leaked regression forgetting: dRMSE={d_rmse} dSpear={d_spear}"
        d_auc = None
        if need_cls:
            d_auc = old_cls_after["auc"] - old_cls_before["auc"]
            assert abs(d_auc) < 1e-6, f"frozen leaked classification forgetting: dAUC={d_auc}"
    else:
        old_reg_after = old_cls_after = None
        d_rmse = d_spear = d_auc = None

    print(f"\n[NEW {args.new_organism}] test @best-val (epoch {best_val['epoch']}): "
          f"rmse {new_test['rmse']:.4f}  mse {new_test['mse']:.4f}  pcc {new_test['pcc']:.4f}  "
          f"spearman {new_test['spearman']:.4f}  ktc {new_test['ktc']:.4f}")
    if has_base:
        auc_str = f"{d_auc:+.6f}" if d_auc is not None else "n/a (no cls head)"
        print(f"[OLD {args.base_organism}] forgetting (0 by construction): "
              f"ΔRMSE {d_rmse:+.6f}  ΔSpearman {d_spear:+.6f}  ΔAUC {auc_str}")
    else:
        print("[RAW base] no forgetting probe (no base head)")

    if wandb_run is not None:
        summary = {**{f"new_test/{k}": v for k, v in new_test.items() if k != "n"},
                   "new/best_val_epoch": best_val["epoch"],
                   "new/n_train": len(loaders["new_train"].dataset)}
        if has_base:
            summary["forgetting/delta_rmse"] = d_rmse
            summary["forgetting/delta_spearman"] = d_spear
        wandb_run.summary.update(summary)
        wandb_run.finish()

    forgetting = None if not has_base else {
        "base_organism": args.base_organism,
        "reg_before": old_reg_before, "reg_after": old_reg_after,
        "cls_before": old_cls_before, "cls_after": old_cls_after,
        "delta_rmse": d_rmse, "delta_spearman": d_spear, "delta_auc": d_auc}
    result = {
        "new_organism": {"organism": args.new_organism, "n_sub": args.n_sub,
                         "n_train": len(loaders["new_train"].dataset),
                         "best_val": best_val, "test": new_test},
        "forgetting": forgetting,
        "base_load": info}

    if args.save_ckpt:
        os.makedirs(CKPT_DIR, exist_ok=True)
        path = os.path.join(CKPT_DIR, ckpt_name(args) + ".pt")
        torch.save({"state_dict": model.state_dict(), "organisms": model.organisms(),
                    "args": vars(args), "new_test": new_test}, path)
        print("saved ckpt ->", path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, args.out or (ckpt_name(args) + ".json"))
    with open(out, "w") as f:
        json.dump({"args": vars(args), "result": result}, f, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
