# Training the classification head based on the fine-tuned ESM2
import os
import sys
import argparse
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import json
import pandas as pd

# make the carp package importable when run from CARP/ root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from carp.esmfeature import get_tokenizer, cached_embeddings
from carp.models import ESMClassRegressor, MLPHead
from carp.dataset import read_classification_split, split_train_val, EmbeddingDataset
from carp.metrics import classification_metrics
from carp.utils import save_ckpt


_HERE = os.path.dirname(__file__)  # CARP/scripts
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # CARP
CKPT_DIR = os.path.join(_ROOT, "checkpoints", "02-esm-classifier")
RESULTS_DIR = os.path.join(_ROOT, "results", "02-esm-classifier")
CACHE_DIR = os.path.join(_ROOT, "cache")


def build_model(args, device):
    model = ESMClassRegressor(
        args.esm_name,
        frozen=True,
        esm_dropout=args.esm_dropout,
        head_hidden=(512, 128),
        head_dropout=args.head_dropout,
    )
    if args.init_from:
        info = model.load_init(args.init_from)
        print(
            f"init from {args.init_from}: loaded {info['loaded']} tensors"
            + (
                f", MISSING(non-cls)={info['missing_non_cls']}"
                if info["missing_non_cls"]
                else ""
            )
        )
    model.freeze_regression()
    return model.to(device)


def build_bestval_ckpt_path(args):
    """best-val path, or None if --no-save-ckpt."""
    if not args.save_ckpt:
        return None
    org = args.organism.replace(" ", "").replace(".", "")
    base = f"fine-tuned_{args.esm_name.split('/')[-1]}_{org}_r{args.neg_pos_ratio}_seed{args.seed}"
    return os.path.join(CKPT_DIR, f"{base}_bestval.pt")


@torch.no_grad()
def evaluate(predict_fn, loader, device):
    probs, labels = [], []
    for batch in loader:
        logit, target = predict_fn(batch, device)
        probs.append(torch.sigmoid(logit).detach().cpu().numpy())
        labels.append(target.detach().cpu().numpy())
    return classification_metrics(np.concatenate(probs), np.concatenate(labels))


@torch.no_grad()
def eval_test_strategies(model, args, tok, device, strategies):
    model.to(device).eval()
    out = {}
    for strat in strategies:
        seqs, labels = read_classification_split(
            args.organism, strat, args.test_neg_pos_ratio, "test"
        )
        enc = tok(
            seqs,
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        )
        z = model.backbone(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        prob = torch.sigmoid(model.cls_head(z).view(-1)).cpu().numpy()
        out[strat] = classification_metrics(prob, labels.numpy())
    return out


def _trainable_state(model):
    """Clone only the trainable params (cls_head in frozen mode; + backbone in
    finetune) — cheap, and enough to restore the model to its best-val weights."""
    return {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }


def _load_trainable(model, state):
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, v in state.items():
            params[n].copy_(v)


def fit_cls(
    model,
    train_loader,
    val_loader,
    test_loader,
    predict_fn,
    optimizer,
    epochs,
    device,
    grad_clip=0.0,
    wandb_on=False,
    bestval_ckpt_path=None,
):
    """Train loop for binary classification. Selection = best VAL AUC (higher is
    better). Reports test@best-val (honest) and best-on-test (optimistic)."""
    loss_fn = nn.BCEWithLogitsLoss()
    best_val = {"auc": -float("inf"), "epoch": -1}
    test_at_best_val = {}
    best_test = {"auc": -float("inf"), "epoch": -1}
    best_state = None
    epoch_bar = tqdm(range(epochs), desc="train", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        batches = train_loader
        loss_sum, n_batches = 0.0, 0
        for batch in batches:
            logit, target = predict_fn(batch, device)
            loss = loss_fn(logit.view(-1), target.float().view(-1))
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        train_loss = loss_sum / max(1, n_batches)

        model.eval()
        vm = evaluate(predict_fn, val_loader, device)
        tm = evaluate(predict_fn, test_loader, device)
        if vm["auc"] > best_val["auc"]:
            best_val = {**vm, "epoch": epoch}
            test_at_best_val = {**tm, "epoch": epoch}
            best_state = _trainable_state(model)
            save_ckpt(bestval_ckpt_path, model, epoch, vm, tm, "best_val")
        if tm["auc"] > best_test["auc"]:
            best_test = {**tm, "epoch": epoch}
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_auc=f"{vm['auc']:.4f}",
            test_auc=f"{tm['auc']:.4f}",
            best_val=f"{best_val['auc']:.4f}@{best_val['epoch']}",
            best_test=f"{best_test['auc']:.4f}@{best_test['epoch']}",
        )
        if wandb_on:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    **{f"val/{k}": v for k, v in vm.items() if k != "n"},
                    **{f"test/{k}": v for k, v in tm.items() if k != "n"},
                    **{
                        f"best_val/{k}": v
                        for k, v in best_val.items()
                        if k not in ("n", "epoch")
                    },
                    **{
                        f"test_at_best_val/{k}": v
                        for k, v in test_at_best_val.items()
                        if k not in ("n", "epoch")
                    },
                    **{
                        f"best_test/{k}": v
                        for k, v in best_test.items()
                        if k not in ("n", "epoch")
                    },
                }
            )
    epoch_bar.close()

    if best_state:
        _load_trainable(model, best_state)

    return {
        "selected_by": "val/auc",
        "best_val": best_val,
        "test_at_best_val": test_at_best_val,
        "best_test": best_test,
        "ckpt_val": bestval_ckpt_path,
    }


def run_frozen(args, device):
    tokenizer = get_tokenizer(args.esm_name)
    model = build_model(args, device)
    esm_tag = args.esm_name.split("/")[-1]
    org = args.organism.replace(" ", "").replace(".", "")
    init_tag = (
        os.path.splitext(os.path.basename(args.init_from))[0]
        if args.init_from
        else "native"
    )

    emb = {}
    for split in ("train", "test"):
        ratio = args.neg_pos_ratio if split == "train" else args.test_neg_pos_ratio
        seqs, labels = read_classification_split(
            args.organism, args.strategy, ratio, split
        )
        cache = os.path.join(
            CACHE_DIR,
            f"cls_emb_{org}_{args.strategy}_r{ratio:g}_{esm_tag}_{init_tag}_{split}.pt",
        )
        e = cached_embeddings(
            seqs,
            cache,
            tokenizer,
            model.backbone,
            max_len=args.max_len,
            batch_size=128,
            device=device,
        )
        emb[split] = (e, labels)

    tr_idx, va_idx = split_train_val(len(emb["train"][0]), args.val_frac, args.seed)
    e_tr, y_tr = emb["train"]  # embeddings and labels
    train_ds = EmbeddingDataset(e_tr[tr_idx], y_tr[tr_idx])
    val_ds = EmbeddingDataset(e_tr[va_idx], y_tr[va_idx])
    test_ds = EmbeddingDataset(*emb["test"])
    print(f"split: train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.param_groups(args.esm_lr, args.head_lr, args.weight_decay)
    )

    def predict_fn(batch, dev):
        e, y = batch
        return model.cls_head(e.to(dev)), y.to(dev)

    bestval_ckpt_path = build_bestval_ckpt_path(args)
    return fit_cls(
        model,
        train_loader,
        val_loader,
        test_loader,
        predict_fn,
        optimizer,
        args.epochs,
        device,
        grad_clip=0.0,
        wandb_on=args.wandb,
        bestval_ckpt_path=bestval_ckpt_path,
    ), model


def get_args():
    p = argparse.ArgumentParser(
        description="Train a classification head on top of a regression fine-tuned  ESM-2, "
        "with the ESM-2 backbone and regression head frozen. "
        "Test on different negative-sampling strategies and negative-to-positive ratios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--organism", default="E. coli", help="E. coli | S. aureus")
    p.add_argument(
        "--strategy",
        choices=["random", "length_matched", "composition_matched", "dual"],
        default="length_matched",
        help="negative-sampling strategy (selects the cls dataset file)",
    )
    p.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=1.0,
        help="neg:pos ratio (selects the cls dataset file)",
    )
    p.add_argument(
        "--test-neg-pos-ratio",
        type=float,
        default=None,
        help="test neg:pos ratio (selects the cls dataset file for test)",
    )
    p.add_argument(
        "--eval-test-strategies",
        default=None,
        help="comma-separated negative-sampling strategies to additionally evaluate the final (best-val) model on",
    )
    p.add_argument(
        "--init-from",
        default=None,
        help="checkpoint to init backbone + regression head",
    )
    p.add_argument(
        "--esm-name", default="facebook/esm2_t30_150M_UR50D", help="HF ESM-2 model id"
    )
    p.add_argument("--max-len", type=int, default=64, help="token truncation length")
    p.add_argument("--batch-size", type=int, default=16, help="batch size")
    p.add_argument("--epochs", type=int, default=100, help="training epochs")
    p.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
        help="head LR",
    )
    p.add_argument("--esm-lr", type=float, default=1e-5, help="backbone LR (finetune)")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="weight decay",
    )
    p.add_argument("--head-dropout", type=float, default=0.2, help="head dropout")
    p.add_argument(
        "--esm-dropout", type=float, default=0.0, help="ESM dropout (finetune)"
    )
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    p.add_argument(
        "--val-frac", type=float, default=0.1, help="train fraction held as val"
    )
    p.add_argument("--seed", type=int, default=0, help="random seed (also val split)")
    p.add_argument("--out-json", default=None, help="output JSON path")
    p.add_argument(
        "--save-ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save best-on-val model weights",
    )
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="CARP", help="wandb project")
    p.add_argument("--wandb-name", default=None, help="wandb run name (auto if unset)")
    p.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode",
    )
    args = p.parse_args()

    if args.test_neg_pos_ratio is None:
        args.test_neg_pos_ratio = args.neg_pos_ratio

    return args


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"organism={args.organism} esm={args.esm_name} device={device}",
        flush=True,
    )

    if args.wandb:
        wandb.init(
            project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode
        )
        wandb.config.update(vars(args))

    result, model = run_frozen(args, device)
    eval_by_strategy = None
    if args.eval_test_strategies:
        strategies = [
            s.strip() for s in args.eval_test_strategies.split(",") if s.strip()
        ]
        tok = get_tokenizer(args.esm_name)
        print(
            f"\ncross-strategy TEST eval @r{args.test_neg_pos_ratio:g} "
            f"(same {args.organism} positives, negatives per strategy):"
        )
        eval_by_strategy = eval_test_strategies(model, args, tok, device, strategies)

    if args.wandb:
        wandb.summary.update(
            {
                "test_at_best_val/auc": result["test_at_best_val"]["auc"],
                "test_at_best_val/f1": result["test_at_best_val"]["f1"],
                "test_at_best_val/mcc": result["test_at_best_val"]["mcc"],
                "best_test/auc": result["best_test"]["auc"],
                "best_test/f1": result["best_test"]["f1"],
                "best_test/mcc": result["best_test"]["mcc"],
            }
        )

        wandb.finish()

    org = args.organism.replace(" ", "").replace(".", "")
    default = f"cls_fine-tuned_{args.esm_name.split('/')[-1]}_{org}_{args.strategy}_r{args.neg_pos_ratio:g}_seed{args.seed}.json"
    out = os.path.join(RESULTS_DIR, args.out_json or default)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "result": result,
                "eval_by_strategy": eval_by_strategy,
            },
            f,
            indent=2,
        )
    print("saved ->", out)
