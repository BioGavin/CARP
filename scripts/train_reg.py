import argparse
import torch
import numpy as np
import json
import wandb
import sys
import os
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from tqdm import tqdm

# make the carp package importable when run from CARP/ root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carp.esmfeature import ESMBackbone, get_tokenizer, cached_embeddings
from carp.models import MLPHead, ESMRegressor
from carp.dataset import (
    read_split,
    EmbeddingDataset,
    split_train_val,
    BertAmPEPTokenDataset,
)
from carp.metrics import regression_metrics, ranking_metrics
from carp.utils import save_ckpt


_HERE = os.path.dirname(__file__)  # CARP/scripts
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # CARP
CACHE_DIR = os.path.abspath(os.path.join(_ROOT, "cache"))
CKPT_DIR = os.path.join(_ROOT, "checkpoints", "01-esm-regressor")
RESULTS_DIR = os.path.join(_ROOT, "results", "01-esm-regressor")


def build_bestval_ckpt_path(args):
    """best-val path, or None if --no-save-ckpt."""
    if not args.save_ckpt:
        return None
    base = f"{args.esm_name.split('/')[-1]}_{args.mode}_{args.organism.replace(' ', '')}_seed{args.seed}"
    return os.path.join(CKPT_DIR, f"{base}_bestval.pt")


@torch.no_grad()
def evaluate(predict_fn, loader, device, type):
    preds, trues = [], []
    for batch in loader:
        out, target = predict_fn(batch, device)
        preds.append(out.detach().cpu().numpy())
        trues.append(target.detach().cpu().numpy())
    if type == "regression":
        return regression_metrics(np.concatenate(preds), np.concatenate(trues))
    elif type == "ranking":
        return ranking_metrics(
            np.concatenate(preds), np.concatenate(trues), ks=(10, 50, 100)
        )
    elif type == "both":
        return {
            **regression_metrics(np.concatenate(preds), np.concatenate(trues)),
            **ranking_metrics(
                np.concatenate(preds), np.concatenate(trues), ks=(10, 50, 100)
            ),
        }


def fit(
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
    loss_fn = nn.MSELoss()
    best_val = {"mse": float("inf"), "epoch": -1}
    test_at_best_val = {}
    best_test = {"mse": float("inf"), "epoch": -1}
    epoch_bar = tqdm(range(epochs), desc="train", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        batches = train_loader
        loss_sum, n_batches = 0.0, 0
        for batch in batches:
            out, target = predict_fn(batch, device)
            loss = loss_fn(out.view(-1), target.float().view(-1))
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        train_loss = loss_sum / max(1, n_batches)

        model.eval()
        vm = evaluate(predict_fn, val_loader, device, type="both")
        tm = evaluate(predict_fn, test_loader, device, type="both")
        if vm["mse"] < best_val["mse"]:
            best_val = {**vm, "epoch": epoch}
            test_at_best_val = {**tm, "epoch": epoch}
            save_ckpt(
                path=bestval_ckpt_path,
                model=model,
                epoch=epoch,
                val_metrics=vm,
                test_metrics=tm,
                tag="best_val",
            )
        if tm["mse"] < best_test["mse"]:
            best_test = {**tm, "epoch": epoch}
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_mse=f"{vm['mse']:.4f}",
            test_mse=f"{tm['mse']:.4f}",
            best_val=f"{best_val['mse']:.4f}@{best_val['epoch']}",
            best_test=f"{best_test['mse']:.4f}@{best_test['epoch']}",
        )
        if wandb_on:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    **{
                        f"val/{k}": v for k, v in vm.items() if k != "n"
                    },  # all val metrics except n
                    **{
                        f"test/{k}": v for k, v in tm.items() if k != "n"
                    },  # all test metrics except n
                    **{
                        f"best_val/{k}": v
                        for k, v in best_val.items()
                        if k not in ("epoch", "n")
                    },  # all best val metrics except epoch and n
                    **{
                        f"test_at_best_val/{k}": v
                        for k, v in test_at_best_val.items()
                        if k not in ("epoch", "n")
                    },  # all test at best val metrics except epoch and n
                    **{
                        f"best_test/{k}": v
                        for k, v in best_test.items()
                        if k not in ("epoch", "n")
                    },  # all best test metrics except epoch and n
                }
            )
    epoch_bar.close()
    return {
        "selected_by": "val/mse",
        "best_val": best_val,
        "test_at_best_val": test_at_best_val,
        "best_test": best_test,
        "ckpt_val": bestval_ckpt_path,
    }


def run_frozen(args, device):
    tokenizer = get_tokenizer(args.esm_name)
    backbone = ESMBackbone(args.esm_name, frozen=False, dropout=args.esm_dropout)
    esm_tag = args.esm_name.split("/")[-1]
    org_tag = args.organism.replace(" ", "").replace(".", "")  # E. coli -> Ecoli

    emb = {}
    for split in ("train", "test"):
        seqs, targets = read_split(
            args.organism, split
        )  # seqs: list[str], targets: 1-D float tensor
        cache = os.path.join(CACHE_DIR, f"bertampep_{esm_tag}_{org_tag}_{split}.pt")
        e = cached_embeddings(
            seqs,
            cache,
            tokenizer,
            backbone,
            max_len=args.max_len,
            batch_size=128,
            device=device,
        )
        emb[split] = (e, targets)

    # 从train_set中划分出val_set (9:1)
    tr_idx, va_idx = split_train_val(len(emb["train"][0]), args.val_frac, args.seed)
    e_tr, y_tr = emb["train"]
    train_ds = EmbeddingDataset(e_tr[tr_idx], y_tr[tr_idx])
    val_ds = EmbeddingDataset(e_tr[va_idx], y_tr[va_idx])
    test_ds = EmbeddingDataset(*emb["test"])
    print(f"split: train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = MLPHead(
        backbone.hidden_size, hidden=(512, 128), dropout=args.head_dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
    )

    def predict_fn(batch, dev):
        e, target = batch
        return model(e.to(dev)), target.to(dev)

    bestval_ckpt_path = build_bestval_ckpt_path(args)  # return ckpt path on best val
    return fit(
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
    )


def run_finetune(args, device):
    tokenizer = get_tokenizer(args.esm_name)
    full_train = BertAmPEPTokenDataset.from_split(
        args.organism, "train", tokenizer, max_len=args.max_len
    )
    test_ds = BertAmPEPTokenDataset.from_split(
        args.organism, "test", tokenizer, max_len=args.max_len
    )

    tr_idx, va_idx = split_train_val(len(full_train), args.val_frac, args.seed)
    train_ds, val_ds = Subset(full_train, tr_idx), Subset(full_train, va_idx)
    print(f"split: train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = ESMRegressor(
        args.esm_name,
        frozen=False,
        esm_dropout=args.esm_dropout,
        head_hidden=(512, 128),
        head_dropout=args.head_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(args.esm_lr, args.head_lr, args.weight_decay)
    )

    def predict_fn(batch, dev):
        out = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
        return out, batch["target"].to(dev)

    bestval_ckpt_path = build_bestval_ckpt_path(args)
    return fit(
        model,
        train_loader,
        val_loader,
        test_loader,
        predict_fn,
        optimizer,
        args.epochs,
        device,
        grad_clip=args.grad_clip,
        wandb_on=args.wandb,
        bestval_ckpt_path=bestval_ckpt_path,
    )


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--mode",
        choices=["frozen", "finetune"],
        required=True,
        help="frozen feature extractor vs end-to-end fine-tune",
    )
    p.add_argument("--organism", default="E. coli", help="target organism")
    p.add_argument(
        "--esm-name", default="facebook/esm2_t30_150M_UR50D", help="HF ESM-2 model id"
    )
    p.add_argument("--max-len", type=int, default=64, help="token truncation length")
    p.add_argument("--batch-size", type=int, default=16, help="batch size")
    # mode-dependent defaults are resolved after parsing (None means 'auto')
    p.add_argument(
        "--epochs", type=int, default=None, help="auto: 500 (frozen) / 100 (finetune)"
    )
    p.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="MLP head LR; auto: 1e-3 (frozen) / 1e-4 (finetune)",
    )
    p.add_argument(
        "--esm-lr", type=float, default=1e-5, help="backbone LR (finetune only)"
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="auto: 1e-2 (frozen) / 3e-3 (finetune)",
    )
    p.add_argument("--head-dropout", type=float, default=0.2, help="MLP head dropout")
    p.add_argument(
        "--esm-dropout", type=float, default=0.0, help="ESM-2 dropout (finetune only)"
    )
    p.add_argument(
        "--grad-clip", type=float, default=1.0, help="0 disables (finetune only)"
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="fraction of train held out as val (default: 9:1)",
    )
    p.add_argument("--seed", type=int, default=0, help="random seed (also val split)")
    p.add_argument("--out-json", default=None, help="output JSON path")
    p.add_argument(
        "--save-ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save best-on-val model weights",
    )
    # wandb logging (optional; lazily imported, off by default)
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

    # mode-specific defaults
    if args.epochs is None:
        args.epochs = 500 if args.mode == "frozen" else 100
    if args.head_lr is None:
        args.head_lr = 1e-3 if args.mode == "frozen" else 1e-4
    if args.weight_decay is None:
        args.weight_decay = 1e-2 if args.mode == "frozen" else 3e-3

    return args


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"mode={args.mode} organism={args.organism} esm={args.esm_name} "
        f"device={device}",
        flush=True,
    )

    if args.wandb:
        wandb.init(
            project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode
        )
        wandb.config.update(vars(args))

    run = run_frozen if args.mode == "frozen" else run_finetune
    result = run(args, device)

    if args.wandb:
        wandb.summary.update(
            {
                "test_at_best_val/mse": result["test_at_best_val"]["mse"],
                "test_at_best_val/pcc": result["test_at_best_val"]["pcc"],
                "test_at_best_val/ktc": result["test_at_best_val"]["ktc"],
                "best_test/mse": result["best_test"]["mse"],
                "best_test/pcc": result["best_test"]["pcc"],
                "best_test/ktc": result["best_test"]["ktc"],
            }
        )
        wandb.finish()

    org = args.organism.replace(" ", "").replace(".", "")
    default = f"{args.esm_name.split('/')[-1]}_{args.mode}_{org}_seed{args.seed}.json"
    out = os.path.join(RESULTS_DIR, args.out_json or default)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "result": result}, f, indent=2)
    print("saved ->", out)
