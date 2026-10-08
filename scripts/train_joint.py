import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from carp.dataset import (
    BertAmPEPTokenDataset,
    read_classification_split,
    read_split,
    split_train_val,
)
from carp.esmfeature import get_tokenizer
from carp.metrics import classification_metrics, ranking_metrics, regression_metrics
from carp.models import ESMClassRegressor

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
RESULTS_DIR = os.path.join(_ROOT, "results", "03-joint-classreg")
CKPT_DIR = os.path.join(_ROOT, "checkpoints", "03-joint-classreg")


def _cycle(loader):
    while True:
        yield from loader


def save_ckpt(path, model, epoch, val_metrics, test_metrics, tag):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "selected_by": tag,
            "val": val_metrics,
            "test": test_metrics,
        },
        path,
    )


def build_bestval_ckpt_path(args, name):
    if not args.save_ckpt:
        return None
    org = args.organism.replace(" ", "").replace(".", "")
    base = f"joint_{args.esm_name.split('/')[-1]}_{org}_{args.strategy}_r{args.neg_pos_ratio:g}_lam{args.mse_weight:g}_seed{args.seed}"
    return os.path.join(CKPT_DIR, f"{base}_{name}.pt")




@torch.no_grad()
def evaluate(model, loader, device, head, metric_fn):
    preds, trues = [], []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))[
            head
        ]
        if head == "logit":
            out = torch.sigmoid(out)
        preds.append(out.detach().cpu().numpy())
        trues.append(batch["target"].numpy())
    return metric_fn(np.concatenate(preds), np.concatenate(trues))


def fit_joint(
    model, loaders, optimizer, epochs, device, mse_weight, grad_clip, wandb_on, ckpts
):
    mse_fn, bce_fn = nn.MSELoss(), nn.BCEWithLogitsLoss()
    lam = mse_weight
    reg_best_val = {"mse": float("inf"), "epoch": -1}
    all_test_at_best_reg_val = {}
    reg_best_test = {"mse": float("inf"), "epoch": -1}
    cls_best_val = {"auc": -float("inf"), "epoch": -1}
    all_test_at_best_cls_val = {}
    cls_best_test = {"auc": -float("inf"), "epoch": -1}
    # snapshots = {}

    n_steps = max(len(loaders["reg_train"]), len(loaders["cls_train"]))
    epoch_bar = tqdm(range(epochs), desc="train", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        reg_gen, cls_gen = (
            _cycle(loaders["reg_train"]),
            _cycle(loaders["cls_train"]),
        )  # 各取一个 batch: 16 条回归 + 16 条分类
        total_loss = mse_loss = bce_loss = 0.0
        for _ in range(n_steps):
            rb, cb = next(reg_gen), next(cls_gen)
            pmic = model(rb["input_ids"].to(device), rb["attention_mask"].to(device))[
                "pmic"
            ]
            logit = model(cb["input_ids"].to(device), cb["attention_mask"].to(device))[
                "logit"
            ]
            mse = mse_fn(pmic.view(-1), rb["target"].float().view(-1).to(device))
            bce = bce_fn(logit.view(-1), cb["target"].float().view(-1).to(device))
            loss = (1 - lam) * bce + lam * mse
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            mse_loss += mse.item()
            bce_loss += bce.item()
        total_loss, mse_loss, bce_loss = (
            x / n_steps for x in (total_loss, mse_loss, bce_loss)
        )

        model.eval()

        def regresion2_metrics(p, t):
            return {**regression_metrics(p, t), **ranking_metrics(p, t, (10, 50, 100))}

        rv = evaluate(model, loaders["reg_val"], device, "pmic", regresion2_metrics)
        cv = evaluate(
            model, loaders["cls_val"], device, "logit", classification_metrics
        )
        rt = evaluate(model, loaders["reg_test"], device, "pmic", regresion2_metrics)
        ct = evaluate(
            model, loaders["cls_test"], device, "logit", classification_metrics
        )

        # def both():
        #     return {
        #         "epoch": epoch,
        #         "reg_test": rt,
        #         "cls_test": ct,
        #     }  # this snapshot on both tasks

        if rv["mse"] < reg_best_val["mse"]:
            reg_best_val = {**rv, "epoch": epoch}
            reg_test_at_best_reg_val = {**rt, "epoch": epoch}
            cls_test_at_best_reg_val = {**ct, "epoch": epoch}
            all_test_at_best_reg_val = {
                **reg_test_at_best_reg_val,
                **cls_test_at_best_reg_val,
            }
            del all_test_at_best_reg_val["n"]
            all_test_at_best_reg_val["reg_n"] = reg_test_at_best_reg_val["n"]
            all_test_at_best_reg_val["cls_n"] = cls_test_at_best_reg_val["n"]
            # snapshots["reg_bestval"] = both()
            save_ckpt(ckpts.get("reg_bestval"), model, epoch, rv, rt, "reg/best_val")
        if rt["mse"] < reg_best_test["mse"]:
            reg_best_test = {**rt, "epoch": epoch}
            # snapshots["reg_besttest"] = both()
        if cv["auc"] > cls_best_val["auc"]:
            cls_best_val = {**cv, "epoch": epoch}
            cls_test_at_best_cls_val = {**ct, "epoch": epoch}
            reg_test_at_best_cls_val = {**rt, "epoch": epoch}
            all_test_at_best_cls_val = {
                **cls_test_at_best_cls_val,
                **reg_test_at_best_cls_val,
            }
            del all_test_at_best_cls_val["n"]
            all_test_at_best_cls_val["reg_n"] = reg_test_at_best_cls_val["n"]
            all_test_at_best_cls_val["cls_n"] = cls_test_at_best_cls_val["n"]
            # snapshots["cls_bestval"] = both()
            save_ckpt(ckpts.get("cls_bestval"), model, epoch, cv, ct, "cls/best_val")
        if ct["auc"] > cls_best_test["auc"]:
            cls_best_test = {**ct, "epoch": epoch}
            # snapshots["cls_besttest"] = both()

        epoch_bar.set_postfix(
            train_loss=f"{total_loss:.4f}",
            reg_val_mse=f"{rv['mse']:.4f}",
            cls_val_auc=f"{cv['auc']:.4f}",
            best_val=f"reg{reg_best_val['mse']:.4f}@{reg_best_val['epoch']}/cls{cls_best_val['auc']:.4f}@{cls_best_val['epoch']}",
        )
        if wandb_on:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": total_loss,
                    "train/mse": mse_loss,
                    "train/bce": bce_loss,
                    **{f"val/{k}": v for k, v in rv.items() if k != "n"},
                    **{f"test/{k}": v for k, v in rt.items() if k != "n"},
                    **{f"val/{k}": v for k, v in cv.items() if k != "n"},
                    **{f"test/{k}": v for k, v in ct.items() if k != "n"},
                    **{
                        f"best_val/{k}": v
                        for k, v in reg_best_val.items()
                        if k not in ("n", "epoch")
                    },
                    **{
                        f"test_at_best_reg_val/{k}": v
                        for k, v in all_test_at_best_reg_val.items()
                        if k not in ("n", "epoch")
                    },
                    **{
                        f"best_test/{k}": v
                        for k, v in reg_best_test.items()
                        if k not in ("n", "epoch")
                    },
                    **{
                        f"best_val/{k}": v
                        for k, v in cls_best_val.items()
                        if k not in ("n", "epoch")
                    },
                    **{
                        f"test_at_best_cls_val/{k}": v
                        for k, v in all_test_at_best_cls_val.items()
                        if k not in ("n", "epoch")
                    },
                    **{
                        f"best_test/{k}": v
                        for k, v in cls_best_test.items()
                        if k not in ("n", "epoch")
                    },
                }
            )
    epoch_bar.close()

    return {
        "selected_by": "val/mse-auc",
        "mse_weight": lam,
        "reg": {
            "best_val": reg_best_val,
            "all_test_at_best_val": all_test_at_best_reg_val,
            "best_test": reg_best_test,
        },
        "cls": {
            "best_val": cls_best_val,
            "all_test_at_best_val": all_test_at_best_cls_val,
            "best_test": cls_best_test,
        },
        # "snapshots": snapshots,
        "ckpt_val": ckpts,
    }


def build_model(args, device):
    model = ESMClassRegressor(
        args.esm_name,
        frozen=False,
        esm_dropout=args.esm_dropout,
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
    return model.to(device)


def build_loaders(args, tok):
    def loaders_for(seqs, targets, is_train, bs_train, bs_eval):
        ds = BertAmPEPTokenDataset(seqs, targets, tok, max_len=args.max_len)
        if not is_train:
            return DataLoader(ds, batch_size=bs_eval, shuffle=False)
        tr_idx, va_idx = split_train_val(len(ds), args.val_frac, args.seed)
        train = DataLoader(Subset(ds, tr_idx), batch_size=bs_train, shuffle=True)
        val = DataLoader(Subset(ds, va_idx), batch_size=bs_eval, shuffle=False)
        return train, val

    reg_seqs, reg_pmic = read_split(args.organism, "train")
    reg_train, reg_val = loaders_for(
        reg_seqs, reg_pmic, True, args.batch_size, 128
    )  # 获得回归train、val dataloader
    reg_te_seqs, reg_te_pmic = read_split(args.organism, "test")
    reg_test = loaders_for(
        reg_te_seqs, reg_te_pmic, False, None, 128
    )  # 获得回归test dataloader

    cls_seqs, cls_lab = read_classification_split(
        args.organism, args.strategy, args.neg_pos_ratio, "train"
    )
    cls_train, cls_val = loaders_for(cls_seqs, cls_lab, True, args.batch_size, 128)
    cls_te_seqs, cls_te_lab = read_classification_split(
        args.organism, args.strategy, args.test_neg_pos_ratio, "test"
    )
    cls_test = loaders_for(cls_te_seqs, cls_te_lab, False, None, 128)

    print(
        f"reg stream: train {len(reg_train.dataset)} / val {len(reg_val.dataset)} / test {len(reg_test.dataset)}"
    )
    print(
        f"cls stream: train {len(cls_train.dataset)} / val {len(cls_val.dataset)} / test {len(cls_test.dataset)}"
    )
    return {
        "reg_train": reg_train,
        "reg_val": reg_val,
        "reg_test": reg_test,
        "cls_train": cls_train,
        "cls_val": cls_val,
        "cls_test": cls_test,
    }


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--organism", default="E. coli", help="E. coli | S. aureus")
    p.add_argument(
        "--mse-weight",
        type=float,
        default=0.5,
        help="λ in loss=(1-λ)·BCE + λ·MSE; 0=pure cls, 1=pure reg",
    )
    p.add_argument(
        "--strategy",
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
        "--esm-name",
        default="facebook/esm2_t30_150M_UR50D",
        help="HF ESM-2 id",
    )
    p.add_argument(
        "--init-from",
        default=None,
        help="init backbone",
    )
    p.add_argument("--max-len", type=int, default=64, help="token truncation length")
    p.add_argument("--batch-size", type=int, default=16, help="batch size (per stream)")
    p.add_argument("--epochs", type=int, default=100, help="epochs")
    p.add_argument("--head-lr", type=float, default=1e-4, help="head LR")
    p.add_argument("--esm-lr", type=float, default=1e-5, help="backbone LR")
    p.add_argument("--weight-decay", type=float, default=3e-3, help="weight decay")
    p.add_argument("--head-dropout", type=float, default=0.2, help="head dropout")
    p.add_argument("--esm-dropout", type=float, default=0.0, help="ESM dropout")
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
        help="save per-task best-on-val weights (2 files)",
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
        f"joint finetune  organism={args.organism}  λ(mse)={args.mse_weight:g}  "
        f"esm={args.esm_name}  device={device}",
        flush=True,
    )

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode
        )
        wandb.config.update(vars(args))

    tokenizer = get_tokenizer(args.esm_name)
    model = build_model(args, device)
    loaders = build_loaders(args, tokenizer)
    optimizer = torch.optim.AdamW(
        model.param_groups(args.esm_lr, args.head_lr, args.weight_decay)
    )
    ckpts = {
        name: build_bestval_ckpt_path(args, name)
        for name in ("reg_bestval", "cls_bestval")
    }
    result = fit_joint(
        model,
        loaders,
        optimizer,
        args.epochs,
        device,
        args.mse_weight,
        args.grad_clip,
        args.wandb,
        ckpts,
    )

    if args.wandb:
        wandb.finish()

    org = args.organism.replace(" ", "").replace(".", "")
    default = f"joint_finetune_MSEWeight{args.mse_weight}_{args.esm_name.split('/')[-1]}_{org}_{args.strategy}_r{args.neg_pos_ratio:g}_seed{args.seed}.json"
    out = os.path.join(RESULTS_DIR, args.out_json or default)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "result": result,
            },
            f,
            indent=2,
        )
    print("saved ->", out)
