"""Train a ProtBERT pMIC regressor under CARP's protocol (val-selected, fixed test).

Mirrors scripts/train_reg.py --mode finetune: reuses carp.dataset/carp.metrics, selects
the epoch with the lowest validation MSE, and reports test_at_best_val (primary) plus
best_test (== the original BERT-AmPEP60 leaky reporting, for reference).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from carp.dataset import read_split, split_train_val, ORGANISM_CODE
from carp.metrics import regression_metrics
from repro_BERTAmPEP60.model import ProtBertRegressor
from repro_BERTAmPEP60.dataset import ProtBertTokenDataset, load_tokenizer

RESULTS_DIR = os.path.join(_ROOT, "results", "05-repro-bertampep60")
SEEDS = (0, 42, 123, 666, 2026)


class BestValTracker:
    """Keep the test metrics of the lowest-val-MSE epoch, and the running-min test MSE."""

    def __init__(self):
        self.best_val = {"mse": float("inf"), "epoch": -1}
        self.test_at_best_val = {}
        self.best_test = {"mse": float("inf"), "epoch": -1}

    def update(self, epoch, val_metrics, test_metrics):
        if val_metrics["mse"] < self.best_val["mse"]:
            self.best_val = {**val_metrics, "epoch": epoch}
            self.test_at_best_val = {**test_metrics, "epoch": epoch}
        if test_metrics["mse"] < self.best_test["mse"]:
            self.best_test = {**test_metrics, "epoch": epoch}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        pred = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.append(np.atleast_1d(pred.cpu().numpy()))
        trues.append(np.atleast_1d(batch["target"].numpy()))
    return regression_metrics(np.concatenate(preds), np.concatenate(trues))


def fit(model, train_loader, val_loader, test_loader, optimizer, device, epochs, grad_clip):
    loss_fn = nn.MSELoss()
    tracker = BestValTracker()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            pred = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(pred.view(-1), batch["target"].to(device).float().view(-1))
            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        vm = evaluate(model, val_loader, device)
        tm = evaluate(model, test_loader, device)
        tracker.update(epoch, vm, tm)
        print(f"epoch {epoch} val_mse={vm['mse']:.4f} test_mse={tm['mse']:.4f} "
              f"best_val={tracker.best_val['mse']:.4f}@{tracker.best_val['epoch']}")
    return {
        "selected_by": "val/mse",
        "best_val": tracker.best_val,
        "test_at_best_val": tracker.test_at_best_val,
        "best_test": tracker.best_test,
    }


def _subset(seqs, ys, idx):
    return [seqs[i] for i in idx], ys[idx]


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--organism", default="E. coli", help="E. coli | S. aureus")
    p.add_argument("--model-name", default="Rostlab/prot_bert", help="HF BERT id")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=3e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-ckpt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out", default=None, help="relative name -> results/05-repro-bertampep60/")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="CARP")
    p.add_argument("--wandb-name", default=None)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seqs_tr, y_tr = read_split(args.organism, "train")
    seqs_te, y_te = read_split(args.organism, "test")
    tr_idx, va_idx = split_train_val(len(y_tr), args.val_frac, args.seed)
    tr_seqs, tr_y = _subset(seqs_tr, y_tr, tr_idx)
    va_seqs, va_y = _subset(seqs_tr, y_tr, va_idx)

    tok = load_tokenizer(args.model_name)
    mk = lambda s, y: ProtBertTokenDataset(s, y, tok, args.max_len)
    train_loader = DataLoader(mk(tr_seqs, tr_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(mk(va_seqs, va_y), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(mk(seqs_te, y_te), batch_size=args.batch_size, shuffle=False)

    model = ProtBertRegressor(args.model_name, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    result = fit(model, train_loader, val_loader, test_loader, optimizer,
                 device, args.epochs, args.grad_clip)
    result.update({"n_train": len(tr_seqs), "n_val": len(va_seqs), "n_test": len(seqs_te)})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = args.out or f"protbert_{ORGANISM_CODE[args.organism]}_seed{args.seed}.json"
    out_path = out if os.path.isabs(out) else os.path.join(RESULTS_DIR, out)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "result": result}, f, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
