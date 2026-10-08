import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset


# CARP/carp/dataset.py -> CARP/dataset/bertampep
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_DIR = os.path.join(_ROOT, "dataset", "bertampep")
_CLS_DIR = os.path.join(_ROOT, "dataset", "classification")

_NEG_SPLIT_SOURCES = {"train": {"Fold1", "Fold2", "AMPBench_train"},
                      "test": {"Test", "AMPBench_test"}}
ORGANISM_CODE = {"E. coli": "EC", "S. aureus": "SA"}


def read_split(organism, split="train", data_dir=_DATA_DIR):
    """-> (sequences: list[str], pmic: 1-D float tensor) in BERT-AmPEP60 pMIC."""
    code = ORGANISM_CODE.get(organism)
    if code is None:
        raise ValueError(f"BERT-AmPEP60 only has E. coli / S. aureus, got {organism!r}")
    fname = {
        "full": f"{code}.csv",
        "train": f"train-{code}.csv",
        "test": f"test-{code}.csv",
    }[split]
    df = pd.read_csv(os.path.join(data_dir, fname))
    seqs = df["SEQUENCE"].astype(str).str.strip().str.upper().tolist()
    pmic = torch.tensor(df[f"{code}_pMIC"].astype(float).to_numpy(), dtype=torch.float)
    return seqs, pmic  # seqs: list[str], pmic: 1-D float tensor


def read_classification_split(organism, strategy, ratio: int, split, cls_dir=_CLS_DIR):
    """Read a dataset built by build_classification_data.py ->
    (sequences: list[str], labels: 1-D float tensor {0,1}). Filename encodes the
    organism / negative-sampling strategy / neg:pos ratio, so it matches the
    builder's output exactly."""
    org = organism.replace(" ", "").replace(".", "")  # E. coli -> Ecoli
    fname = f"cls_{org}_{strategy}_r{ratio:g}_{split}.csv"  # cls_Ecoli_length_matched_r1_train.csv
    df = pd.read_csv(os.path.join(cls_dir, fname))
    seqs = df["sequence"].astype(str).str.strip().str.upper().tolist()
    labels = torch.tensor(df["is_active"].astype(float).to_numpy(), dtype=torch.float)
    return seqs, labels


def read_negative_pool(split, neg_file):
    df = pd.read_csv(neg_file)
    sel = (df["is_active"] == 0) & df["source"].isin(_NEG_SPLIT_SOURCES[split])
    seqs = df.loc[sel, "sequence"].astype(str).str.strip().str.upper()
    return list(dict.fromkeys(seqs.tolist()))          # dedupe, keep order


def split_train_val(n, val_frac, seed):
    """Split the train set into train and validation sets (default 9:1).
    Args:
        n (int): number of samples in the train set
        val_frac (float): fraction of train set to use as validation set
        seed (int): random seed for reproducibility
    Returns:
        train_idx (np.ndarray): indices for the training set
        val_idx (np.ndarray): indices for the validation set
    """
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(
        idx
    )  # default_rng创建局部随机数生成器实例，不影响全局
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]  # train_idx, val_idx


def all_positive_sequences(data_dir=_DATA_DIR):
    """Every AMP sequence across organisms/splits — used to drop any pool
    sequence that is actually a known positive (label conflict)."""
    seen = set()
    for org in ORGANISM_CODE:
        for split in ("train", "test"):
            seen.update(read_split(org, split, data_dir)[0])
    return seen


def EmbeddingDataset(embeddings, targets):
    """Frozen-head path: a plain TensorDataset of (embedding, target)."""
    return TensorDataset(embeddings, targets)


class BertAmPEPTokenDataset(Dataset):
    """Tokenized bertampep sequences + pMIC target (for fine-tune path)."""

    def __init__(self, sequences, targets, tokenizer, max_len):
        self.sequences = sequences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.sequences[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": self.targets[i],
        }

    @classmethod
    def from_split(cls, organism, split, tokenizer, max_len=64, data_dir=_DATA_DIR):
        """Build from bertampep organism/split files (fine-tune path)."""
        seqs, targets = read_split(organism, split, data_dir)
        return cls(seqs, targets, tokenizer, max_len)


# train_extend.py refers to the fine-tune token dataset as TokenDataset.
TokenDataset = BertAmPEPTokenDataset
