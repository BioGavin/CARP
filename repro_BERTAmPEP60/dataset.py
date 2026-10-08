"""ProtBERT-tokenized pMIC regression dataset.

ProtBERT expects space-separated residues; sequences from ``carp.dataset.read_split`` are
joined with `' '.join(seq)`. The tokenizer is injected so tests can stub it.
"""
import torch
from torch.utils.data import Dataset


class ProtBertTokenDataset(Dataset):
    def __init__(self, sequences, targets, tokenizer, max_len=64):
        self.sequences = list(sequences)
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        spaced = " ".join(list(str(self.sequences[i]).strip()))
        # Use the tokenizer's __call__ (canonical, version-stable) rather than the legacy
        # encode_plus, which newer transformers removes from the public API.
        enc = self.tokenizer(
            spaced,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor(float(self.targets[i]), dtype=torch.float),
        }


def load_tokenizer(name="Rostlab/prot_bert"):
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(name, do_lower_case=False)
