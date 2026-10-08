import torch
import torch.nn as nn
from transformers import EsmModel, AutoTokenizer
import os
from tqdm import tqdm

ESM_NAME = "facebook/esm2_t30_150M_UR50D"

def get_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


class ESMBackbone(nn.Module):
    """ESM-2 + mean pooling -> (B, hidden_size)."""

    def __init__(self, name, frozen=False, dropout=0.0):
        super().__init__()
        kw = {}
        if dropout > 0:
            kw = dict(hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
        self.esm = EsmModel.from_pretrained(name, add_pooling_layer=False, **kw)
        self.hidden_size = self.esm.config.hidden_size
        self.frozen = frozen
        if frozen:
            for p in self.esm.parameters():
                p.requires_grad = False
            self.esm.eval()  # 防止dropout在训练时被启用

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.esm.eval()  # 防止dropout在训练时被启用
        return self

    def forward(self, input_ids, attention_mask):
        out = self.esm(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        # CLS/EOS 一起参与 mean pooling
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        # 只对真实残基做 mean pooling，排除 CLS/EOS
        # m = attention_mask.clone()
        # m[:, 0] = 0  # 去掉 CLS（永远在第0位）
        # seq_lens = attention_mask.sum(1)  # 每条序列的真实长度（含CLS+EOS）
        # eos_pos = seq_lens - 1
        # m[torch.arange(m.size(0)), eos_pos] = 0
        # m = m.unsqueeze(-1).to(out.dtype)  # (B, L) -> (B, L, 1)
        return (out * m).sum(1) / m.sum(1).clamp_min(1e-6)  # mean pooling


@torch.no_grad()
def embed_sequences(
    sequences, tokenizer, backbone, max_len=64, batch_size=64, device="cpu"
):
    """Pooled ESM embeddings for a list of sequences -> tensor (N, hidden)."""
    backbone = backbone.to(device).eval()
    chunks = []
    for i in tqdm(
        range(0, len(sequences), batch_size),
        desc="embedding",
        leave=False,
        dynamic_ncols=True,
    ):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        pooled = backbone(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        chunks.append(pooled.float().cpu())
    return torch.cat(chunks, 0)  # 拼接所有样本的 pooled embedding


def cached_embeddings(sequences, cache_path, tokenizer, backbone, **kw):
    """Load embeddings from ``cache_path`` if present, else compute and save."""
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path)
    emb = embed_sequences(sequences, tokenizer, backbone, **kw)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(emb, cache_path)
    return emb
