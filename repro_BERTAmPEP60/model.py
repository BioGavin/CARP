"""ProtBERT + MLP head pMIC regressor — a re-implementation of BERT-AmPEP60's REG.

Faithful to the published model: full fine-tuning, CLS ``pooler_output`` pooling, and a
head of LayerNorm -> 512 -> 128 -> 1 with LeakyReLU + dropout. The head input dim is read
from the backbone config, so a tiny test BertModel works without downloading ProtBERT.
"""
import torch.nn as nn
from transformers import BertModel


class ProtBertRegressor(nn.Module):
    def __init__(self, name="Rostlab/prot_bert", dropout=0.2, bert=None):
        super().__init__()
        self.bert = bert if bert is not None else BertModel.from_pretrained(name)
        h = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 512),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.head(pooled).squeeze(-1)
