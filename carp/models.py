import torch.nn as nn
import torch
from carp.esmfeature import ESMBackbone


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=(512, 128), dropout=0.2):
        super().__init__()
        layers = [nn.LayerNorm(in_dim)]
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ESMRegressor(nn.Module):
    """End-to-end: tokenized ids -> ESM-2 -> head. Use frozen=False to fine-tune."""

    def __init__(
        self,
        esm_name,
        frozen=False,
        esm_dropout=0.0,
        head_hidden=(512, 128),
        head_dropout=0.2,
    ):
        super().__init__()
        self.backbone = ESMBackbone(esm_name, frozen=frozen, dropout=esm_dropout)
        self.head = MLPHead(self.backbone.hidden_size, head_hidden, head_dropout)

    def forward(self, input_ids, attention_mask):
        return self.head(self.backbone(input_ids, attention_mask))

    def param_groups(self, esm_lr, head_lr, weight_decay):
        """Discriminative LR: backbone at esm_lr, head at head_lr. No decay on biases / LayerNorm (ndim <= 1)."""
        groups = []
        for module, lr in [(self.backbone, esm_lr), (self.head, head_lr)]:
            decay = [p for p in module.parameters() if p.requires_grad and p.ndim > 1]
            nodecay = [
                p for p in module.parameters() if p.requires_grad and p.ndim <= 1
            ]
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if nodecay:
                groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
        return groups


class ESMClassRegressor(nn.Module):
    def __init__(
        self,
        esm_name,
        frozen,
        esm_dropout=0.0,
        head_hidden=(512, 128),
        head_dropout=0.2,
    ):
        super().__init__()
        self.backbone = ESMBackbone(esm_name, frozen=frozen, dropout=esm_dropout)
        h = self.backbone.hidden_size
        self.reg_head = MLPHead(h, head_hidden, head_dropout)  # frozen
        self.cls_head = MLPHead(h, head_hidden, head_dropout)

    def forward(self, input_ids, attention_mask):
        z = self.backbone(input_ids, attention_mask)
        return {"pmic": self.reg_head(z), "logit": self.cls_head(z)}

    def freeze_regression(self):
        for p in self.reg_head.parameters():
            p.requires_grad = False
        return self

    def load_init(self, ckpt_path, map_location="cpu"):
        sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)[
            "state_dict"
        ]
        remap = {}
        for k, v in sd.items():
            if k.startswith("head."):  # previous head == our reg_head
                remap["reg_head." + k[len("head.") :]] = v
            else:  # backbone.*
                remap[k] = v
        missing, unexpected = self.load_state_dict(remap, strict=False)
        # cls_head.* is expected to be "missing" (fresh); nothing else should be.
        leftover = [m for m in missing if not m.startswith("cls_head.")]
        return {
            "loaded": len(remap),
            "unexpected": list(unexpected),
            "missing_non_cls": leftover,
        }

    def param_groups(self, esm_lr, head_lr, weight_decay):
        modules = [(self.cls_head, head_lr), (self.reg_head, head_lr)]
        if not self.backbone.frozen:
            modules.append((self.backbone, esm_lr))
        groups = []
        for module, lr in modules:
            decay = [p for p in module.parameters() if p.requires_grad and p.ndim > 1]
            nodecay = [
                p for p in module.parameters() if p.requires_grad and p.ndim <= 1
            ]
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if nodecay:
                groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
        return groups


def _key(organism: str) -> str:
    """Normalize an organism name into an nn.ModuleDict-safe key.

    'E. coli' -> 'Ecoli', 'S. aureus' -> 'Saureus'. Matches the classification
    file-naming convention and contains no '.' (illegal in ModuleDict keys)."""
    return organism.replace(" ", "").replace(".", "")


class ESMMultiOrganismRegressor(nn.Module):
    """Shared (frozen) ESM-2 backbone + one MLP regression head per organism,
    plus an optional shared classification head carried over from a joint base.

    Only heads added AFTER freeze_trunk() stay trainable, so extending the model
    with a new organism trains just that organism's head."""

    def __init__(self, esm_name, frozen=True, head_dropout=0.2, head_hidden=(512, 128)):
        super().__init__()
        self.backbone = ESMBackbone(esm_name, frozen=frozen)
        self.reg_heads = nn.ModuleDict()
        self.cls_head = None
        self.head_dropout = head_dropout
        self.head_hidden = tuple(head_hidden)
        self._names = {}  # key -> human organism name

    def _new_head(self):
        return MLPHead(self.backbone.hidden_size, self.head_hidden, self.head_dropout)

    def add_organism(self, organism):
        """Register a fresh regression head for ``organism`` (no-op if it exists)."""
        key = _key(organism)
        if key not in self.reg_heads:
            self.reg_heads[key] = self._new_head()
            self._names[key] = organism
        return key

    def organisms(self):
        return [self._names[k] for k in self.reg_heads.keys()]

    def _ensure_cls_head(self):
        if self.cls_head is None:
            self.cls_head = self._new_head()
        return self.cls_head

    def load_phase1_base(self, ckpt_path, organism, map_location="cpu"):
        """Load a step-1 regression (ESMRegressor: backbone.* + head.*) or step-3
        joint (ESMClassRegressor: backbone.* + reg_head.* + cls_head.*) checkpoint.
        The base organism gets a reg head; a cls head is created iff present."""
        sd = torch.load(ckpt_path, map_location=map_location, weights_only=False)["state_dict"]
        has_reg_head = any(k.startswith("reg_head.") for k in sd)
        kind = "joint" if has_reg_head else "regression"
        reg_prefix = "reg_head." if has_reg_head else "head."

        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(bb, strict=True)

        key = self.add_organism(organism)
        reg = {k[len(reg_prefix):]: v for k, v in sd.items() if k.startswith(reg_prefix)}
        self.reg_heads[key].load_state_dict(reg, strict=True)

        has_cls = any(k.startswith("cls_head.") for k in sd)
        if has_cls:
            cls = {k[len("cls_head."):]: v for k, v in sd.items() if k.startswith("cls_head.")}
            self._ensure_cls_head().load_state_dict(cls, strict=True)

        return {"kind": kind, "organism": organism, "has_cls": has_cls,
                "loaded_backbone": len(bb)}

    def freeze_trunk(self):
        """Freeze the backbone and every head that exists right now. Heads added
        afterwards stay trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        for head in self.reg_heads.values():
            for p in head.parameters():
                p.requires_grad = False
        if self.cls_head is not None:
            for p in self.cls_head.parameters():
                p.requires_grad = False
        return self

    def param_groups(self, head_lr, weight_decay):
        """AdamW groups over currently-trainable params (in frozen use, exactly
        the newest head). No weight decay on biases / LayerNorm (ndim <= 1)."""
        decay = [p for p in self.parameters() if p.requires_grad and p.ndim > 1]
        nodecay = [p for p in self.parameters() if p.requires_grad and p.ndim <= 1]
        groups = []
        if decay:
            groups.append({"params": decay, "lr": head_lr, "weight_decay": weight_decay})
        if nodecay:
            groups.append({"params": nodecay, "lr": head_lr, "weight_decay": 0.0})
        return groups

    def forward(self, input_ids, attention_mask, organism):
        z = self.backbone(input_ids, attention_mask)
        out = {"pmic": self.reg_heads[_key(organism)](z)}
        if self.cls_head is not None:
            out["logit"] = self.cls_head(z)
        return out
