import os
import torch
import numpy as np


AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX = {a: i for i, a in enumerate(AA)}


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


def aa_freq(seq):
    """20-dim amino-acid frequency vector (unknown chars ignored)."""
    v = np.zeros(len(AA))
    for c in seq:
        j = _AA_IDX.get(c)
        if j is not None:
            v[j] += 1
    s = v.sum()
    return v / s if s > 0 else v


def mean_composition(seqs):
    if not seqs:
        return np.ones(len(AA)) / len(AA)
    return np.mean([aa_freq(s) for s in seqs], axis=0)


class NegativeSampler:
    STRATEGIES = ("random", "length_matched", "composition_matched", "dual")

    def __init__(self, strategy, seed, comp_temp):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.STRATEGIES}, got {strategy!r}"
            )
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)
        self.comp_temp = comp_temp  # composition temp???

    def _length_ratio(self, positives, pool):  # 长度匹配权重
        """p_pos(len)/p_pool(len) per pool sequence (0 where AMPs never occur)."""
        pos_len = np.fromiter((len(s) for s in positives), int)
        pool_len = np.fromiter((len(s) for s in pool), int)
        m = int(max(pool_len.max(), pos_len.max())) + 1
        pos_h = np.bincount(pos_len, minlength=m).astype(float)
        pool_h = np.bincount(pool_len, minlength=m).astype(float)
        pos_h /= pos_h.sum()
        pool_h /= pool_h.sum()
        ratio = np.zeros(m)
        nz = pool_h > 0
        ratio[nz] = pos_h[nz] / pool_h[nz]
        return ratio[pool_len]

    def _comp_weight(self, positives, pool):  # 组成匹配权重
        """exp(-distance to the AMP mean composition), Euclidean on freq vectors."""
        pos_c = mean_composition(positives)
        pool_c = np.stack([aa_freq(s) for s in pool])
        dist = np.linalg.norm(pool_c - pos_c, axis=1)
        return np.exp(-dist / self.comp_temp)

    def weights(self, positives, pool):
        w = np.ones(len(pool), dtype=float)
        if self.strategy in ("length_matched", "dual"):
            w = w * self._length_ratio(positives, pool)
        if self.strategy in ("composition_matched", "dual"):
            w = w * self._comp_weight(positives, pool)
        return w

    def sample(self, positives, pool, n):  # 按权重概率采样
        """Return (sampled_sequences, used_replacement: bool). ``n`` negatives
        drawn from ``pool`` weighted toward ``positives`` per the strategy."""
        if n <= 0 or not pool:
            return [], False
        w = self.weights(positives, pool)
        total = w.sum()
        if total <= 0:  # degenerate (e.g. comp_temp too small)
            w = np.ones(len(pool))
            total = w.sum()
        p = w / total
        n_pos_weight = int((w > 0).sum())
        replace = n > n_pos_weight
        idx = self.rng.choice(len(pool), size=n, replace=replace, p=p)
        return [pool[i] for i in idx], replace
