import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from carp.dataset import read_split, all_positive_sequences, read_negative_pool
from carp.utils import NegativeSampler, aa_freq


_HERE = os.path.dirname(__file__)  # CARP/scripts
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # CARP
_NEG_FILE = os.path.join(_ROOT, "dataset", "negatives", "negpool.csv")
OUT_DIR = os.path.join(_ROOT, "dataset", "classification")


def length_auc(pos_len, neg_len):
    """ROC-AUC of using length alone to tell positive from negative. 0.5 = length
    carries no signal (shortcut removed); →1 or →0 = strong length shortcut."""
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.ones(len(pos_len)), np.zeros(len(neg_len))]
    score = np.r_[pos_len, neg_len].astype(float)
    return roc_auc_score(y, score)


def describe(name, seqs):
    L = np.sort(np.fromiter((len(s) for s in seqs), int))
    n = len(L)
    q = lambda p: int(L[int(p * (n - 1))])
    print(
        f"  {name:14s} n={n:5d}  len[min/q25/med/q75/max]="
        f"{L[0]}/{q(0.25)}/{q(0.5)}/{q(0.75)}/{L[-1]}  <=25aa:{(L <= 25).mean() * 100:.0f}%"
    )


def build_split(split, organism, pool_split, sampler, ratio, banned, neg_file):
    positives = read_split(organism, split)[0]
    pool = [s for s in read_negative_pool(pool_split, neg_file) if s not in banned]
    n_neg = int(round(ratio * len(positives)))
    negatives, replaced = sampler.sample(positives, pool, n_neg)

    rows = [{"sequence": s, "is_active": 1, "length": len(s)} for s in positives] + [
        {"sequence": s, "is_active": 0, "length": len(s)} for s in negatives
    ]
    df = pd.DataFrame(rows)

    print(
        f"[{split}]  pool={len(pool)} usable"
        + ("  (WITH replacement — ratio too high for matched pool)" if replaced else "")
    )
    describe("positives", positives)
    describe("negatives", negatives)
    la = length_auc([len(s) for s in positives], [len(s) for s in negatives])
    dc = np.abs(aa_freq_mean(positives) - aa_freq_mean(negatives)).sum()
    print(f"  length-only AUC: {la:.3f}   (0.50 = no length shortcut)")
    print(f"  composition L1 gap (pos vs neg): {dc:.3f}")
    return df


def aa_freq_mean(seqs):
    return np.mean([aa_freq(s) for s in seqs], axis=0)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--organism", default="E. coli", help="E. coli | S. aureus")
    p.add_argument(
        "--strategy",
        default="length_matched",
        choices=NegativeSampler.STRATEGIES,
        help="how negatives are matched to the AMPs: random=no match "
        "(keeps length shortcut); length_matched=match length "
        "distribution; composition_matched=match AA composition; "
        "dual=both at once (weight = length x composition)",
    )
    p.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=1.0,
        help="negatives per positive (e.g. 1, 3, 9)",
    )
    p.add_argument(
        "--comp-temp",
        type=float,
        default=0.05,
        help="softness of composition matching (smaller = stricter)",
    )
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    p.add_argument(
        "--neg-file",
        default=_NEG_FILE,
        help="negative pool CSV (sequence,is_active,source)",
    )
    p.add_argument("--out-dir", default=OUT_DIR, help="output directory")
    args = p.parse_args()

    sampler = NegativeSampler(
        strategy=args.strategy, seed=args.seed, comp_temp=args.comp_temp
    )
    banned = all_positive_sequences()  # never sample a known AMP as a negative

    org_tag = args.organism.replace(" ", "").replace(".", "")
    r_tag = f"{args.neg_pos_ratio:g}"
    print(
        f"organism={args.organism}\tstrategy={args.strategy}\tneg:pos={r_tag}\tseed={args.seed}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for split, pool_split in (("train", "train"), ("test", "test")):
        df = build_split(
            split,
            args.organism,
            pool_split,
            sampler,
            args.neg_pos_ratio,
            banned,
            args.neg_file,
        )
        out = os.path.join(
            args.out_dir, f"cls_{org_tag}_{args.strategy}_r{r_tag}_{split}.csv"
        )
        df.to_csv(out, index=False)
        print("saved ->", out)


if __name__ == "__main__":
    main()
