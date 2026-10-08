#!/usr/bin/env python3
"""The model-free E. coli -> S. aureus baselines, and the pMIC scatter that explains them.

Everything here is computed from the BERT-AmPEP60 splits alone -- no model, no new runs.

  ec_label_copy()  the "EC-1NN" reference: for each S. aureus test row, take its nearest
                   E. coli *training* peptide by sequence identity, take that peptide's
                   E. coli pMIC, and map it onto the S. aureus scale with SA = a*EC + b,
                   fitted on the peptides measured in both organisms. No S. aureus test
                   label is used, and no S. aureus training label either, so the reference
                   is flat in n. 83% of the test peptides have an exact match in the
                   E. coli training pool, which is what makes it a memorisation reference
                   rather than a homology-based prediction.
  panel_a()        E. coli vs S. aureus pMIC on those shared peptides, with that same
                   fitted map drawn and spelled out.

Identity is LCS / max(len) from a global alignment (match 1, mismatch/gap 0).

Library only: its callers are plot_extend_vs_knn_refs.py, plot_ec_sa_overlap.py and
plot_ec_sa_why_transfer.py.
"""
import os
import sys

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    sys.path.insert(0, p)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (after _style sets the Agg backend)

from _style import ANNOT_FS, C, INK2  # noqa: E402
from carp.dataset import read_split  # noqa: E402

def ec_label_copy(near=0.8):
    """The model-free 'copy the nearest E. coli training label' baseline.

    -> (DataFrame of the scored SA test rows, PCC over all of them, slope, intercept).
    For each raw S. aureus test row: take its nearest E. coli *training* peptide by
    sequence identity, take that peptide's EC pMIC, and rescale it with the EC->SA linear
    map fitted on the training peptides measured in both organisms. No S. aureus test
    label is used anywhere, and the result does not depend on how much S. aureus training
    data a model is given -- which is what makes it a flat reference line.

    Costs ~10 s (it aligns all 328 test rows against the 3638 EC-train sequences), so
    callers that only need the number may cache it; it is computed rather than stored
    because figure/ is git-ignored."""
    ec_seq, ec_y = read_split("E. coli", "train")
    sa_seq, sa_y = read_split("S. aureus", "train")
    te_seq, te_y = read_split("S. aureus", "test")
    ec_pool = dedup(ec_seq, ec_y.numpy())
    shared = dedup(sa_seq, sa_y.numpy()).merge(ec_pool, on="seq", suffixes=("_sa", "_ec"))
    slope, intercept = np.polyfit(shared.pmic_ec, shared.pmic_sa, 1)

    nn_label, nn_id = nn_copy(identity_matrix(te_seq, ec_pool.seq.tolist()), ec_pool.pmic.values)
    test = pd.DataFrame({"seq": te_seq, "sa_pmic": te_y.numpy(), "nn_ec_identity": nn_id,
                         "pred_ec_copy": slope * nn_label + intercept})
    test["subset"] = np.where(test.nn_ec_identity >= 1.0, "seen",
                              np.where(test.nn_ec_identity >= near, "near", "unseen"))
    test.attrs["mse"] = float(np.mean((test.sa_pmic - test.pred_ec_copy) ** 2))
    return test, float(pearsonr(test.sa_pmic, test.pred_ec_copy)[0]), float(slope), float(intercept)


def dedup(seqs, pmic):
    """-> DataFrame[seq, pmic] with duplicate sequences averaged."""
    df = pd.DataFrame({"seq": seqs, "pmic": np.asarray(pmic, float)})
    return df.groupby("seq", as_index=False)["pmic"].mean()


def identity_matrix(queries, refs):
    """(len(queries), len(refs)) identity = LCS / max(len), from a global alignment."""
    al = PairwiseAligner(mode="global", match_score=1, mismatch_score=0, gap_score=0)
    ref_len = np.array([len(r) for r in refs], dtype=float)
    return np.stack([np.asarray([al.score(q, r) for r in refs]) / np.maximum(ref_len, len(q))
                     for q in queries])


def nn_copy(sim, ref_values, cols=None):
    """Nearest-neighbour label copy: -> (values of the best ref per query, its identity).

    ``cols`` restricts the reference pool to those column indices (used to give each
    train_extend run exactly the training rows it saw)."""
    s = sim if cols is None else sim[:, cols]
    vals = ref_values if cols is None else np.asarray(ref_values)[cols]
    j = s.argmax(1)
    return vals[j], s[np.arange(len(j)), j]


def boot_ci(y, yhat, n_boot, seed=0, q=(2.5, 97.5)):
    """Percentile bootstrap CI for the PCC (resampling peptides)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    stats = np.array([pearsonr(y[i], yhat[i])[0] for i in idx])
    return tuple(np.percentile(stats, q))


def panel_a(ax, shared, slope, intercept, square=False):
    """EC vs SA pMIC on peptides measured in both organisms, one point per row of
    ``shared`` (the caller decides whether that is a peptide or a measurement).

    ``square=True`` gives both axes the same limits and an equal aspect, so the panel is
    symmetric about the 45-degree line and the fitted slope can be read against it."""
    r = pearsonr(shared.pmic_ec, shared.pmic_sa)[0]
    ax.scatter(shared.pmic_ec, shared.pmic_sa, s=3, lw=0, alpha=0.3, color=C[0], zorder=2)
    xs = np.array([shared.pmic_ec.min(), shared.pmic_ec.max()])
    ax.plot(xs, slope * xs + intercept, color=INK2, lw=1.2, zorder=3)
    ax.set_xlabel(r"$\it{E.}$ $\it{coli}$ pMIC")
    ax.set_ylabel(r"$\it{S.}$ $\it{aureus}$ pMIC")
    if square:
        lo = min(shared.pmic_ec.min(), shared.pmic_sa.min())
        hi = max(shared.pmic_ec.max(), shared.pmic_sa.max())
        pad = 0.04 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        # identical ticks too, or the two axes still look unmatched
        ticks = [t for t in matplotlib.ticker.MaxNLocator(nbins=6).tick_values(*lim)
                 if lim[0] <= t <= lim[1]]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    # the fitted map, spelled out: it is the one EC-1NN applies and the one the "shared"
    # ridge probe targets, so readers need its coefficients, not just its slope on paper
    eq = (f"SA = {slope:.3f}\u00b7EC {'\u2212' if intercept < 0 else '+'} {abs(intercept):.3f}")
    ax.annotate(f"$r$ = {r:.2f}\n$n$ = {len(shared):,} peptides\n{eq}", xy=(0.04, 0.96),
                xycoords="axes fraction", va="top", ha="left", color=INK2,
                fontsize=plt.rcParams["xtick.labelsize"] * ANNOT_FS)
    return {"panel": "a", "group": "shared_train_peptides", "n": len(shared), "pcc": r,
            "slope": slope, "intercept": intercept}
