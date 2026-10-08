#!/usr/bin/env bash
# Experiment 4: extend the joint (λ=0.25) E. coli model with an S. aureus regression head,
# sweeping how much SA training data is available (--n-sub) across 5 paired seeds.
#
# Run from anywhere (the script cd's to the CARP repo root):
#   bash scripts/shell/train_extend_all_exps.sh
# Other backbone sizes / a subset of n_sub (the joint base ckpt must exist for that size):
#   ESM=facebook/esm2_t6_8M_UR50D N_SUBS="0 1000 200 50" bash scripts/shell/train_extend_all_exps.sh
#   BASE_DIR=checkpoints/03-joint-classreg ESM=facebook/esm2_t33_650M_UR50D \
#       bash scripts/shell/train_extend_all_exps.sh
#
# Each run freezes the joint base's ESM-2 backbone, adds a fresh SA regression head, and
# trains ONLY that head. The base seed and the extend --seed are PAIRED, so the 5 seeds are
# independent repeats: the SA train/val split and the --n-sub subsample both vary with --seed.
# Because the base is a joint (cls-carrying) checkpoint, each run also reports the E. coli
# classification-forgetting probe (should be ΔAUC≈0), which reads
#   dataset/classification/cls_Ecoli_length_matched_r1_test.csv
# — make sure that file exists on this machine (build it with build_classification_data.py if not).
set -euo pipefail

# cd to repo root so relative paths (results/, scripts/) resolve wherever this is launched.
cd "$(cd "$(dirname "$0")/../.." && pwd)"

# --- config ------------------------------------------------------------------
SEEDS=(0 42 123 666 2026)                        # base seed == extend --seed (paired)
read -r -a N_SUBS <<< "${N_SUBS:-0 2500 2000 1500 1000 500 200 100 50}"  # 0 = full SA train set

ESM="${ESM:-facebook/esm2_t30_150M_UR50D}"       # MUST match the joint base
ESM_ID="${ESM##*/}"                               # e.g. esm2_t30_150M_UR50D
SIZE="$(echo "$ESM_ID" | cut -d_ -f3)"            # e.g. 150M
# 150M keeps its original (untagged) result names so existing runs are skipped
if [ "$SIZE" = "150M" ]; then SIZE_TAG=""; else SIZE_TAG="${SIZE}_"; fi
LAM=0.25                                          # best joint config (paper Appendix C)
VARIANT="reg_bestval"                             # which joint ckpt to load as the base.
                                                  #   reg_bestval = backbone at best REGRESSION epoch (recommended
                                                  #     for a regression extension, matches the paper's reporting).
                                                  #   Your example used cls_bestval — set VARIANT="cls_bestval" for that.
EPOCHS=60                                         # SA-head training epochs (train_extend default)

BASE_DIR="${BASE_DIR:-checkpoints/03-joint-classreg}"
RESULTS_DIR="results/04-extend"

# Offline HF cache (uncomment if the server has no internet):
# export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

# --- sweep -------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    BASE_CKPT="${BASE_DIR}/joint_${ESM_ID}_Ecoli_length_matched_r7_lam${LAM}_seed${SEED}_${VARIANT}.pt"
    if [ ! -f "$BASE_CKPT" ]; then
        echo "!! base ckpt missing, skipping seed=${SEED}: $BASE_CKPT"
        continue
    fi

    for NSUB in "${N_SUBS[@]}"; do
        if [ "$NSUB" -le 0 ]; then NTAG="full"; else NTAG="n${NSUB}"; fi
        OUT="extend_Saureus_${SIZE_TAG}lam${LAM}_${VARIANT}_${NTAG}_seed${SEED}.json"

        echo "######## extend  ${ESM_ID}  seed=${SEED}  n_sub=${NTAG}  base=${VARIANT}  λ=${LAM} ########"
        if [ -f "${RESULTS_DIR}/${OUT}" ]; then
            echo "-- skip (result JSON 已存在: ${RESULTS_DIR}/${OUT})"
            continue
        fi

        python3 scripts/train_extend.py \
            --base-ckpt "$BASE_CKPT" \
            --base-organism "E. coli" \
            --new-organism "S. aureus" \
            --esm-name "$ESM" \
            --n-sub "$NSUB" \
            --epochs "$EPOCHS" \
            --seed "$SEED" \
            --no-save-ckpt \
            --out "$OUT"
    done
done

echo "done. results in ${RESULTS_DIR}/"
