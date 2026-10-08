#!/usr/bin/env bash
# Experiment 4 (control arm): extend a *raw* pretrained ESM-2 backbone with an
# S. aureus regression head, sweeping how much SA training data is available
# (--n-sub) across 5 seeds. Same fit_frozen recipe as the joint arm; the only
# difference is the backbone source (raw ESM-2 vs a joint lam=0.25 checkpoint),
# so results drop straight into results/04-extend/ next to the joint arm.
#
# Run from anywhere (the script cd's to the CARP repo root):
#   bash scripts/shell/train_extend_rawbase_all_exps.sh
# Other backbone sizes / a subset of n_sub:
#   ESM=facebook/esm2_t6_8M_UR50D N_SUBS="0 1000 200 50" bash scripts/shell/train_extend_rawbase_all_exps.sh
#
# Raw base has no base organism and no classification head, so there is no
# forgetting probe (result JSON: "forgetting": null). Backbone weights are
# identical across seeds; only the SA subsample + val split + head init vary.
set -euo pipefail

# cd to repo root so relative paths (results/, scripts/) resolve wherever this is launched.
cd "$(cd "$(dirname "$0")/../.." && pwd)"

# --- config ------------------------------------------------------------------
SEEDS=(0 42 123 666 2026)                        # subsample + val split + head init
read -r -a N_SUBS <<< "${N_SUBS:-0 2500 2000 1500 1000 500 200 100 50}"  # 0 = full SA train set

ESM="${ESM:-facebook/esm2_t30_150M_UR50D}"       # MUST match the joint arm it is compared to
ESM_ID="${ESM##*/}"                               # e.g. esm2_t30_150M_UR50D
SIZE="$(echo "$ESM_ID" | cut -d_ -f3)"            # e.g. 150M
# 150M keeps its original (untagged) result names so existing runs are skipped
if [ "$SIZE" = "150M" ]; then SIZE_TAG=""; else SIZE_TAG="${SIZE}_"; fi
EPOCHS=60                                         # SA-head training epochs (train_extend default)

RESULTS_DIR="results/04-extend"

# Offline HF cache (uncomment if the server has no internet):
# export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

# --- sweep -------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    for NSUB in "${N_SUBS[@]}"; do
        if [ "$NSUB" -le 0 ]; then NTAG="full"; else NTAG="n${NSUB}"; fi
        OUT="extend_Saureus_${SIZE_TAG}rawbase_${NTAG}_seed${SEED}.json"

        echo "######## extend  raw-base  ${ESM_ID}  seed=${SEED}  n_sub=${NTAG} ########"
        if [ -f "${RESULTS_DIR}/${OUT}" ]; then
            echo "-- skip (result JSON 已存在: ${RESULTS_DIR}/${OUT})"
            continue
        fi

        python3 scripts/train_extend.py \
            --raw-base \
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
