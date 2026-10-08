#!/usr/bin/env bash
# Regression-only control (λ=1) for the scale ablation: same recipe as the λ=0.25 runs
# (epochs 50, max_len 60, r7 train / r1 test). Usage: bash scripts/shell/train_joint_lam1_scale.sh <8M|35M|650M> [seeds...]
set -u
KEY=${1:?model key: 8M | 35M | 650M}
shift
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 42 123 666 2026)

declare -A ESM_MODELS=(
  [8M]="facebook/esm2_t6_8M_UR50D"
  [35M]="facebook/esm2_t12_35M_UR50D"
  [650M]="facebook/esm2_t33_650M_UR50D"
)
ESM="${ESM_MODELS[$KEY]}"

for SEED in "${SEEDS[@]}"; do
    OUT="joint_${KEY}_Ecoli_lam1.0_seed${SEED}.json"
    if [ -f "results/03-joint-classreg/${OUT}" ]; then
        echo "-- skip ${OUT} (exists)"
        continue
    fi
    echo "######## ${KEY} λ=1.0 seed=${SEED} ########"
    python3 scripts/train_joint.py \
        --esm-name "$ESM" \
        --mse-weight 1.0 \
        --neg-pos-ratio 7 \
        --test-neg-pos-ratio 1 \
        --epochs 50 \
        --max-len 60 \
        --seed "$SEED" \
        --out-json "$OUT"
done
