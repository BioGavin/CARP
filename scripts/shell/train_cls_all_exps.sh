#!/usr/bin/env bash
# training classification head on regression-fine-tuned ESM2 backbone

SEEDS=(0 42 123 666 2026)

declare -A MODELS=(
    [150M]="facebook/esm2_t30_150M_UR50D"
    [650M]="facebook/esm2_t33_650M_UR50D"
)
MODEL_ORDER=(150M 650M)

for SEED in "${SEEDS[@]}"; do
    # 每个 seed 先生成一次负样本数据
    echo "######## build data  seed=${SEED} ########"
    python3 scripts/build_classification_data.py --strategy length_matched --neg-pos-ratio 7 --seed ${SEED};  # for training
    for s in random length_matched composition_matched dual; do
        python3 scripts/build_classification_data.py --strategy "$s" --neg-pos-ratio 1 --seed 0;  # for testing
    done

    for MODEL_KEY in "${MODEL_ORDER[@]}"; do
        ESM_NAME="${MODELS[$MODEL_KEY]}"
        CKPT_NAME="${ESM_NAME##*/}"   # 去掉 facebook/ 前缀 -> esm2_t30_150M_UR50D
        RUN_NAME=""
        echo "========FT  ${MODEL_KEY}  seed=${SEED} ========"
        python3 scripts/train_cls.py \
            --init-from "checkpoints/01-esm-regressor/${CKPT_NAME}_finetune_E.coli_seed${SEED}_bestval.pt" \
            --esm-name "${ESM_NAME}" \
            --neg-pos-ratio 7 \
            --test-neg-pos-ratio 1 \
            --eval-test-strategies "random,length_matched,composition_matched,dual" \
            --epochs 30 \
            --seed "${SEED}" \
            --out-json "finetuned_backbone_${MODEL_KEY}_Ecoli_r7_seed${SEED}.json"

        # native
        echo "========RAW  ${MODEL_KEY}  seed=${SEED} ========"
        python3 scripts/train_cls.py \
            --esm-name "${ESM_NAME}" \
            --neg-pos-ratio 7 --test-neg-pos-ratio 1 \
            --eval-test-strategies "random,length_matched,composition_matched,dual" \
            --epochs 30 --seed "${SEED}" \
            --out-json "native_backbone_${MODEL_KEY}_Ecoli_r7_seed${SEED}.json"
        
    done
done

