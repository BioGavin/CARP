
SEEDS=(0 42 123 666 2026)

declare -A ESM_MODELS=(
  [150M]="facebook/esm2_t30_150M_UR50D"
  [650M]="facebook/esm2_t33_650M_UR50D"
)
MODEL_ORDER=(150M 650M)

# ESM="facebook/esm2_t30_150M_UR50D"
NEG_POS_RATIO=7            # 训练 r7
TEST_NEG_POS_RATIO=1       # 测试 1:1
EPOCHS=200

LAMBDAS=(0.0 0.25 0.5 0.75 1.0)

for LAM in "${LAMBDAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for MODEL_KEY in "${MODEL_ORDER[@]}"; do
            ESM="${ESM_MODELS[$MODEL_KEY]}"
            echo "######## train joint model  seed=${SEED}  λ=${LAM}  model=${MODEL_KEY} ########"

            if compgen -G "results/03-joint-classreg/*Weight${LAM}_*${MODEL_KEY}*_r${NEG_POS_RATIO}_seed${SEED}.json" > /dev/null; then
                echo "-- skip λ=$LAM (result JSON 已存在)"
                continue
            fi

            # NAME="joint_finetune_trainR${NEG_POS_RATIO}_testR${TEST_NEG_POS_RATIO}_lam${LAM}_seed${SEED}"

            python3 scripts/train_joint.py \
                --esm-name "$ESM" \
                --mse-weight "$LAM" \
                --neg-pos-ratio "$NEG_POS_RATIO" \
                --test-neg-pos-ratio "$TEST_NEG_POS_RATIO" \
                --epochs "$EPOCHS" \
                --seed "$SEED"
                # --wandb --wandb-name "$NAME"
        done
    done
done
