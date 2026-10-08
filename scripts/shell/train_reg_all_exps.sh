SEEDS=(0 42 123 666 2026)

declare -A ESM_MODELS=(
  [8M]="facebook/esm2_t6_8M_UR50D"
  [35M]="facebook/esm2_t12_35M_UR50D"
  [150M]="facebook/esm2_t30_150M_UR50D"
  [650M]="facebook/esm2_t33_650M_UR50D"
)
MODEL_ORDER=(8M 35M 150M 650M)
MODES=(${MODES:-frozen finetune})

run_count=0
total=$(( ${#MODEL_ORDER[@]} * ${#MODES[@]} * ${#SEEDS[@]} ))

for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    for model_key in "${MODEL_ORDER[@]}"; do
      esm_name="${ESM_MODELS[$model_key]}"
      run_count=$((run_count + 1))
      run_name="01-esm-${model_key}-${mode}-regression-seed${seed}"

      echo "[${run_count}/${total}]: ${run_name}"

      python3 scripts/train_reg.py \
          --mode "${mode}" \
          --organism "E. coli" \
          --esm-name "${esm_name}" \
          --seed "${seed}"
          # --wandb \
          # --wandb-name "${run_name}"

    done
  done
done

echo "Done!"