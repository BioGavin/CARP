# CARP

**<u>C</u>lassification <u>A</u>nd organism-specific <u>R</u>egression for antimicrobial <u>P</u>eptides**



## Experimental Reproduction

### 1. ESM2 frozen *vs.* fine-tune for regression

```bash
# export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

# frozen example
python3 scripts/train_reg.py --mode frozen --organism "E. coli" --esm-name "facebook/esm2_t30_150M_UR50D" --seed 0 --wandb --wandb-name "esm-150M-frozen-regression-seed0"
# finetune example
python3 scripts/train_reg.py --mode finetune --organism "E. coli" --esm-name "facebook/esm2_t30_150M_UR50D" --seed 0 --wandb --wandb-name "esm-150M-finetune-regression-seed0"
# run all experiments
bash scripts/shell/train_reg_all_exps.sh
```

### 2. Fine-tuned ESM2 for classification 

Build negative dataset

```bash
python3 scripts/build_classification_data.py --organism "E. coli" --strategy length_matched --neg-pos-ratio 9 --seed 0
```

Sweep: classification performance across negative-sampling strategies and negative-to-positive ratios

```bash
# example
python3 scripts/train_cls.py --neg-pos-ratio 1 --test-neg-pos-ratio 1 --eval-test-strategies "random,length_matched,composition_matched,dual" --strategy dual --init-from checkpoints/01-esm-regressor/esm2_t30_150M_UR50D_finetune_E.coli_seed0_bestval.pt --esm-name "facebook/esm2_t30_150M_UR50D" --seed 0 --no-save-ckpt --wandb --wandb-name "esm-cls-dual-r1-test4-seed0"

# run all experiments
bash scripts/shell/sweep_ratio_strat.sh
```

Classification performance using native vs. regression-fine-tuned ESM-2 embeddings under the default length_match negative-sampling strategy (7:1 negative-to-positive ratio for training, 1:1 for testing)

```bash
# native
python3 scripts/train_cls.py --esm-name facebook/esm2_t30_150M_UR50D --neg-pos-ratio 7 --test-neg-pos-ratio 1 --eval-test-strategies "random,length_matched,composition_matched,dual" --epochs 30 --seed 0 --no-save-ckpt --out-json "native_backbone_Ecoli_r7_seed0.json" --wandb --wandb-name "native_backbone_Ecoli_r7_seed0"
# fine-tuned
python3 scripts/train_cls.py --init-from checkpoints/01-esm-regressor/esm2_t30_150M_UR50D_finetune_E.coli_seed0_bestval.pt --esm-name facebook/esm2_t30_150M_UR50D --neg-pos-ratio 7 --test-neg-pos-ratio 1 --eval-test-strategies "random,length_matched,composition_matched,dual" --epochs 30 --seed 0 --no-save-ckpt --out-json "finetuned_backbone_Ecoli_r7_seed0.json" --wandb --wandb-name "finetuned_backbone_Ecoli_r7_seed0"
```

### 3. Joint fine-tuning ESM2 for classification & regression

```bash
# init from native
python3 scripts/train_joint.py --esm-name facebook/esm2_t30_150M_UR50D --mse-weight 0.25 --neg-pos-ratio 7 --test-neg-pos-ratio 1 --epochs 50 --seed 0 --wandb --wandb-name joint_finetune_length_matched_trainR7_testR1_lam0.25_seed0

# run all joint training: 
bash scripts/shell/train_joint_all_exps.sh
```

### 4. Add new pathogen MIC regression head

```bash
# single run — add an S. aureus head on the frozen joint (λ=0.25) E. coli backbone
python3 scripts/train_extend.py --base-ckpt checkpoints/03-joint-classreg/joint_esm2_t30_150M_UR50D_Ecoli_length_matched_r7_lam0.25_seed0_reg_bestval.pt --base-organism "E. coli" --new-organism "S. aureus" --esm-name facebook/esm2_t30_150M_UR50D --seed 0 --out extend_Saureus_lam0.25_reg_bestval_full_seed0.json

# single run (control arm) — add an S. aureus head on the frozen raw pretrained ESM-2 backbone
python3 scripts/train_extend.py --raw-base --new-organism "S. aureus" --esm-name facebook/esm2_t30_150M_UR50D --seed 0 --out extend_Saureus_rawbase_full_seed0.json

# run the full joint-base sweep (9 n_sub × 5 seeds):
bash scripts/shell/train_extend_all_exps.sh

# run the full raw-base control sweep (frozen raw ESM-2; 9 n_sub × 5 seeds):
bash scripts/shell/train_extend_rawbase_all_exps.sh
```



## Final Model

The final CARP model is the 150M-parameter ESM-2 backbone trained jointly on classification and regression (λ = 0.25, seven negatives per positive), with the *S. aureus* regression head added on the frozen backbone. The commands below train it for the five standard seeds.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Stage 0 — build negative dataset
# r7 -> cls_Ecoli_length_matched_r7_{train,test}.csv
python3 scripts/build_classification_data.py --organism "E. coli" --strategy length_matched --neg-pos-ratio 7 --seed 0

# r1 -> cls_Ecoli_length_matched_r1_{train,test}.csv
python3 scripts/build_classification_data.py --organism "E. coli" --strategy length_matched --neg-pos-ratio 1 --seed 0

# Stage 1 — 150M joint E. coli base (5 seeds)
ESM="facebook/esm2_t30_150M_UR50D"

for SEED in 0 42 123 666 2026; do
  python3 scripts/train_joint.py \
    --esm-name "$ESM" \
    --organism "E. coli" \
    --mse-weight 0.25 \
    --strategy length_matched --neg-pos-ratio 7 --test-neg-pos-ratio 1 \
    --epochs 50 \
    --seed "$SEED" \
    --out-json "joint_150M_Ecoli_lam0.25_seed${SEED}.json"
done

# Stage 2 — freeze base, add S. aureus regression head (5 seeds, full SA)
ESM="facebook/esm2_t30_150M_UR50D"
BASE_DIR="checkpoints/03-joint-classreg"

for SEED in 0 42 123 666 2026; do
  BASE="${BASE_DIR}/joint_esm2_t30_150M_UR50D_Ecoli_length_matched_r7_lam0.25_seed${SEED}_reg_bestval.pt"
  python3 scripts/train_extend.py \
    --base-ckpt "$BASE" \
    --base-organism "E. coli" \
    --new-organism "S. aureus" \
    --esm-name "$ESM" \
    --n-sub 0 \
    --epochs 60 \
    --seed "$SEED" \
    --out "extend_Saureus_lam0.25_reg_bestval_full_seed${SEED}.json"
done
```





## How to use
