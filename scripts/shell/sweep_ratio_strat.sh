#!/usr/bin/env bash
# Sweep the TRAIN neg:pos ratio for the frozen-backbone classifier while evaluating
# every model on ONE fixed test set (r=TEST_R), so metrics are comparable across ratios.
#
# Fixed: mode=frozen (only cls head trains), 150M backbone (from --init-from),
# organism=E. coli, strategy=length_matched, single seed. Varied: train neg:pos.
#
# Run from the CARP/ root on the server:
#   bash scripts/shell/sweep_neg_ratio.sh
set -euo pipefail


ORG="E. coli"
STRATS="${STRATS:-random length_matched composition_matched dual}"   # train strategies
SEED="${SEED:-0}"
ESM="facebook/esm2_t30_150M_UR50D"
INIT="checkpoints/01-esm-regressor/esm2_t30_150M_UR50D_finetune_E.coli_seed0_bestval.pt"
RATIOS="${RATIOS:-1 3 5 7 9}"                   # train neg:pos grid
TEST_R="${TEST_R:-1}"                           # fixed test neg:pos
EVAL_STRATS="random,length_matched,composition_matched,dual"          # test on all 4
BUILD_DATA="${BUILD_DATA:-1}"                    # 0 = reuse existing CSVs

BUILD="scripts/build_classification_data.py"
TRAIN="scripts/train_cls.py"

if [ "$BUILD_DATA" = "1" ]; then
  echo "### build fixed test set (r=$TEST_R) + train sets for ratios: $RATIOS"

  for s in $STRATS; do
    python3 "$BUILD" --organism "$ORG" --strategy "$s" --neg-pos-ratio "$TEST_R" --seed "$SEED"
    for r in $RATIOS; do
      if [ "$r" != "$TEST_R" ]; then
        python3 "$BUILD" --organism "$ORG" --strategy "$s" --neg-pos-ratio "$r" --seed "$SEED"
      fi
    done
  done
fi

for s in $STRATS; do
  for r in $RATIOS; do
    echo "############## fine-tuned backbone classifier: train strat=$s r=$r, test r=$TEST_R ##############"
    python3 "$TRAIN" --organism "$ORG" --strategy "$s" \
        --neg-pos-ratio "$r" --test-neg-pos-ratio "$TEST_R" \
        --epochs 50 --eval-test-strategies "$EVAL_STRATS" \
        --init-from "$INIT" --esm-name "$ESM" --seed "$SEED" --no-save-ckpt \
        --out-json "sweep_ft_esm_cls_${s}_r${r}_test${TEST_R}_seed${SEED}.json" \
        --wandb --wandb-name "02-sweep-ft-esm-cls-${s}-r${r}-seed${SEED}"
  done
done

echo "### done! Results in results/02-esm-classifier/sweep_ft_esm_cls_*_seed${SEED}.json"