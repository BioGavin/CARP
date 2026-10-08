#!/usr/bin/env bash
# Why does an E. coli backbone transfer to S. aureus? — the model-based inputs (150M).
#
#   bash scripts/shell/ec_sa_transfer.sh
#
# Runs scripts/probe_ec_sa_transfer.py on the raw ESM-2 backbone (once) and on the joint
# (λ=0.25) E. coli backbone for each seed: zero-shot EC head on SA test, plus the ridge
# probes (SA / shared / residual / EC) that figure/ec_sa_why_transfer.pdf is built from.
# Costs minutes. Re-uses whatever is already in results/, so a second run only fills gaps.
#   -> results/ec-sa-transfer/probe/
#
# The data-efficiency figure reads results/04-extend/ instead, which the main pipeline's
# own sweep produces; nothing here needs to re-run train_extend.
#
# PY overrides the interpreter (the server's torch lives in a conda env, not /usr/bin).
# If the host cannot reach huggingface.co, use the local caches offline:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PY=/path/to/python bash scripts/shell/ec_sa_transfer.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

PY="${PY:-python3}"
SEEDS=(0 42 123 666 2026)
ESM="facebook/esm2_t30_150M_UR50D"
BASE_DIR="checkpoints/03-joint-classreg"
PROBE_DIR="$PWD/results/ec-sa-transfer/probe"
mkdir -p "$PROBE_DIR"

base_ckpt() { echo "${BASE_DIR}/joint_esm2_t30_150M_UR50D_Ecoli_length_matched_r7_lam0.25_seed$1_reg_bestval.pt"; }

[ -f "$PROBE_DIR/probe_raw.json" ] || \
    "$PY" scripts/probe_ec_sa_transfer.py --raw-base --esm-name "$ESM" --tag raw
for SEED in "${SEEDS[@]}"; do
    TAG="joint_seed${SEED}"
    [ -f "$PROBE_DIR/probe_${TAG}.json" ] && { echo "-- skip probe $TAG"; continue; }
    "$PY" scripts/probe_ec_sa_transfer.py --base-ckpt "$(base_ckpt "$SEED")" --esm-name "$ESM" --tag "$TAG"
done
echo "done. results in ${PROBE_DIR}/"
