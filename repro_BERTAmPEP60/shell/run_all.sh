#!/usr/bin/env bash
# ProtBERT (BERT-AmPEP60) pMIC regression baseline under CARP's protocol.
# 2 organisms x 5 seeds, fixed test set, best-val selection. Run from anywhere.
#   bash repro_BERTAmPEP60/shell/run_all.sh
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"   # repo root

SEEDS=(0 42 123 666 2026)
ORGANISMS=("E. coli" "S. aureus")
RESULTS_DIR="results/05-repro-bertampep60"

# Offline HF cache (uncomment if the server has no internet AND prot_bert is cached):
# export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

for ORG in "${ORGANISMS[@]}"; do
    CODE=$([ "$ORG" = "E. coli" ] && echo EC || echo SA)
    for SEED in "${SEEDS[@]}"; do
        OUT="protbert_${CODE}_seed${SEED}.json"
        echo "######## ProtBERT baseline  organism=${ORG}  seed=${SEED} ########"
        if [ -f "${RESULTS_DIR}/${OUT}" ]; then
            echo "-- skip (exists: ${RESULTS_DIR}/${OUT})"
            continue
        fi
        python3 repro_BERTAmPEP60/train_protbert_reg.py \
            --organism "$ORG" \
            --seed "$SEED" \
            --out "$OUT"
    done
done

echo "done. Aggregate with: python3 repro_BERTAmPEP60/aggregate.py"
