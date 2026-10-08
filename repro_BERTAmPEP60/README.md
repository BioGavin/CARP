# repro_BERTAmPEP60

A faithful **BERT-AmPEP60 (ProtBERT)** pMIC regression baseline, evaluated under
**CARP's protocol** so the only difference from CARP's ESM-2 regression results is the
backbone.

- Model: `Rostlab/prot_bert`, full fine-tune, `pooler_output` pooling, head
  `LayerNorm -> 512 -> 128 -> 1` (LeakyReLU, dropout 0.2) — re-implemented in `model.py`.
- Data / metrics: reused from `carp.dataset` / `carp.metrics` (same fixed test set as CARP,
  same 90/10 seed-based train/val split, same `mse/rmse/r2/pcc/ktc/spearman`).
- Selection: best epoch by **validation MSE**; we report `test_at_best_val` (primary) and
  `best_test` (reference == the original leaky reporting).
- Hyperparameters (faithful): AdamW lr 1e-5, weight decay 3e-3, batch 12, epochs 100,
  grad clip 1.0, `MAX_LEN=64` (original paper: 77; numerically equivalent here — longest sequence is 60 aa = 62 tokens, `[CLS]` pooling ignores padding), 5 seeds `{0,42,123,666,2026}`.

## Run (on the GPU server)

```bash
# single run
python3 repro_BERTAmPEP60/train_protbert_reg.py --organism "E. coli" --seed 0

# full sweep (2 organisms x 5 seeds)
bash repro_BERTAmPEP60/shell/run_all.sh

# aggregate -> results/05-repro-bertampep60/protbert_baseline_summary.csv
python3 repro_BERTAmPEP60/aggregate.py
```

Per-run JSONs and the summary CSV are written to `results/05-repro-bertampep60/`.
