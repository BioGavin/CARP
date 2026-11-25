# Joint Training Rerun Results (max_len=64)

Archived 40 rerun JSON files on 2026-11-25. Source copies were verified using SHA-256 before sanitization. Server-specific model paths were replaced with public model identifiers, and checkpoint paths with project-relative paths; training settings and metrics are unchanged. Original results remain in `../03-joint-classreg/`.

- **150M:** λ = 0, 0.25, 0.5, 0.75, and 1; five seeds each (25 runs).
- **8M, 35M, and 650M:** λ = 0.25; five seeds each (15 runs).
- **Seeds:** 0, 42, 123, 666, and 2026; 50 epochs per run.
- **Launch scripts** (relative to the project root): `scripts/shell/train_joint_150M_sweep_lamda.sh` and `scripts/shell/train_joint_lam0.25_sweep_4scales.sh`.

## Comparison with Original Results

The main training parameter change recorded in the JSON files is `max_len: 60 → 64`; other training hyperparameters match. Model paths and output naming also differ. Values below are means across five seeds, using test metrics from the checkpoint selected by regression validation MSE (`result.reg.all_test_at_best_val`).

| Model (λ=0.25) | Original MSE | Rerun MSE | Original PCC | Rerun PCC | Original AUROC | Rerun AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 8M | 0.2937 | 0.2935 | 0.7819 | 0.7829 | 0.9660 | 0.9672 |
| 35M | 0.2912 | 0.2973 | 0.7825 | 0.7788 | 0.9679 | 0.9692 |
| 150M | 0.2899 | 0.2921 | 0.7843 | 0.7818 | 0.9716 | 0.9700 |
| 650M | 0.2914 | 0.2865 | 0.7844 | 0.7877 | 0.9717 | 0.9709 |

For the 150M reruns, mean MSE at λ = 0.25, 0.5, 0.75, and 1 is 0.2921, 0.2922, 0.3024, and 0.3008, respectively; mean AUROC at λ = 0 is 0.9678. Performance at λ = 0.25 and 0.5 is similar, while 650M shows improved mean regression metrics. Overall changes are small; statistical significance and performance equivalence have not been established, and differences cannot be attributed solely to max_len.

This directory archives rerun results only. Manuscript figures and plotting outputs have not been updated.
