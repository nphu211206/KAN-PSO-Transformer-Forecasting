# Kaggle 09: Uniform Canonical Diagnostics

Bundle: `KAGGLE_09_UNIFORM_CANONICAL_DIAGNOSTICS.py`  
SHA-256: `59D42770B33D95BCEDD5B3DC2F950B9BB183D9D1A5400C942E0A6EBCDFE02F3C`  
Size: 99679 bytes  
Campaign ID: `canonical_uniform_best_diagnostics_v1`

## Scientific purpose

K09 addresses the largest limitation in K08: checkpoint selection was not
uniform across conditions. K09 uses minimum validation MSE for every trainable
model and saves predictions/checkpoints needed for residual and compute
diagnostics.

## Planned matrix

- ETTh1 and ETTh2.
- Canonical `ett_canonical_12_4_4_v1` split.
- Horizon 96 and look-back 96.
- Seeds 1000-1004.
- DLinear, corrected KAN, no-update KAN and moderate-update KAN.
- 30 epochs.
- 40 tasks total.

## Kaggle execution

1. Create a new Kaggle notebook with GPU enabled.
2. Paste the complete contents of the bundle into one code cell.
3. Run the cell once. Do not mix it with K01-K08 notebook cells.
4. Confirm the log reports Tesla T4 and 40 planned tasks.
5. Download `canonical_uniform_best_diagnostics.zip` using the generated blue
   download button before ending the notebook session.
6. Do not rename, modify or unpack/repack the downloaded archive before audit.

## Expected archive contents

- `records/`: 40 JSON task records.
- `predictions/`: NPZ files with standardized and recorded-OT arrays.
- `checkpoints/`: validation-selected model state files.
- `summary.csv`: metrics, selected epoch, latency, memory and artifact hashes.

## Acceptance checks after download

- 40/40 records completed or every failure retained.
- Every trainable record reports
  `selection_rule=minimum_validation_mse_checkpoint`.
- Every completed record has prediction and checkpoint SHA-256 values.
- Prediction arrays contain `y_true_standardized`, `y_pred_standardized`,
  `y_true_ot` and `y_pred_ot`.
- Dataset/source/campaign hashes are consistent across the matrix.
- Pilot, K08 and K09 records remain separate.

K09 results must not enter the manuscript until the archive passes a dedicated
audit and the manuscript explicitly distinguishes K08 from K09.
