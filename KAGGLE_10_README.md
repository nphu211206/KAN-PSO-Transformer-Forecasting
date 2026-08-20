# Kaggle 10: Canonical Multi-Horizon Baselines

Corrected download-safe bundle: `KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES_V2_DOWNLOAD_SAFE.py`  
SHA-256: `ED3639ABF7310DED1C5EB092004FA64E4419C5D5139846CE92A8FA31092DA44F`  
Size: 103620 bytes  
Campaign ID: `canonical_multihorizon_baselines_v2`

The v1 bundle (`619846...`) is withdrawn because it planned 360 tasks. The
first v2 run completed 120/120, but its ZIP was lost after the wrapper tried to
embed the large archive as base64 and the Kaggle session reset. Stop old
notebooks and do not reuse partial result directories. Both amendments are documented in
`CANONICAL_MULTIHORIZON_BASELINE_PREREGISTRATION_V2.md`.

## Scientific purpose

K10 extends the canonical comparison from h96 to the four standard ETT
horizons. It compares DLinear, corrected KAN and PatchTST under one
validation-best checkpoint contract and retains prediction/compute artifacts.
It is a new campaign and must not be pooled with K08 or K09.

## Frozen matrix

- ETTh1 and ETTh2.
- Canonical `ett_canonical_12_4_4_v1` protocol.
- Lookback 96; horizons 96, 192, 336 and 720.
- Seeds 1000-1004.
- DLinear, corrected KAN (`KAN_WOA_RegBestClip`) and PatchTST.
- 30 epochs and validation-best checkpoint selection for every model.
- 120 tasks total.
- Predictions, checkpoints, latency and peak GPU memory saved per task.

## Kaggle execution

1. Create a new Kaggle notebook and enable a Tesla T4 GPU.
2. Paste the complete bundle into one code cell and run it once.
3. Confirm the log reports exactly `Planned tasks: 120` and campaign ID
   `canonical_multihorizon_baselines_v2` in task records. The v2 runner also
   refuses to start if its matrix is not exactly 120.
4. The first PatchTST task establishes `transformers==4.48.3`; do not change
   the embedded dependency or configuration after execution starts.
5. Download `canonical_multihorizon_baselines_v2.zip` immediately using the
   lightweight `FileLink` or the refreshed Kaggle Files pane. The safe wrapper
   never embeds the ZIP as base64.
6. Do not edit, rename internally, unpack/repack or selectively rerun the
   archive before audit.

## Acceptance command

```powershell
python audit_k10.py canonical_multihorizon_baselines_v2.zip
```

Do not add K10 values to the manuscript until this audit passes all 120
records, artifact hashes, prediction shapes, recomputed metrics and campaign
boundaries.
