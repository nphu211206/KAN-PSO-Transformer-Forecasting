# Canonical Uniform-Best Diagnostics Preregistration

Preregistered: 2026-08-16, before execution of `KAGGLE_09`.

## Purpose

The frozen K08 canonical campaign used condition-specific checkpoint rules:
corrected KAN restored the validation-best checkpoint, while DLinear,
no-update KAN and moderate-update KAN were evaluated at the final epoch. K09
tests the same h96 conditions under one uniform validation-only selection rule
and stores the diagnostics missing from the frozen archive.

K09 does not overwrite, repair or replace K08. It creates a new evidence layer
with campaign identifier `canonical_uniform_best_diagnostics_v1`.

## Questions

1. Does DLinear remain stronger than corrected KAN when every trainable
   condition restores its minimum-validation-MSE checkpoint?
2. Does moderate Grid Update remain less reliable than no update under the same
   checkpoint contract?
3. What do stored predictions, residuals, inference latency and peak GPU memory
   reveal about accuracy, failure modes and operational cost?

## Frozen design

- Data protocol: `ett_canonical_12_4_4_v1`.
- Datasets: ETTh1 and ETTh2.
- Target/features: univariate OT (`S`).
- Input length: 96.
- Forecast horizon: 96.
- Seeds: 1000, 1001, 1002, 1003 and 1004.
- Epoch budget: 30.
- Hardware target: Kaggle Tesla T4 when available.
- Conditions:
  - DLinear;
  - corrected KAN (`KAN_WOA_RegBestClip`);
  - no-update KAN (`none_retain`);
  - moderate-update KAN (`moderate_retain`).
- Model selection: minimum validation MSE for every trainable condition.
- Total planned tasks: 40.

The test set must not be used for checkpoint selection, early stopping,
configuration selection or rerun decisions.

## Required saved evidence

Each completed task must contain:

- source, dataset, protocol and campaign identifiers;
- validation/test MSE and MAE;
- selected epoch and validation selection rule;
- seed-level `y_true` and `y_pred` in standardized and recorded OT units;
- model checkpoint and SHA-256 hashes for prediction/checkpoint artifacts;
- end-to-end test-loader inference time and milliseconds per sample;
- peak allocated GPU memory during model training;
- Grid Update event logs where applicable;
- parameter count, environment metadata and elapsed training time.

## Analysis policy

- Report all five seeds, including catastrophic or unfavorable results.
- Report means, sample SDs, paired seed differences and descriptive residual
  diagnostics.
- Do not add a post-hoc significance threshold after observing K09.
- Do not pool K08 and K09 as if they used the same checkpoint contract.
- Compare K09 conditions only within the new campaign.
- Prediction plots must be generated directly from saved NPZ arrays.
- Latency must be described as a single full test-loader pass including loader
  overhead, host-to-device transfer and prediction copy to host, not pure
  kernel latency.

## Claim boundary

Even if KAN wins a condition, K09 cannot support SOTA, universal KAN
superiority, universal Grid Update behavior or cross-domain generalization.
ETTh1 and ETTh2 remain related datasets at a single canonical horizon.

## Stop rule

The campaign is complete only when all 40 records are completed or every
failure is retained with its traceback and task identity. No condition may be
silently removed after results are observed.
