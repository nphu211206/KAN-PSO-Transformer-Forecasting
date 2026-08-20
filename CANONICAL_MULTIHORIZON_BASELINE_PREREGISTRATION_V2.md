# Canonical Multi-Horizon Baseline Preregistration Amendment (K10 v2)

Amended: 2026-08-17, immediately after the first execution log exposed a task
enumeration defect and before any K10 campaign was completed or analyzed.

## Reason for amendment

The v1 input correctly declared 120 model-comparison tasks, but an
implementation branch also enumerated 240 unrelated optimizer-state tasks.
Kaggle reported `Planned tasks: 360`. The run was stopped and v1 is abandoned;
no v1 result may be used as K10 evidence.

The defect did not arise from observed test performance and no model,
hyperparameter, dataset, seed, horizon, checkpoint rule or analysis policy was
changed. V2 only isolates the preregistered model-comparison branch from the
optimizer-state branch and uses a fresh campaign/result identity.

## Frozen v2 contract

- Campaign ID: `canonical_multihorizon_baselines_v2`.
- Result directory/archive: `canonical_multihorizon_baselines_v2` and
  `canonical_multihorizon_baselines_v2.zip`.
- Protocol: `ett_canonical_12_4_4_v1`.
- Datasets: ETTh1 and ETTh2.
- Univariate OT, lookback 96.
- Horizons: 96, 192, 336 and 720.
- Seeds: 1000-1004.
- Models: DLinear, PatchTST and corrected KAN
  (`KAN_WOA_RegBestClip`).
- Epochs: 30.
- Uniform validation-best checkpoints and saved prediction/checkpoint/compute
  diagnostics.
- Exact total: 120 tasks.

All analysis and claim boundaries in the original preregistration remain in
force. The v1 partial run is an implementation-validation artifact, not a
scientific campaign or source of reusable results.

## Non-scientific download-wrapper amendment

The first complete v2 interactive run reported 120/120 tasks and zero failures,
but its archive was lost when the wrapper attempted to embed the large ZIP as
a base64 notebook output and the Kaggle session was subsequently reset. No
archive or task record was retrieved for analysis. The rerun wrapper removes
base64 transfer, computes SHA-256 by streaming and exposes a lightweight
`FileLink`. The embedded training sources, campaign ID, task matrix and all
scientific settings are unchanged.
