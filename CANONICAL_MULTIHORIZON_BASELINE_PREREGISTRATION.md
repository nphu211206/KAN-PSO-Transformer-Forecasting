# Canonical Multi-Horizon Baseline Preregistration (K10)

**Historical status:** abandoned before completion because the v1 runner
enumerated 360 rather than 120 tasks. Do not execute this version or use its
partial outputs. The transparent corrective amendment is
`CANONICAL_MULTIHORIZON_BASELINE_PREREGISTRATION_V2.md`.

Preregistered: 2026-08-17, before execution of `KAGGLE_10`.

## Purpose

K09 establishes the h96 reliability and Grid Update evidence under one
validation-best checkpoint contract. K10 extends the model-comparison question
only: does the corrected KAN remain competitive with a linear engineering
baseline and a neural long-horizon baseline across the standard ETT horizons?

K10 is an independent, versioned campaign with identifier
`canonical_multihorizon_baselines_v1`. It neither overwrites nor pools with
K08/K09. Its results must be summarized separately before any descriptive
cross-campaign discussion.

## Frozen design

- Protocol: `ett_canonical_12_4_4_v1`.
- Datasets: ETTh1 and ETTh2.
- Target/features: univariate OT (`S`).
- Lookback: 96 observations.
- Horizons: 96, 192, 336 and 720 observations.
- Seeds: 1000, 1001, 1002, 1003 and 1004.
- Epoch budget: 30 for every trainable model.
- Hardware target: Kaggle Tesla T4 when available.
- Conditions at every dataset/horizon/seed:
  - DLinear with the frozen kernel-25/default learning-rate configuration;
  - corrected KAN (`KAN_WOA_RegBestClip`), using the validation-selected WOA
    configuration transferred with the same 96-step lookback and no Grid
    Update; and
  - PatchTST with the fixed implementation/configuration embedded in
    `baseline_models.py`.
- Model selection: minimum validation MSE checkpoint for every trainable
  condition, selected without consulting the test split.
- Total planned tasks: `2 datasets x 4 horizons x 3 models x 5 seeds = 120`.

PatchTST is included as an independently implemented neural baseline, not as a
claim that its fixed configuration is globally optimized. No condition may be
tuned, restarted selectively, or removed after observing test performance.

## Required saved evidence

Each record must retain source, protocol and dataset hashes; complete model
configuration; validation and test MSE/MAE; selected epoch and selection rule;
parameter count; elapsed training time; full test-loader inference time;
peak allocated GPU memory; prediction arrays in standardized and OT units; and
the selected checkpoint. Every artifact must be SHA-256-addressed in its JSON
record.

## Analysis policy

- Analyze all five seeds for every condition; do not discard unfavorable runs.
- Compute summary statistics within each dataset/horizon/campaign cell only.
- Derive actual-vs-predicted, residual, per-step error and
  accuracy-latency-memory figures only from retained K10 predictions and
  record fields.
- K09 remains the causal Grid Update mechanism study. K10 must not be used to
  make a new all-horizon Grid Update claim.
- No post-hoc p-value threshold, test-set selection, SOTA claim, universal
  KAN claim or cross-domain generalization claim is permitted.

## Completion and failure policy

K10 is complete only when 120 completed task records exist, or every failed
task is preserved with its task identity and traceback. A failure prevents a
complete-cell claim; it must not be silently retried under an altered setup.
