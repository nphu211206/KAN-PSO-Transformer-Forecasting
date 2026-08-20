# Canonical ETT Confirmation Preregistration

**Version:** 1.0
**Status:** Executed; one implementation deviation documented below; audited 2026-08-15
**Purpose:** Validate whether the frozen pilot-protocol conclusions survive the
official ETT 12/4/4-month split. This is a confirmation study, not a new HPO
search.

## Research question

Under the canonical ETT protocol, do the following conclusions remain true?

1. DLinear is at least as strong as the corrected direct KAN configuration.
2. Corrected KAN training is more stable than the historical final-epoch,
   no-regularization procedure.
3. Repeated Grid Updates do not provide a reliable improvement over no update.

## Fixed design

- Datasets: `ETTh1`, `ETTh2`.
- Target: univariate `OT` (`features=S`).
- Horizons: `96`.
- Look-back: `96`.
- Protocol: `ett_canonical_12_4_4_v1`, using official hourly ETT boundaries:
  train rows `[0, 8640)`, validation `[8640, 11520)`, test `[11520, 14400)`.
- Validation/test windows overlap the preceding history by `seq_len`, matching
  the official Autoformer/Time-Series-Library loaders.
- Scaler: `StandardScaler` fit only on training rows.
- Seeds: `1000, 1001, 1002, 1003, 1004`.
- Training epochs: `30`.
- Intended selection rule: minimum validation MSE; test data is never used for selection.
- Metrics: MSE and MAE in standardized space, reported as mean and sample SD.

## Conditions

The confirmation runner executes:

- DLinear baseline.
- Corrected direct KAN-WOA with regularization, validation-best checkpoint and
  gradient clipping.
- Fixed KAN with no Grid Update (`none_retain`).
- Fixed KAN with the moderate Grid Update schedule and retained Adam state
  (`moderate_retain`).

The configuration, implementation patch and source files are inherited from
the frozen K06/K07 source manifest. No hyperparameter is selected using the
canonical test set. No additional variant will be added after seeing results.

## Execution deviation

The executed runner did not apply the intended minimum-validation-MSE rule
uniformly. Corrected KAN restored its validation-best checkpoint, while
DLinear, no-update KAN and moderate-update KAN retained the final epoch to
match their historical evaluation contracts. The JSON records expose both
`selection_rule` and `selected_epoch`; this deviation is retained in the
archive and is disclosed in the manuscript. Therefore the campaign is treated
as descriptive multi-seed evidence under condition-specific training
contracts, not as a strict same-checkpoint confirmatory comparison. No result
was removed or rerun after inspecting the outcomes.

## Analysis plan

- Report each dataset/condition with all five seed values, mean, sample SD,
  median and range.
- Compare conditions descriptively; do not claim significance from five runs.
- Use paired seed differences only as exploratory evidence.
- Do not pool ETTh1 and ETTh2 into one performance claim.
- If canonical results disagree with pilot results, report the disagreement as
  protocol sensitivity rather than selecting the favorable protocol.

## Stopping rule

This campaign is complete after all planned tasks finish or a documented
infrastructure failure prevents completion. No test-driven rerun, architecture
search or post-hoc HPO is permitted.

## Reproducibility

The result archive, source combined SHA-256, dataset SHA-256, Python/PyTorch/
CUDA/GPU metadata and complete JSON records are primary evidence. Results will
be stored under `canonical_confirmation_results/` and must not be merged into
the frozen registry until independently audited.

## Execution record

The campaign completed 40/40 tasks using the registered design. The audited
archive SHA-256 is
`7F0A3DA091857B0CAF8BD72DAD1850EA2A3A1A005079B05D639D540FA48DA378`.
Results and interpretation are recorded in
`analysis/canonical_confirmation/CANONICAL_CONFIRMATION_AUDIT.md`.
