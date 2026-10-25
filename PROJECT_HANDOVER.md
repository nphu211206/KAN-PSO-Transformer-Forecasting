# KAN-HPO Time-Series Forecasting: Project Handover

**Updated:** 2026-08-17  
**Target journal:** Electrical Engineering (Archiv fur Elektrotechnik) (Electrical Engineering)  
**Current evidence:** 380 frozen experiment records through K10  
**Current paper:** about 7000 words, 25 references, 12 figures, 9 tables  
**Primary task:** univariate oil-temperature (`OT`) forecasting on ETTh1/ETTh2

This is the canonical onboarding manual. Read it before changing code, results,
tables or claims. The controlling next-work document is
`MASTER_UPGRADE_ROADMAP.md`; the numerical source of truth is
`EXPERIMENT_REGISTRY.md` plus the frozen JSON/ZIP archives.

## 1. Sixty-Minute Onboarding Route

Read and run in this order:

1. Read this handover for project history and file ownership.
2. Read `EXPERIMENT_REGISTRY.md` for campaign counts, hashes and frozen results.
3. Read `CLAIM_EVIDENCE_MATRIX.md` before editing any scientific claim.
4. Read `paper_draft.md` to understand the current narrative.
5. Read `MASTER_UPGRADE_ROADMAP.md` for all remaining work and priorities.
6. Read `Electrical Engineering_SUBMISSION_REQUIREMENTS.md` and
   `Electrical Engineering_ACCEPTED_ARTICLE_DEEP_AUDIT.md` for venue expectations.
7. Run the audits listed in Section 14 using a Python environment with NumPy
   and python-docx where required.

Do not begin by rerunning Kaggle. The current manuscript requires no additional
training campaign for its bounded claims.

## 2. What the Project Studies

The project asks whether metaheuristic hyperparameter optimization (HPO) makes
Kolmogorov-Arnold Networks (KANs) reliable enough for transformer-temperature
forecasting, and why KAN spline Grid Updates can destabilize training.

KAN differs from an ordinary MLP by placing learnable univariate functions on
edges. In the efficient-kan implementation, these functions combine a base
activation with B-spline coefficients defined on a grid. Updating the grid can
adapt spline support to the activation distribution, but it also changes the
parameter/function geometry while Adam retains momentum estimates. This creates
the Grid Update Dilemma: adaptation can help representation while disrupting
optimization and checkpoint stability.

The paper's final contribution is therefore not “KAN beats all baselines.” It is
a reproducible engineering reliability study of:

- HPO/final-training consistency;
- checkpoint-selection contracts;
- Grid Update schedule instability;
- Adam-state retain/reset interventions;
- negative residual-learning results;
- cross-dataset and canonical-protocol confirmation;
- prediction-level and compute diagnostics.

## 3. Defensible Conclusions

The following conclusions are supported within the tested scope.

1. Under the pilot ETTh1/h96 protocol, making final KAN training consistent
   with the HPO objective reduces mean MSE from `0.175458` to `0.113224`.
2. Corrected KAN improves materially but remains weaker than DLinear under the
   same pilot and canonical comparisons.
3. Residual KAN variants are rejected by validation and do not improve the
   selected DLinear checkpoint.
4. Repeated Grid Updates can produce high variance and catastrophic outcomes.
5. Clearing Adam state after Grid Update is not a sufficient remedy and can
   worsen instability.
6. ETTh2 confirms the direction that corrected KAN remains behind DLinear and
   that Grid Update behavior is dataset-dependent.
7. Under K09's uniform validation-best checkpoint rule, DLinear beats corrected
   KAN in all ten dataset-seed pairs.
8. DLinear has lower original-unit RMSE at all 96 forecast steps on ETTh1 and
   ETTh2.
9. Corrected KAN has fewer parameters (`9216` versus `18624`) but trains roughly
   7.2-7.6 times slower and infers roughly five times slower on the recorded T4
   environment.

Do not extend these conclusions to universal KAN/DLinear superiority, SOTA,
multivariate forecasting, other engineering domains or every Grid Update rule.

## 4. Current K09 Publication Results

K09 is the primary uniform-checkpoint canonical publication campaign.

| Dataset | Condition | Standardized MSE | OT RMSE | Training time |
|---|---|---:|---:|---:|
| ETTh1 | DLinear | `0.064489 +/- 0.001843` | `2.3302 +/- 0.0332` | `11.88 +/- 1.87 s` |
| ETTh1 | Corrected KAN | `0.164913 +/- 0.014927` | `3.7235 +/- 0.1683` | `85.83 +/- 4.53 s` |
| ETTh1 | No-update KAN | `0.277007 +/- 0.039970` | `4.8188 +/- 0.3636` | `38.71 +/- 0.15 s` |
| ETTh1 | Moderate-update KAN | `0.417187 +/- 0.159815` | `5.8265 +/- 1.2155` | `40.04 +/- 0.38 s` |
| ETTh2 | DLinear | `0.127925 +/- 0.000598` | `4.1435 +/- 0.0097` | `10.96 +/- 0.05 s` |
| ETTh2 | Corrected KAN | `0.195536 +/- 0.028679` | `5.1120 +/- 0.3698` | `83.73 +/- 0.19 s` |
| ETTh2 | No-update KAN | `0.329198 +/- 0.071826` | `6.6150 +/- 0.7259` | `38.74 +/- 0.05 s` |
| ETTh2 | Moderate-update KAN | `0.369897 +/- 0.081077` | `7.0152 +/- 0.7324` | `39.83 +/- 0.17 s` |

K09 archive SHA-256:
`0685DEAE0D453FE2135AB8855FC68CB635D53663F09699F1FA537A014D124004`

K09 source combined SHA-256:
`138ddf1b1feffc4fa03cb75c3e6746b67ace7c401cf397f3503f7d8635aa8a49`

K08 and K09 use the same data-split protocol but different checkpoint contracts.
K08 used condition-specific evaluation rules; K09 uses minimum validation MSE
for every condition. Keep both, use K09 for the uniform publication comparison,
and use K08 only as historical checkpoint-contract sensitivity evidence.

## 5. Project History

### Phase 0: research and feasibility

The project collected literature on KAN, B-splines, metaheuristic optimization,
ETT forecasting, efficient-kan and Electrical Engineering reporting. The original ambition was a
hybrid KAN-PSO paper with strong model-performance claims.

### Phase 1: historical HPO

Six search procedures explored a 14-dimensional KAN space. The campaign created
18 records and useful descriptive search evidence, but only three repetitions
per algorithm and unequal historical budgets. It cannot support a formal
optimizer-ranking claim.

### Phase 2: targeted baseline and Grid Update experiments

ETTh1 baselines and KAN variants were evaluated over horizons 96/192/336/720.
These experiments established DLinear as the strongest same-protocol baseline
and exposed Grid Update instability.

### Phase 3: training-consistency repair

The team discovered that historical final training did not fully match the HPO
objective: regularization and validation-best checkpoint behavior differed.
Stage 1 and Stage 2 experiments restored consistency, tested clipping and tested
residual KAN variants. Corrected direct KAN improved; residual variants failed
validation.

### Phase 4: optimizer-state intervention and ETTh2

K06 paired retain/reset Adam-state conditions around Grid Updates on ETTh1.
K07 repeated the major questions on ETTh2. Resetting Adam state did not solve
the problem and often worsened results.

### Phase 5: canonical confirmation

K08 moved to the official hourly ETT 12/4/4-month split with five seeds. It
confirmed DLinear's advantage but retained condition-specific checkpoint rules.

### Phase 6: uniform-checkpoint diagnostics

K09 was preregistered and executed as 40 tasks: two datasets, four conditions
and five seeds. It applied validation-best selection uniformly and saved JSON,
predictions, checkpoints, latency and peak-memory artifacts. K09 closed the
largest fairness and diagnostics gap.

### Phase 7: Electrical Engineering packaging

Recent accepted Electrical Engineering articles and archived official requirements were audited.
The manuscript was rebuilt around reliability, negative results and engineering
interpretation. It now contains workflow/data figures, baseline and Grid Update
evidence, prediction/residual/forecast-step plots and compute tables.

## 6. Current State and Remaining Difficulties

### Complete

- All K01-K10 campaigns used by the current claim are complete.
- 380 records and all registered archives pass integrity audit.
- ETTh1/ETTh2 dataset identity and temporal protocol tests pass.
- K09 predictions/checkpoints/diagnostic artifacts pass dedicated audit.
- Manuscript claim audit passes with 25 references.
- Submission DOCX structural audit passes with 9 editable tables and 12 figures.
- Sole-author identity, affiliation, ORCID, corresponding email and CRediT roles
  are now supplied and written into the manuscript.

### Not complete

- Stable public repository and archival DOI.
- A clean public release separated from unrelated/private workspace files.
- Live Electrical Engineering portal confirmation immediately before upload.
- Visual page-by-page inspection of a fresh PDF export. The prior user PDF was
  rejected because display equations rendered incorrectly; the DOCX is fixed.
- Portable clean Python environments; legacy `.venv*` directories are
  machine-specific or inaccessible.
- Optional spline-function interpretation, independent external-domain data,
  multivariate evidence, calibrated uncertainty and FLOPs.

These optional scientific extensions are not blockers for the present bounded
paper. The administrative, repository and visual-render items are submission
blockers.

## 7. Source-of-Truth Hierarchy

Use this priority order:

1. Original result ZIP and completed JSON record.
2. Registered archive/source/dataset hash.
3. Campaign `summary.csv` generated from records.
4. `analysis/` CSV and audit report.
5. Generated figure/table.
6. Manuscript prose.

Never edit a lower layer to contradict a higher layer. A best run does not
override a seed summary. A manually typed table does not override JSON.

## 8. Core Code Files

| File | Role | Notes |
|---|---|---|
| `datasets.py` | Dataset registry, paths and dataset metadata | Identifies ETT datasets; do not silently change identity |
| `data_loader.py` | Time-series reading, chronological splits, scaling and windows | Critical leakage boundary; scaler must fit training rows only |
| `experiment_protocols.py` | Named pilot/canonical protocols and configuration contracts | Protocol IDs are part of evidence provenance |
| `baseline_models.py` | DLinear, Linear, NLinear, PatchTST/other baseline definitions | Keep comparison inputs/windows identical |
| `kan_search_space.py` | 14-dimensional KAN HPO bounds and decoding | Historical Kaggle bounds are distinct from legacy local defaults |
| `kan_hpo_problem.py` | HPO objective/problem wrapper | Defines validation fitness contract |
| `kaggle_experiment.py` | Historical Kaggle HPO execution logic | Legacy campaign source; preserve for provenance |
| `experiment_runner.py` | Shared targeted/canonical execution machinery | Handles training, evaluation, records and event logging |
| `targeted_experiments.py` | Modern targeted experiment matrix and model variants | Source for K01-K08 bundle generation |
| `analysis.py` | General analysis and legacy/campaign figure generation | Requires scientific Python stack for full regeneration |
| `analysis/analyze_k09_diagnostics.py` | K09 ZIP validation, extraction, summaries and four publication figures | Reads archive without repacking it |
| `analysis/analyze_k10_multihorizon.py` | K10 120-record summaries, per-seed/step tables and three publication figures | Reads the immutable K10 ZIP only |
| `publication_tables.py` | Regenerates publication CSV/Markdown tables | Keeps K08/K09 and pilot/canonical groupings separate |
| `build_public_release_manifest.py` | Builds an allowlisted SHA-256 release manifest | Does not copy files or grant a license |
| `run_release_qa.ps1` | Runs the canonical publication/release QA pipeline | Does not train models; optional DOCX rebuild switch |
| `stage_public_release.py` | Copies only manifest-allowlisted files into private staging | Refuses to overwrite a non-empty directory |
| `audit_public_release.py` | Verifies staged files against manifest hashes/sizes | Rejects missing or unexpected staged files |
| `validate_pipeline.py` | Local pipeline sanity validation | Diagnostic, not primary evidence |
| `build_kaggle_bundle.py` | Builds self-contained Kaggle scripts | Regenerate bundles only for a new source version |
| `run_all_experiments.ps1` | Historical local orchestration | Do not treat local legacy output as new canonical evidence |

## 9. Kaggle Bundle Map

| Bundle | Purpose | Status/result location |
|---|---|---|
| `KAGGLE_01_SMOKE.py` | Small smoke validation of pipeline | Complete; `canonical_smoke_local/` is local diagnostic only |
| `KAGGLE_02_MAIN_H96.py` | ETTh1 h96 targeted baselines/ablations | Complete; `targeted_results/` |
| `KAGGLE_03_REMAINING_HORIZONS.py` | ETTh1 h192/h336/h720 | Complete; `targeted_results_03/` |
| `KAGGLE_04_KAN_IMPROVEMENT.py` | Training-consistency Stage 1 | Complete; `kan_improvement_results/` |
| `KAGGLE_05_KAN_IMPROVEMENT_STAGE2.py` | Clipping/residual Stage 2 | Complete; `kan_improvement_stage2_results/` |
| `KAGGLE_06_OPTIMIZER_STATE_CAUSAL.py` | ETTh1 Adam retain/reset intervention | Complete; `optimizer_state_causal_results/` |
| `KAGGLE_07_ETTH2_CONFIRMATION.py` | ETTh2 pilot confirmation | Complete; `etth2_confirmation_results/` |
| `KAGGLE_08_CANONICAL_CONFIRMATION.py` | Five-seed official ETT split, historical checkpoint contracts | Complete; `canonical_confirmation_results/` |
| `KAGGLE_09_UNIFORM_CANONICAL_DIAGNOSTICS.py` | Uniform validation-best plus prediction/checkpoint/compute diagnostics | Complete; `canonical_uniform_best_diagnostics.zip` |
| `KAGGLE_09_README.md` | K09 execution and expected-artifact instructions | Historical execution guide; campaign is now complete |

Never paste multiple bundle versions into one Kaggle notebook. Any future bundle
must use a new number/campaign ID and be preregistered.

## 10. Result and Data Directories

| Path | Contents and authority |
|---|---|
| `my_results/` | Historical 18-record six-algorithm HPO tree |
| `targeted_results/` | ETTh1 h96 targeted records and summary |
| `targeted_results_03/` | ETTh1 remaining-horizon records and summary |
| `targeted_results_all/` | Consolidated 96 targeted records used by analysis |
| `kan_improvement_results/` | 21 Stage 1 records |
| `kan_improvement_stage2_results/` | 9 Stage 2 records |
| `optimizer_state_causal_results/` | 12 K06 ETTh1 causal records |
| `etth2_confirmation_results/` | 24 K07 ETTh2 records |
| `canonical_confirmation_results/` | 40 K08 canonical records |
| `analysis/k09_diagnostics/records/` | Exact extracted copies of 40 K09 JSON records |
| `canonical_uniform_best_diagnostics.zip` | Authoritative K09 ZIP with JSON, NPZ predictions and checkpoints |
| `ETDataset/ETT-small/` | ETTh1/ETTh2 and other ETT CSVs; nested upstream repository |
| `efficient-kan/` | Pinned third-party KAN implementation; nested upstream repository |

Each result folder has a corresponding ZIP where registered. Preserve both.
Do not unpack/repack an archive and reuse the original hash.

## 11. Analysis and Figure Directories

### `analysis/`

- `publication/`: final generated table CSVs and `PUBLICATION_TABLES.md`.
- `publication/CHECKPOINT_CONTRACT_APPENDIX.md`: separate K08-to-K09
  checkpoint-contract sensitivity table and seed-level CSVs.
- `k09_diagnostics/`: K09 summaries, forecast-step values, paired differences
  and diagnostic report.
- `canonical_confirmation/`, `optimizer_state_causal/`,
  `etth2_confirmation/`, `improvement/`, `improvement_stage2/`: campaign-specific
  audits and summaries.
- `jeas_archived/`: archived accepted-article and journal-policy HTML.
- `docx_qa/`: historical DOCX audit intermediates; not submission evidence.
- `docx_render/`: intended PDF/render output; currently empty because
  LibreOffice CLI failed.
- `lo_profile*`: temporary LibreOffice profiles; exclude from public release.
- `literature/`, `jeas_benchmark/`: literature/benchmark support artifacts.

### `figures/`

The twelve manuscript figures are:

1. `study_workflow.png`
2. `ett_canonical_distribution_shift.png`
3. `hpo_fitness_three_runs.png`
4. `grid_update_ablation_h96.png`
5. `grid_update_dynamics.png`
6. `k09_actual_vs_predicted.png`
7. `k09_residual_diagnostics.png`
8. `k09_forecast_step_rmse.png`
9. `k09_compute_checkpoints.png`
10. `k10_canonical_multihorizon.png`
11. `k10_multihorizon_compute.png`
12. `k10_h720_forecasts.png`

All are approximately 300 dpi and below Electrical Engineering's archived 10 MB figure limit.
Other figures and `*_data.json` are supporting or historical artifacts.

## 12. Audit, Test and Document-Build Files

| File | Purpose |
|---|---|
| `audit_evidence.py` | Checks 380 records, archives, hashes and campaign integrity |
| `audit_k09.py` | Dedicated 40-task K09 artifact/checkpoint/prediction audit |
| `audit_k10.py` | Dedicated 120-task K10 multi-horizon artifact audit; use only after K10 execution |
| `audit_manuscript.py` | Checks required claims/headings/references and forbidden wording |
| `audit_submission_docx.py` | Checks tables, figures, alt text, page geometry, metadata and review state |
| `tests/test_frozen_protocol.py` | Dataset, split and K08 contract tests |
| `tests/test_k09_protocol.py` | K09 protocol/static contract tests |
| `tests/test_k09_audit.py` | K09 audit regression tests |
| `tests/test_k10_protocol.py` | Freezes the K10 bundle, matrix and campaign arguments |
| `tests/test_k10_audit.py` | K10 auditor rejection regression test |
| `format_submission_docx.py` | Applies Times New Roman, spacing, margins, numbering and table geometry |
| `strip_docx_comments.py` | Removes Word comments and package relationships |
| `set_figure_dpi.ps1` | Normalizes DPI metadata without changing frozen pixels |
| `strip_figure_titles.ps1` | Historical figure-title cleanup helper |
| `generate_workflow.ps1` | Regenerates workflow PNG |
| `generate_data_protocol_figure.ps1` | Regenerates canonical data/protocol figure |

`python` on this machine may resolve to MSYS and lack NumPy/python-docx. The
working Codex document runtime observed during QA was:

`C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

Do not hard-code that private runtime into the public instructions; create a
portable Python 3.12 environment instead.

## 13. Research, Journal and Submission Documents

| File | Meaning |
|---|---|
| `paper_draft.md` | Canonical manuscript source; current, not an obsolete SOTA draft |
| `paper_submission.docx` | Current author-neutral formatted submission artifact |
| `COVER_LETTER.md` | Current Electrical Engineering cover-letter draft; author approval pending |
| `DECLARATIONS.md` | Standalone declaration draft |
| `SUBMISSION_CHECKLIST.md` | Submission gates and manual checks |
| `FINAL_QA_REPORT.md` | Latest passed checks and unresolved renderer limitation |
| `PUBLICATION_READINESS_AUDIT.md` | Honest readiness verdict and scientific limitations |
| `Electrical Engineering_SUBMISSION_REQUIREMENTS.md` | Archived official requirement extraction; recheck live portal |
| `Electrical Engineering_ACCEPTED_ARTICLE_DEEP_AUDIT.md` | Detailed accepted-paper comparison |
| `Electrical Engineering_FULLTEXT_BENCHMARK.md` | Figure/table/word/reference patterns from full texts |
| `Electrical Engineering_BENCHMARK_AUDIT.md` | Gap matrix against related Electrical Engineering papers |
| `REFERENCE_AUDIT.md` | DOI/reference verification record |
| `PUBLIC_RELEASE_README.md` | Clean-release staging and exclusion rules |
| `PUBLIC_RELEASE_MANIFEST.json` | Generated allowlist, sizes and SHA-256 values; not yet a DOI release |
| `REPRODUCIBILITY.md` | Reproduction commands and evidence manifest; must stay synchronized |
| `ENVIRONMENT_MANIFEST.md` | Kaggle and publication environment distinction |
| `CLAIM_EVIDENCE_MATRIX.md` | Maximum allowed wording for each result |
| `EXPERIMENT_REGISTRY.md` | Authoritative campaign, record and hash registry |
| `MASTER_UPGRADE_ROADMAP.md` | Canonical plan for all future work |

Historical planning/research documents include `MASTER_PLAN.md`,
`MASTER_PLAN_EXECUTION.md`, `TARGETED_EXPERIMENTS.md`, `RESEARCH_REQUESTS.md`,
`plans/`, `r1.docx` through `r9.docx`, Vietnamese research DOCX files,
`all_docs.txt`, `research_content.txt`, `jeas_research.txt`, `r*_out.txt` and
`_tools/*.md`. They provide context but cannot override current evidence.

## 14. Verification Commands

With a Python environment containing NumPy and python-docx:

```powershell
python audit_evidence.py
python audit_k09.py canonical_uniform_best_diagnostics.zip
python audit_k10.py canonical_multihorizon_baselines_v2.zip
python publication_tables.py
python audit_manuscript.py
python -m unittest discover -s tests -v
pandoc paper_draft.md -o paper_submission.docx --resource-path=.
python format_submission_docx.py paper_submission.docx paper_submission.docx
python strip_docx_comments.py paper_submission.docx
python audit_submission_docx.py paper_submission.docx
```

Run the evidence audits before regenerating publication artifacts. After DOCX
generation, render to PDF and inspect every page; structural audit is not a
substitute for visual QA.

## 15. Environment and Repository Cautions

- The workspace root is not currently a Git repository.
- `ETDataset/` and `efficient-kan/` are nested upstream Git repositories.
- `.venv`, `.venv_publication` and `.venv_publication_win` are legacy or
  machine-specific; do not publish them.
- `__pycache__/` is generated and disposable.
- `.claude/` and `plans/` are agent/process metadata, not evidence.
- `qa_docx/`, `analysis/docx_qa/` and `analysis/lo_profile*` are temporary QA
  artifacts.
- Root-level unrelated PDFs/JPG/WEBP and installers such as UltraViewer and
  signing utilities are outside the research project. Preserve user files, but
  exclude them from every scientific/public release.
- `doc_images/` contains extracted images from research documents; copyright
  and provenance must be checked before reuse.

## 16. Immediate Next Actions

Follow `MASTER_UPGRADE_ROADMAP.md`. In short:

1. Back up the frozen workspace and create a clean release tree.
2. Use the completed public-release manifest and K09 per-seed supplement as the
   basis for the clean package; approve licenses before copying evidence.
3. Create portable Python 3.12 audit/publication environments and run tests.
4. Generate optional no-new-training supplement tables/interpretation only if
   their reconstruction validates.
5. Complete final manuscript language/reference QA.
6. Obtain author, CRediT, funding and repository information.
7. Recheck the live Electrical Engineering portal.
8. Fix/use a working DOCX renderer and inspect the final PDF page by page.
9. Deposit a DOI-bearing release, insert links, rerun audits and submit.

K10 canonical multi-horizon is complete. V1 was abandoned; the first v2
execution completed but its archive was not retrievable after a Kaggle reset.
The download-safe v2 rerun is the sole authoritative archive and passes the
dedicated 120-record audit. Do not start external-dataset, multivariate, fair-HPO or
all-horizon Grid Update campaigns unless separately authorized and
preregistered.

## 17. Final Handover Status

| Area | State |
|---|---|
| Historical HPO | Complete, descriptive only |
| Targeted ETTh1 baselines/multi-horizon | Complete |
| Training-consistency repair | Complete |
| Residual KAN negative study | Complete |
| K06 optimizer-state causal study | Complete, 12/12 |
| K07 ETTh2 confirmation | Complete, 24/24 |
| K08 canonical confirmation | Complete, 40/40; historical checkpoint contracts |
| K09 uniform diagnostics | Complete, 40/40; primary canonical publication comparison |
| Frozen evidence | 380 records; K09 and K10 dedicated audits pass |
| Manuscript | Author-complete working submission draft; repository URL/DOI pending |
| Figures/tables | 12 figures and 9 editable tables; structural audit passes |
| Electrical Engineering benchmarking | Complete from archived requirements and accepted papers |
| Public reproducibility release | Private allowlisted staging v13 assembled and audited with authoritative K10 evidence; GitHub/Zenodo publication pending |
| Visual PDF QA | Previous user PDF rejected due broken equations; corrected DOCX ready for fresh export and page-by-page inspection |
| New Kaggle work | Not required for current claims |
| K10 canonical multi-horizon | Complete 120/120; authoritative ZIP audited; DLinear lowest mean in all eight cells |

The project is technically strong enough for review when submission blockers are
closed. Its remaining scientific limitations are scope limitations, not hidden
pipeline failures: two related datasets, univariate OT, five seeds per
canonical cell and one recorded GPU environment.
