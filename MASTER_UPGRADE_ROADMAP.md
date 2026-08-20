# KAN-HPO Project Master Plan

**Plan status:** canonical plan after completed K10  
**Updated:** 2026-08-17  
**Target venue:** Journal of Engineering and Applied Science (JEAS), Springer Nature  
**Evidence freeze:** 380 completed experiment records through K10

**Latest campaign:** corrected K10 v2 canonical multi-horizon study completed
120/120 and passed archive/artifact audit. V1 and the unretrievable first v2
session remain documented execution history, not evidence.

This is the controlling plan. `MASTER_PLAN.md`, `MASTER_PLAN_EXECUTION.md`,
`TARGETED_EXPERIMENTS.md` and `plans/` describe earlier phases and remain
historical records; they do not override this plan, the registry or frozen JSON.

## 1. Current Baseline

The paper is defensible as a bounded engineering reliability study. It is not a
universal KAN-superiority or state-of-the-art forecasting paper.

| Area | Current state | Remaining issue |
|---|---|---|
| Evidence | 380 records; archive/source/dataset hashes audited through K10 | Preserve immutability and publish a clean manifest |
| Data | ETTh1/ETTh2, univariate OT, chronological splits, training-only scaling | Same benchmark family; no external-domain validity |
| Fairness | K09 uses validation-best selection for every condition | Keep K08 separate as historical checkpoint sensitivity |
| Diagnostics | K09 mechanism diagnostics plus K10 multi-horizon predictions/compute/checkpoints | No FLOPs, calibrated uncertainty or independent-domain validation |
| Manuscript | About 7000 words, 25 references, 12 figures, 9 editable tables | Repository links, final PDF QA and portal check |
| Submission DOCX | Structural audit passes | LibreOffice CLI does not create a PDF from the audited DOCX |
| Reproducibility | Registry, automated audits and private release staging are strong | Legacy virtual environments are not portable; license/DOI deposition pending |

### Progress implemented on 2026-08-17

- Generated a 40-row K09 per-seed supplementary table from frozen JSON.
- Generated a descriptive K08-to-K09 checkpoint-contract appendix without
  pooling campaigns or treating versioned campaigns as causal replications.
- Added regression tests that enforce the K08/K09 boundary and exact K09 matrix.
- Added an allowlisted public-release manifest with SHA-256 checks and explicit
  exclusion of private/unrelated workspace classes.
- Built and audited `release_staging_v13` from the manifest. It has not been
  published and must not be described as a DOI-bearing public release.
- The release remains staging-only until license, redistribution policy,
  author information and DOI deposition are complete.

## 2. Non-Negotiable Scientific Boundaries

1. JSON records and original ZIP archives are primary evidence.
2. K08 and K09 are separate campaigns and must never be pooled.
3. Pilot `pilot_60_20_20_v1` and canonical `ett_canonical_12_4_4_v1` results
   must remain in separate comparisons.
4. Never use test metrics for configuration, epoch or schedule selection.
5. Do not claim SOTA, universal KAN/DLinear superiority, universal Grid Update
   harm or broad external generalization.
6. Do not add post-hoc p-values or equivalence claims to five observed seeds.
7. Do not call the univariate OT experiment multivariate or infer feature
   importance from it.
8. Do not edit frozen JSON, repack archives or replace outlier/failed runs.
9. Future experiments require preregistration, a new campaign ID, source hash,
   result directory and archive hash.

## 3. Workstream A: Submission Blockers

These tasks require no new training but must be complete before upload.

### A1. Author and administrative data

- Verified sole-author identity, affiliation, corresponding-author email, ORCID,
  CRediT, funding, competing interests and acknowledgements are now supplied.
- Keep only the repository URL/DOI and any portal-specific author fields open.

**Exit gate:** no placeholder remains in the author-complete copy and all
authors approve the manuscript, cover letter and declarations.

### A2. Repository and archival release

- Create a clean release tree; do not publish this workspace wholesale because
  it contains unrelated PDFs/installers, legacy extracts, nested repositories
  and machine-specific environments.
- Include code, frozen evidence or legally distributable archives, audits,
  tables, figures, manuscript source, requirements, registry and hashes.
- Exclude `.venv*`, `__pycache__`, LibreOffice profiles, temporary QA copies,
  unrelated media, executables, remote-access tools and credentials.
- Add a project license, third-party notices and citations/licenses for ETT and
  efficient-kan.
- Deposit an immutable release in Zenodo or another DOI-bearing archive and
  insert the DOI/URL in the manuscript.
- Record the release SHA-256 in the registry and reproducibility document.

**Exit gate:** a third party can download one deposited release, verify it and
rebuild publication tables without this private workspace. The locally audited
  `release_staging_v13` is only preparation for this gate.

### A3. Live JEAS policy verification

- Recheck aims/scope, Research instructions, file limits, reference style,
  declarations, AI policy, review model and all portal fields.
- Treat the live portal as final authority; archived 2024 guidance is not proof
  that every 2026 field is unchanged.
- Save a dated snapshot and confirm whether graphical abstract, highlights,
  title page, supplementary files or checklists are requested.

Links:

- https://jeas.springeropen.com/
- https://link.springer.com/journal/44147
- https://jeas.springeropen.com/submission-guidelines/preparing-your-manuscript/research

**Exit gate:** `JEAS_SUBMISSION_REQUIREMENTS.md` and
`SUBMISSION_CHECKLIST.md` show the current check date and exact portal fields.

### A4. Final manuscript editorial pass

- Keep the title/abstract centered on reliability and stability, not a winning
  hybrid-model narrative.
- Check every number against generated CSV or JSON and every symbol against its
  method definition.
- Copyedit English, tense, terminology and paragraph length.
- Recheck DOI metadata, citation order and Springer Basic reference style.
- Ensure captions state dataset, protocol, metric, seed count and metric units.
- Keep limitations explicit: two related datasets, univariate OT, five seeds,
  one T4 environment, no FLOPs or probabilistic calibration.
- Do not pad toward an invented 8000-word limit; the archived official guidance
  did not state that cap.

**Exit gate:** `audit_manuscript.py` passes and manual readers identify the
same research question, primary result and claim boundary.

### A5. DOCX/PDF production QA

- Keep `paper_submission.docx` as the current author-complete audited copy.
- Export a fresh PDF from the corrected DOCX. The previously supplied PDF had
  broken display equations and is rejected by the visual QA gate.
- Inspect every rendered page for overflow, orphan captions, clipped equations,
  image legibility, line/page numbers, references and declarations.
- Re-run formatter, comment stripper and DOCX audit after any template edit.
- Verify 9 editable tables, 12 described figures, no shading/comments/tracked
  changes and appropriate metadata.

**Exit gate:** the final PDF is inspected page by page and checksummed.

## 4. Workstream B: Reproducibility and Engineering Hardening

### B1. Portable environments

- Preserve `.venv*` only as local history; do not claim portability.
- Create a clean Python 3.12 training/audit environment from
  `requirements-canonical.txt` and a lightweight publication environment from
  `requirements-publication.txt`.
- Pin versions and document separate requirements for evidence auditing,
  document building and full CUDA retraining.
- Provide one ordered command sequence for all audits, tests, tables and DOCX.

### B2. Automated regression suite

- Retain tests for dataset hashes, temporal boundaries, window counts,
  campaign matrices, selection rules and artifact hashes.
- Add explicit tests that reject pilot/canonical and K08/K09 pooling.
- Test figure/table numbering, declarations, forbidden claims, placeholders and
  release-manifest SHA-256 values.
- Run the suite in a clean environment before DOI deposition.

### B3. Canonical build pipeline

Use this order:

1. `audit_evidence.py`
2. `audit_k09.py canonical_uniform_best_diagnostics.zip`
3. `publication_tables.py`
4. figure regeneration or pixel/DPI validation
5. `audit_manuscript.py`
6. Pandoc DOCX generation
7. `format_submission_docx.py`
8. `strip_docx_comments.py`
9. `audit_submission_docx.py`
10. PDF rendering and visual QA

Avoid manual spreadsheet edits. Store build logs and final checksums in release
QA artifacts.

### B4. Data and license governance

- Confirm redistribution terms for ETT CSVs and efficient-kan.
- Include licenses and attribution in the public package.
- Document LF/CRLF canonicalization without changing values.
- Document target, units, sampling, missingness, timestamp continuity, split
  boundaries and training-only scaler behavior.
- Keep raw data immutable; every derivative must identify dataset hash and
  protocol ID.

## 5. Workstream C: Analysis Without New Training

These tasks can strengthen the paper or supplement using K09 artifacts.

1. Add a checkpoint-sensitivity appendix comparing K08 final-epoch contracts
   with K09 validation-best contracts without pooling them.
2. Generate per-seed supplementary tables for selected epoch, MSE, MAE, OT
   RMSE, latency and peak memory.
3. Show seed dispersion of point forecasts, explicitly not calibrated
   predictive uncertainty.
4. Load selected K09 checkpoints and inspect a preregistered subset of spline
   edge functions or lag responses. Validate exact model reconstruction before
   interpretation.
5. Add one illustrative OT-threshold scenario, not a validated maintenance
   policy, unless domain incident labels exist.
6. Keep parameter/time/latency/memory reporting. Add FLOPs only after validating
   a profiler for custom spline operations.
7. Add a provenance diagram linking dataset hash, protocol, campaign, JSON,
   prediction/checkpoint, derived artifact and manuscript claim.

## 6. Workstream D: Optional New Science

These are not prerequisites for the current JEAS paper.

### D1. Canonical multi-horizon

- K10 completed DLinear versus corrected KAN versus PatchTST on
  ETTh1/ETTh2 at horizons 96/192/336/720 with five seeds, uniform
  validation-best checkpoints and saved diagnostics: 120 completed runs.
- The authoritative archive passed `audit_k10.py`; derived tables and figures
  are stored under `analysis/k10_multihorizon/` and `figures/k10_*`.
- V2 contains a hard pre-execution assertion that the task matrix is exactly
  120; the withdrawn v1 partial run must not be reused.
- A repeated Grid Update mechanism study across all horizons is deliberately
  excluded; K09 remains the h96 mechanism campaign.

### D2. External engineering data

- ETTm1/ETTm2 broaden temporal resolution but remain in the ETT family.
- A genuinely independent engineering dataset is needed for external-domain
  validity.
- Predefine target, units, sampling, missing-data handling, split, scaler,
  baselines and operational consequence.

### D3. Multivariate/interpretability study

- Register multivariate inputs before feature-importance or interaction claims.
- Compare with suitable multivariate baselines under identical windows/splits.
- Treat this as a new campaign or follow-up paper.

### D4. Fair optimizer ranking

- Historical HPO is descriptive because it has three repetitions and unequal
  effective budgets.
- Ranking requires identical search space, evaluations, stopping and objective,
  with at least five runs per method: at least 30 HPO runs for six methods.

### D5. New Grid Update remedies

- Candidate questions: function-preserving optimizer-state transport, warm
  restart, adaptive triggers and constrained grid motion.
- Require paired seeds, event logs and a no-update control.
- Adam reset failure does not prove optimizer state is never causal.

## 7. Final Submission Package

Include manuscript DOCX/PDF, separate high-resolution figures, editable tables
or supplementary CSV, cover letter, declarations, registry, claim matrix,
reproducibility/environment manifests, code/tests, evidence manifest, journal
checklist, README, license and third-party notices.

Exclude private drafts, unrelated files, installers, remote-access tools, local
environments, browser profiles and credentials.

## 8. Reviewer-Response Readiness

Prepare evidence-backed answers for:

- Why only ETTh1/ETTh2 and why these four canonical horizons?
- Why no significance claim?
- Why do K08 and K09 differ?
- Why can a lower-parameter KAN be slower and less accurate than DLinear?
- Why was Grid Update unstable and why did Adam reset fail?
- Why is historical HPO not an optimizer ranking?
- What does error in OT units mean operationally?
- Can every table and figure be regenerated from frozen records?

Do not run reviewer-requested experiments until the requested claim, protocol
and cost are understood and internally preregistered.

## 9. Priority Gates

| Priority | Gate | Required result |
|---|---|---|
| P0 | Submission identity | Verified authors, CRediT and declarations |
| P0 | Public evidence | Clean DOI-bearing release and checksum |
| P0 | Journal compliance | Live portal requirements checked and recorded |
| P0 | Production | Author-complete DOCX/PDF visually inspected |
| P1 | Reproducibility | Clean Python 3.12 environments and full tests |
| P1 | Editorial | Final language/reference/claim audit |
| P1 | Supplement | Per-seed K09 and provenance materials |
| P2 | Analysis | Spline/lag interpretation if reconstruction validates |
| P3 | New science | External data, multivariate, uncertainty or fair HPO |

## 10. Execution Order and Definition of Done

1. Back up and freeze the 380-record workspace.
2. Build the clean release tree and portable audit environment.
3. Finish validated no-new-training supplementary analyses.
4. Perform final manuscript/reference copyedit.
5. Obtain author, CRediT, funding and repository information.
6. Recheck the live JEAS portal.
7. Generate and visually inspect the author-complete DOCX/PDF.
8. Deposit the release, insert DOI/URL, rerun audits and record hashes.
9. Submit only after all P0 gates pass.
10. Run D1-D5 only for an explicit broader objective or reviewer need.

The current paper is ready to submit only when all P0 gates pass. The project is
not complete in every possible dimension until independent external-domain and
uncertainty evidence exists, but those additions are not required for an honest
JEAS reliability paper.
