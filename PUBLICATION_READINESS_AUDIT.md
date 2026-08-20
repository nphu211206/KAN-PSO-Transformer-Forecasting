# Publication Readiness Audit

Audit date: 2026-08-19.

## Executive verdict

The project is technically credible and is within the scientific scope of the
*Journal of Engineering and Applied Science* (JEAS). Its strongest contribution
is a reproducible reliability study of KAN training contracts, Grid Update
instability and optimizer-state interventions, not a claim that KAN or a
metaheuristic is universally superior.

The evidence is sufficient for the current narrow claims. The project is not
yet submission-complete because author metadata, repository deposition, final
page-render inspection and a live-portal policy check remain outstanding. Acceptance
cannot be guaranteed. The main scientific risks are limited domain breadth and
limited external-domain breadth, rather than insufficient experiment count or
unresolved checkpoint fairness.

## Readiness by dimension

| Dimension | Verdict | Evidence and consequence |
|---|---|---|
| Journal scope | Ready | Electrical engineering, machine intelligence, optimization, modelling and simulation are explicitly in JEAS scope. |
| Core technical validity | Ready with disclosed limitation | Splits are chronological, the scaler is fitted on training rows only, targets remain within assigned splits, metrics aggregate all forecast elements, seeds are controlled and Grid Updates are transactional. |
| Data adequacy | Adequate for narrow claims | ETTh1 and ETTh2 provide two canonical hourly transformer-temperature benchmarks with verified content hashes. They support conclusions about these ETT/OT settings, not universal forecasting or cross-domain claims. |
| Experimental depth | Strong for mechanism and reproducibility | 380 frozen records cover HPO, multi-horizon pilots, training corrections, Grid Update schedules, Adam-state interventions, K08 sensitivity evidence, K09 uniform-checkpoint diagnostics and K10 canonical multi-horizon baselines. |
| Comparison fairness | Addressed in K09, historical limitation retained | K08 condition-specific checkpointing is retained as historical sensitivity evidence. K09 applies validation-best checkpointing uniformly to all four conditions and is the canonical publication comparison. |
| Baseline breadth | Strong, protocol-bounded | Six pilot baselines cover four horizons; K10 compares DLinear, PatchTST and corrected KAN over both datasets and four canonical horizons. Pilot, K08, K09 and K10 evidence must never be pooled. |
| Statistical reporting | Appropriate | Seed-level values, means and sample SDs are reported. Post-hoc p-values and unsupported significance language are intentionally excluded. |
| Reproducibility | Exceeds sampled JEAS papers | Frozen JSON, source/dataset/archive hashes, protocol identifiers, registries, automated audits and dependency-light tests provide unusually strong provenance. |
| Engineering interpretation | Ready | Standardized errors are translated into recorded OT units and connected to deployment safeguards and instability risk. |
| Manuscript presentation | Credible | Twelve figures and nine editable tables are publication-ready. The manuscript has 25 verified references, within the 23-67 range of the audited comparable articles. |
| Submission administration | Partly complete | Sole-author identity, affiliation, corresponding email, ORCID, CRediT, funding, competing interests and acknowledgements are complete. Repository DOI/URL, fresh PDF visual QA and final portal checks remain. |

## Comparison with accepted JEAS papers

Four archived publisher full texts closest to the topic were audited. They
contain 5449-7705 body words, 9-14 figures, 5-7 tables and 23-67 references.
Common elements include a workflow, several baselines, ablation or sensitivity,
compute reporting, prediction/residual diagnostics and an operational decision
context.

This project already matches or exceeds the sample in workflow reporting,
baseline/ablation coverage, compute reporting, operational interpretation and
especially reproducibility. It is weaker in three areas:

1. The evidence covers only ETTh1 and ETTh2, both from the same benchmark family,
   with univariate OT.
2. K09 supplies mechanism diagnostics at h96; K10 supplies prediction,
   checkpoint, timing and memory artifacts across four horizons, but neither
   campaign establishes calibrated uncertainty or external-domain validity.
3. The manuscript remains less broad in interpretability than papers with
   multivariate inputs because the present study is univariate.

The official archived JEAS Research guideline states no manuscript word limit,
numerical abstract limit or total figure/table cap. The manuscript should not
be padded toward an assumed 8000-word ceiling. A defensible target is the length
needed to close the literature and discussion gaps, likely within the observed
accepted-paper range, subject to a final live-portal check.

## Data and protocol judgement

The local ETTh1 and ETTh2 files differ from Kaggle bytes only by Windows CRLF
line endings. CRLF-to-LF canonicalization reproduces the frozen Kaggle SHA-256
hashes exactly; row counts, timestamps and numerical values are unchanged.
Automated tests also verify hourly continuity, canonical split/window counts,
the K08/K09 h96 matrices, K10's 120-record multi-horizon matrix and executed
checkpoint contracts.

The data are therefore intact and adequate for the paper's bounded conclusions.
They are not adequate for claims of universal KAN behavior, multivariate feature
importance, cross-industry generalization or state-of-the-art forecasting.

## Technical strengths that should lead the paper

- Training-only scaling and chronological temporal evaluation avoid leakage.
- Negative, residual and catastrophic outcomes are retained rather than filtered.
- Grid Update events, probe displacement and Adam-state transitions provide
  mechanism-level evidence beyond aggregate accuracy tables.
- The optimizer reset intervention directly tests a plausible causal remedy and
  shows that resetting Adam is not sufficient.
- Canonical ETTh1/ETTh2 evidence shows that corrected KAN improved substantially
  but did not beat DLinear, preventing an inflated hybrid-model narrative.
- Hashes and automated audits make every numerical claim traceable to frozen
  records and protect protocol boundaries.

## Scientific risks that must remain explicit

- Two related datasets do not establish broad external validity.
- Univariate OT cannot support feature-importance claims.
- K10 establishes the registered four-horizon behavior only for ETTh1/ETTh2,
  DLinear, PatchTST and the frozen corrected-KAN configuration.
- K08 condition-specific checkpointing is historical sensitivity evidence;
  K09 and K10 use uniform validation-best checkpointing.
- Three historical HPO repetitions and unequal historical budgets limit claims
  about optimizer ranking.
- K09 and K10 provide prediction arrays, model checkpoints, peak GPU memory and
  timing; FLOPs and calibrated predictive uncertainty are still not measured.

These are limitations, not reasons to discard the paper. They become rejection
risks only if the manuscript overstates its scope or hides the checkpoint issue.

## Required before submission

1. Deposit the final reproducibility package and insert a stable repository URL
   or DOI without changing frozen evidence.
2. Export the corrected DOCX to PDF and inspect every page; the user's previous
   PDF failed because display equations were broken.
3. Reconfirm the live JEAS article type, portal fields, file limits and current
   policy immediately before upload.

## Optional work requiring a new registered experiment

No additional Kaggle run is required for the claims currently made. K09 and
K10 are complete; new runs would be justified only to support a materially
broader question, for example:

- a third, genuinely external dataset family;
- multivariate forecasting with a pre-specified interpretability analysis;
- calibration or uncertainty diagnostics beyond the stored K09 predictions.

These additions could reduce reviewer risk, but they must be preregistered and
reported as a new campaign. They must not silently repair or replace the frozen
canonical evidence.

## Submission decision

Proceed with manuscript completion for JEAS using the current frozen evidence.
Do not claim SOTA, universal KAN superiority, strict uniform-checkpoint
confirmation or broad generalization. The project is already strong enough to
be reviewed as an engineering reliability and reproducibility study; it is not
strong enough to support a generic "KAN beats baselines" paper.
