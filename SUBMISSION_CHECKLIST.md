# Electrical Engineering Submission Checklist - PSO-TCWavKAN v9.1

**Manuscript:** `paper_v91_FINAL.docx` / `paper_v91_FINAL.pdf` (34 pages, 9 tables, 10 figures)
**Target:** Electrical Engineering (Archiv fuer Elektrotechnik), Springer
**Updated:** 2026-11-01

## Manuscript and figures

- [x] Apply all reviewer points: duplicate Table 6 note, duplicate consent, Table 6 provenance, CLI mismatch, 25-evaluation CLPSO count, v9.0 labels, PatchTST P=16/S=8, raw-q scale, ddof=1, A5 provenance, Table 7/Fig. 3 support.
- [x] Reconcile abstract and Section 4 statistics: full-precision Wilcoxon W=28.0, p=0.0386, Holm p=0.116; rounded variant W=29.0, p=0.0443, Holm 0.133.
- [x] Correct Table 6 caption to an exploratory mixed-source diagnostic and note that the first three clean entries are benchmark means; the fourth row uses the pre-SWA Seed 1000 checkpoint. Report raw MSE only and remove all within-checkpoint degradation percentages.
- [x] Add `reproduce_table7_params.py` and document the Fig. 3 source-data/script dependencies.
- [x] Correct Table 7 to an observational profiling statement; remove any claim of archived/reproduced profiling logs.
- [x] Re-label Fig. 3 as a B-spline-only diagnostic and Figs. 5, 6, 9, 10 as predecessor-checkpoint diagnostics.
- [x] Neutralize ETT load labels and avoid unsupported seasonal interpretation.
- [x] Confirm PatchTST Lite uses patch length 16 and stride 8 for both lookbacks.
- [x] Confirm the searched parameter is raw q=`log_scale`, with effective scale `softplus(q)+1e-4`.
- [x] Confirm exactly one Table 4 caption, one "Consent for publication" heading, and one "Note: Table 6" in the final PDF.
- [x] Verify 9 editable Word table objects and 10 inline figures in the DOCX.
- [x] Inspect the final 34-page PDF page by page: equation numbers are right-aligned, Table 4/5 headers no longer split model names, Table 6 note is on the same page 27, and Table 7/declarations render without overflow.
- [ ] Confirm the journal's current article type, word/figure limits, and APC policy immediately before submission; archived policy pages may be stale.

## Final consistency audit (2026-11-01)

- [x] Correct A3 ETTh2 L=336/H=720 MSE from 0.4682 to 0.5004; the 0.4682 value remains only in its legitimate PSO MAE cell in Table 4 (ETTh2 L=96/H=336).
- [x] Correct ETT provenance to "two electricity transformers at two stations in two regions of the same province"; remove unsupported dispatch/volatility attribution.
- [x] Correct P191 physical-error conversion to the 70/10/20 training SD (11.40 degC): 5.54 vs 5.72 degC (0.18) and 6.41 vs 6.67 degC (0.26); distinguish the 12/4/4 diagnostic archive (11.59 degC).
- [x] Remove unsupported "tournament", "parameter-matched", "definitive state-of-the-art", "commanding performance", "maintains its dominance", and "grounded, stable" wording.
- [x] Neutralize Table 5 assessment mechanisms ("trend aliasing", "suboptimal ripple basin", "dampening") to factual descriptions; hedge the H=720 phase-drift explanation.
- [x] Regenerate Fig. 8 with short A1-A5 labels and the annotation clear of the bars; embed the updated image in the DOCX and regenerate the PDF.
- [x] Verify final PDF pages 21, 24-26, and 30: Table 4 cell, Table 5, Fig. 8, A3 0.5004, and 11.40 degC render correctly.
- [x] Sync `paper_v91_FINAL.*`, `paper_submission.*`, cover letter, declarations, `paper_v91_FINAL.md`, and Fig. 8 between the root and `github_release`; hashes match.
- [x] Regenerate cover-letter and declarations DOCX/PDF from the corrected Markdown and sync to the release package.
- [x] Confirm the cover-letter date with the author: November 1, 2026.
- [x] Authorize and create the release commit (`c473e97`) and update `REPRODUCIBILITY.md` / `EXPERIMENT_REGISTRY.md` with the base commit hash.

## Statistics and data provenance

- [x] Reproduce rounded Table 4 tests: NLinear W=29.0, p=0.0443, Holm 0.1329; DLinear W=46.0, p=0.2744; PatchTST Lite W=35.0, p=0.0934.
- [x] Reproduce full-precision CSV tests: NLinear W=28.0, p=0.0386, Holm 0.1159; DLinear W=46.0, p=0.2744; PatchTST Lite W=35.0, p=0.0934.
- [x] Synchronize the A5 row in `TABLE2_ABLATION_FINAL.csv` to the benchmark CSV (mean, SD, CI, seeds 1000-3000); document that A5 is benchmark-referenced and its mean includes seeds 4000-5000.
- [x] Document that A1-A4 use 3 seeds (192 runs), while A5 is the five-seed benchmark mean.
- [x] Retain per-seed raw logs/checkpoints on request only; the release states they are not included.
- [x] Retain the Table 7 profiling caveat; profiling logs/script are not archived, so only parameter counts are reproducible.

## Release package

- [x] Patch `pso_tcwavkan_v91.py`: v9.1 labels, argparse `--run_mode`/`--mode`, `DEMO` dispatch, 25-evaluation CLPSO counter/assert, raw-q comments, PatchTST P=16/S=8 comments, sample SD ddof=1.
- [x] Verify `python pso_tcwavkan_v91.py --help` and `py_compile`.
- [x] Add `reproduce_statistical_tests.py` and verify both rounded and full-precision blocks against `results/TABLE1_BENCHMARK_FINAL.csv`.
- [x] Update `README.md`, `REPRODUCIBILITY.md`, `EXPERIMENT_REGISTRY.md`, and `CLAIM_EVIDENCE_MATRIX.md` to the corrected claims.
- [x] Add the final repository URL and commit hash; [ ] add the Zenodo DOI after the DOI-bearing archive is deposited.
- [ ] Confirm `LICENSE`, third-party notices, and ETT redistribution terms before publishing the DOI-bearing archive.
- [x] Restore the tracked `__pycache__/pso_tcwavkan_v91.cpython-313.pyc` and remove untracked pycache files.

## Cover letter and declarations

- [x] Identify the stale claims in the current cover letter and declarations (old title, 240 runs, Holm 0.133 as if significant, "under 1.4%", 3 seeds).
- [x] Rewrite the cover letter to: 192 A1-A4 runs plus benchmark-referenced A5; nominal NLinear p=0.0386 but Holm p=0.116; ETTh2 multi-horizon gains and ETTh1/H=720 limitations; diagnostic-only figure/Table 7 caveats.
- [x] Rewrite the declarations to match the same seed/provenance/statistical boundaries; the cover letter is dated November 1, 2026.
- [x] Regenerate DOCX via `generate_cover_letter_ee_docx.py` and `generate_declarations_ee_docx.py`; export PDF via LibreOffice.
- [x] Proofread the final cover letter and declarations PDFs for stale "Lower-Bound Guarantees" wording and old numbers.

## Final gate

- [x] Re-run `reproduce_statistical_tests.py` against the release CSV.
- [x] Re-run DOCX-to-PDF conversion and confirm 34 pages and unchanged key counts.
- [x] Ran a text search over the manuscript, README, cover letter, and declarations for: `240`, `564c881`, `SESSION3`, `Lower-Bound Guarantees`, `under 1.4%`, `Seed 42`, `v9.0`, `30 evaluations`, `log10(s)`, `139.6`, `+13.0%`, `3.3x faster than PatchTST`, `tournament`, `parameter-matched`, `definitive state-of-the-art`, `commanding performance`, `maintains its dominance`, `different provinces`, `consistently underperform`, `grounded, stable`, `Trend aliasing`. Zero stale hits remain; `0.4682` remains only in the legitimate Table 4 MAE cell.
- [ ] Obtain author approval of the final PDF and the public release commit.
