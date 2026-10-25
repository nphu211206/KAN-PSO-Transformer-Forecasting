# Electrical Engineering Submission Checklist

- [ ] Confirm the current Electrical Engineering article type and portal fields immediately before submission. Archived official pages verify no author-payable APC and no stated word/figure cap, but policies may change.
- [x] Insert the verified sole-author identity, affiliation, corresponding email,
  ORCID and CRediT roles supplied by Nguyen Phu Toan.
- [ ] Add final repository/DOI links after GitHub and Zenodo publication. Archived official policy specifies single-blind review.
- [x] Generate an allowlisted `PUBLIC_RELEASE_MANIFEST.json` that excludes
  unrelated files, environments, profiles and executables.
- [x] Build and audit private `release_staging_v13` from the manifest; do not
  publish it until license, redistribution and DOI decisions are approved.
- [x] Generate the 40-row K09 per-seed supplementary CSV directly from frozen JSON.
- [x] Approve a project license, third-party notices and dataset redistribution
  policy before building the DOI-bearing public archive.
- [x] Verify DOI-bearing references against Crossref; record status in `REFERENCE_AUDIT.md`.
- [x] Prepare manuscript, figures, tables, declarations and cover letter.
- [x] Run `python audit_evidence.py` and retain the passing output.
- [x] Add `audit_manuscript.py` for frozen numbers, references and forbidden claims.
- [x] Confirm that pilot and canonical results are never pooled in claims.
- [x] Check figure first-mention order, title length and PNG size against the archived Electrical Engineering guideline.
- [x] Include abbreviation list, exact Declarations headings and LLM-use disclosure required by the archived guideline.
- [x] Check all twelve figures at 300-dpi metadata and verify accessible DOCX descriptions.
- [x] Verify manuscript tables are editable Word table objects.
- [x] Use only the author, affiliation, corresponding-author and ORCID data
  explicitly supplied by Nguyen Phu Toan.
- [x] Run dependency-light dataset and canonical-protocol tests.
- [x] Record technical strengths, data limits, checkpoint deviation and optional
  new-experiment paths in `PUBLICATION_READINESS_AUDIT.md`.
- [x] Strengthen the literature review with six directly relevant,
  Crossref-verified sources; do not pad toward an assumed word count.
- [x] Apply US Letter geometry, 1-inch margins, double line spacing plus line
  and page numbering to the author-neutral submission DOCX.
- [x] Freeze all nine editable tables to matching DXA table/grid/cell widths,
  repeat header rows and remove table shading.
- [x] Run `audit_submission_docx.py`; verify blank personal metadata, no
  comments/tracked changes, twelve described drawings and unchanged text/media.
- [ ] Export the corrected `paper_submission.docx` to PDF and inspect every page.
  The previously exported PDF has broken display equations and must not be used.
- [x] Recheck table color/shading in the current submission DOCX.
- [ ] Recheck table geometry and shading after applying any later journal template.
