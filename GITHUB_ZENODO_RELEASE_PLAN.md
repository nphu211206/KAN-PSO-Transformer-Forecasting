# GitHub and Zenodo Release Plan

Plan date: 2026-08-19.

## Decision

Use both services because they solve different problems:

- GitHub is the readable, version-controlled home for code, documentation,
  lightweight tables, figures, tests and issue history.
- Zenodo stores an immutable release snapshot, supports large files and assigns
  the DOI cited by the manuscript.

The manuscript should cite the Zenodo DOI as the archival object and link the
GitHub repository as the living source repository.

## GitHub release contents

Include project-owned source code, audit scripts, protocol documents,
requirements, manuscript source, derived CSV/Markdown tables, publication
figures, claim matrix, registry, reproducibility instructions, `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

Do not push virtual environments, caches, browser/LibreOffice profiles,
credentials, unrelated documents, nested `.git` directories or raw ETT CSV
files. Do not commit the 675 MB K10 archive to normal Git history. GitHub's
ordinary file limit also makes the large K09/K10 artifacts unsuitable for a
normal repository commit.

## Zenodo deposit contents

Create a versioned record with:

- the exact GitHub release source archive;
- the public release manifest and SHA-256 values;
- legally distributable frozen JSON evidence and supplementary CSV files;
- large K09/K10 archives only after confirming that arrays containing ETT target
  values may be redistributed under the ETTDataset terms;
- a README that maps every archive to its campaign and auditor;
- author ORCID `0009-0005-0529-2131` and affiliation metadata.

If redistribution of prediction archives is uncertain, deposit the code,
metrics, checkpoints where permitted, manifest and hashes, while supplying
instructions for reconstructing prediction artifacts from the official ETT
source. Do not edit or repack a frozen archive merely to remove inconvenient
content; create a separately named public derivative package and retain the
original hash in the private registry.

## Licensing

- Project-owned software: MIT License, copyright Nguyen Phu Toan, 2026.
- `efficient-kan`: retain upstream MIT copyright and license.
- ETTDataset/raw ETT data: retain upstream CC BY-ND 4.0 attribution; do not
  relicense under MIT.
- Derived research tables/figures: publish under the project terms only where
  no third-party restriction applies.

The local `LICENSE` and `THIRD_PARTY_NOTICES.md` are prepared but should be
reviewed before the first public release.

## Release sequence

1. Create an empty public GitHub repository under the author's account.
2. Copy only the audited GitHub allowlist into a clean repository tree.
3. Run tests and audits in the clean tree.
4. Commit and tag `v1.0.0`.
5. Create a GitHub release from the tag.
6. Create a Zenodo draft and enter author, ORCID, title, description, keywords,
   license and related GitHub identifier.
7. Upload the audited archival package and verify file hashes in the Zenodo
   draft before publishing.
8. Publish Zenodo, record the DOI and concept DOI, then insert the DOI and GitHub
   URL into the manuscript, declarations, cover letter and registry.
9. Rebuild DOCX/PDF and run the full release QA once more.

## Metadata draft

- Creator: Nguyen Phu Toan.
- Affiliation: Department of Information Technology, Hanoi University of
  Business and Technology, Hanoi, Vietnam.
- Resource type: Software, with associated research data/supplementary evidence.
- Suggested repository name: `kan-grid-update-reliability`.
- Suggested release title: `Reproducible evidence for KAN training consistency
  and Grid-Update stability in transformer-temperature forecasting`.
