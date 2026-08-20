# Reproducibility Package

This package reproduces the frozen evidence and derived publication artifacts.
The JSON records are authoritative; CSV, Markdown tables and figures are
derived outputs.

**Source repository:** https://github.com/nphu211206/kan-grid-update-reliability  
**Archival DOI:** [ZENODO_DOI] — to be inserted after Zenodo deposit  
**Release tag:** v1.0.0

## Environment recorded by canonical runs

- Python 3.12.13
- PyTorch 2.10.0+cu128, CUDA 12.8
- NumPy 2.0.2, pandas 2.3.3
- NVIDIA Tesla T4

The exact source and dataset identities are recorded in
`EXPERIMENT_REGISTRY.md` and checked by `audit_evidence.py`.

## Verification commands

Preferred full QA command on Windows:

```powershell
.\run_release_qa.ps1 -PythonPath C:\path\to\python.exe
```

Use `-RebuildSubmissionDocx` only when Pandoc and python-docx are available.
The expanded commands are:

```powershell
python audit_evidence.py
python audit_k09.py canonical_uniform_best_diagnostics.zip
python audit_k10.py canonical_multihorizon_baselines_v2.zip
python analysis/analyze_k10_multihorizon.py canonical_multihorizon_baselines_v2.zip
python audit_manuscript.py
python publication_tables.py
python build_public_release_manifest.py
python analysis.py
python analysis/parse_jeas_archived.py
python -m unittest discover -s tests -v
python -m py_compile audit_evidence.py audit_manuscript.py publication_tables.py analysis.py analysis/parse_jeas_archived.py
```

The completed K10 archive is audited with:

```powershell
python audit_k10.py canonical_multihorizon_baselines_v2.zip
```

K10 is part of the frozen evidence only after this dedicated audit passes.

The audit verifies record counts, completed status, archive SHA-256/CRC,
source hashes, dataset hashes, canonical 40-condition coverage and optimizer
reset event semantics. Do not mix the pilot `pilot_60_20_20_v1` and canonical
`ett_canonical_12_4_4_v1` protocols in a single comparison.

Dataset hashes refer to the exact LF-delimited bytes used on Kaggle. The local
integrity check accepts a Windows CRLF materialization only after converting
CRLF to LF; the protocol test separately verifies row count, columns, hourly
continuity, duplicate timestamps and blank values. This prevents an operating-
system newline conversion from being mistaken for a data change.

The evidence audit and publication-table generator use only the Python standard
library. Figure regeneration additionally requires pandas, NumPy and matplotlib.
On systems without that stack, `set_figure_dpi.ps1` can normalize publication
DPI metadata without changing the frozen figure pixels.

`publication_tables.py` also generates the 40-row
`analysis/publication/k09_per_seed_supplement.csv`. The public-release staging
allowlist and file hashes are generated in `PUBLIC_RELEASE_MANIFEST.json`.

## Evidence manifest

- Total frozen records: 380 (K08, K09 and K10 remain separate campaigns)
- Historical canonical archive: `canonical_confirmation_results.zip`
- Historical canonical archive SHA-256:
  `7F0A3DA091857B0CAF8BD72DAD1850EA2A3A1A005079B05D639D540FA48DA378`
- Canonical source combined SHA-256:
  `bd42380f16f1cb8e084c87a5515fa9160fe45aee01e3c1dbf6e15b718b6a1516`
- ETTh1 dataset SHA-256:
  `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
- ETTh2 dataset SHA-256:
  `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`
- K09 archive: `canonical_uniform_best_diagnostics.zip`
- K09 archive SHA-256:
  `0685DEAE0D453FE2135AB8855FC68CB635D53663F09699F1FA537A014D124004`
- K09 source combined SHA-256:
  `138ddf1b1feffc4fa03cb75c3e6746b67ace7c401cf397f3503f7d8635aa8a49`
- K10 v1 input was withdrawn after its startup log exposed 360 enumerated
  tasks; its partial output is not evidence.
- Corrected K10 v2 download-safe input bundle:
  `KAGGLE_10_CANONICAL_MULTIHORIZON_BASELINES_V2_DOWNLOAD_SAFE.py`
- K10 input bundle SHA-256:
  `ED3639ABF7310DED1C5EB092004FA64E4419C5D5139846CE92A8FA31092DA44F`
- K10 archive: `canonical_multihorizon_baselines_v2.zip`
- K10 archive SHA-256:
  `BD3BEA46869C555F89243C954533B827A5151F6F8BD006BAF9693A321281D281`
- K10 executed source combined SHA-256:
  `edb30890ea8ba8097e6072b620f6c13c5cef9d26281d094b329cdcbd9b700fb5`

## Reproduction scope

The package reproduces the audit, summary tables and figures from frozen
records. Re-running model training requires the archived datasets, the pinned
source snapshot and a CUDA-capable environment matching the recorded versions.
