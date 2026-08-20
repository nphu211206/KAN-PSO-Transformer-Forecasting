param(
    [string]$PythonPath = "python",
    [switch]$RebuildSubmissionDocx
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

function Invoke-Step([string]$Name, [scriptblock]$Command) {
    Write-Host "[QA] $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

try {
    Invoke-Step "Compile audit and publication scripts" {
        & $PythonPath -m py_compile audit_evidence.py audit_k09.py audit_k10.py audit_manuscript.py audit_submission_docx.py audit_public_release.py publication_tables.py build_kaggle_bundle.py build_public_release_manifest.py stage_public_release.py analysis\analyze_k09_diagnostics.py analysis\analyze_k10_multihorizon.py analysis\checkpoint_contract_sensitivity.py tests\test_publication_boundaries.py tests\test_k10_protocol.py tests\test_k10_audit.py
    }
    Invoke-Step "Regenerate publication tables" { & $PythonPath publication_tables.py }
    Invoke-Step "Generate checkpoint-contract appendix" { & $PythonPath analysis\checkpoint_contract_sensitivity.py }
    Invoke-Step "Build public-release staging manifest" { & $PythonPath build_public_release_manifest.py }
    if (Test-Path "release_staging_v13") {
        Invoke-Step "Audit existing public-release staging" { & $PythonPath audit_public_release.py release_staging_v13 }
    }
    Invoke-Step "Run regression tests" { & $PythonPath -m unittest discover -s tests -v }
    Invoke-Step "Audit frozen evidence" { & $PythonPath audit_evidence.py }
    Invoke-Step "Audit K09 artifacts" { & $PythonPath audit_k09.py canonical_uniform_best_diagnostics.zip }
    Invoke-Step "Audit K10 artifacts" { & $PythonPath audit_k10.py canonical_multihorizon_baselines_v2.zip }
    Invoke-Step "Audit manuscript claims" { & $PythonPath audit_manuscript.py }

    if ($RebuildSubmissionDocx) {
        if (-not (Get-Command pandoc -ErrorAction SilentlyContinue)) {
            throw "Pandoc is required when -RebuildSubmissionDocx is used"
        }
        Invoke-Step "Build submission DOCX" {
            pandoc paper_draft.md -o paper_submission.docx --resource-path=.
        }
        Invoke-Step "Format submission DOCX" {
            & $PythonPath format_submission_docx.py paper_submission.docx paper_submission.docx
        }
        Invoke-Step "Strip DOCX comments" {
            & $PythonPath strip_docx_comments.py paper_submission.docx
        }
    }

    Invoke-Step "Audit submission DOCX" { & $PythonPath audit_submission_docx.py paper_submission.docx }
    Write-Host "[QA] PASS: release staging, evidence, tests, manuscript and DOCX are consistent"
}
finally {
    Pop-Location
}
