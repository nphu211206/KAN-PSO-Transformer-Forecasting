# Neural Computing and Applications (NCAA) Submission Checklist

**Manuscript:** *PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting*  
**Venue:** *Neural Computing and Applications* (Springer Nature, ISSN 0941-0643 / 1433-3058)  
**Publication Route:** Traditional Subscription Model (**0-APC Free Publishing**)  
**Author:** Phu Toan Nguyen (`nphu211206@gmail.com`, ORCID: `0009-0005-0529-2131`)

---

## Editorial Manager Upload Files

### Route A — Recommended (LaTeX Source Submission)
- `paper_ncaa.tex` (Document class `\documentclass[pdflatex,sn-mathphys-num,referee,lineno]{sn-jnl}`, balanced braces/environments, zero `\bibliographystyle`)
- `sn-bibliography.bib` (76 verified BibTeX keys, 100% bijective with `\cite{}` calls, clean DOIs)
- Figures at >= 600 DPI:
  - `figures/fig1_architecture.png`
  - `figures/fig2_wavelet_basis.png`
  - `figures/fig3_forecast_curves.png`
  - `figures/fig4_residual_boxplots.png`
  - `figures/fig5_pso_convergence.png`
  - `figures/fig6_benchmark_comparison.png`
  > **Important Editorial Manager Note:** `paper_ncaa.tex` configures `\graphicspath{{figures/}{./}}` and references bare filenames (`fig1_architecture.png`–`fig6_benchmark_comparison.png`), so both hierarchical Overleaf ZIP compilation (`figures/` subfolder) and flattened single-directory Editorial Manager uploads build identically without manual path edits.
- `COVER_LETTER_NCAA.docx` (2 balanced pages, clean typography)
- `DECLARATIONS_NCAA.docx` (1 page, all mandatory Springer Nature sections)

### Route B — Word Manuscript Backup (Optional / If Requested)
- `paper_v91_FINAL.docx` (22 pages, continuous line numbering, running header/footer, booktabs tables, embedded 600 DPI figures)

---

## Pre-Submission Technical Audit Status

- [x] **100% Real Kaggle Option A Ground Truth:** 16 configs × 6 models × 3 seeds (`ddof=1`), 4-variant ablation, CLPSO convergence/sensitivity logs, Tesla T4 edge profile (`PyTorch 2.11.0+cu130`).
- [x] **LaTeX & BibTeX Hygiene:** `ingest_kaggle_results.py --verify_only` and `verify_latex_package.py` pass with 0 errors (`76/76` bijective keys).
- [x] **Automated Statistical & Tensor Tests:** `pytest tests/test_challenger_m6_benchmarks.py` (`7/7` passed across all 144 NPZs and Wilcoxon–Holm exact tests).
- [x] **Visual QA:** All 6 figures (600 DPI) and LibreOffice-rendered PDF pages of `paper_v91_FINAL.docx`, `COVER_LETTER_NCAA.docx`, and `DECLARATIONS_NCAA.docx` inspected for zero text clipping, overlaps, orphan signatures, or raw LaTeX leaks.
- [ ] **GitHub Push & Zenodo Tag (`v1.0.0`):** Push `github_release/` to `KAN-PSO-Transformer-Forecasting` and mint live Zenodo DOI replacing `zenodo.xxxxxxx`.
- [ ] **Overleaf Compile Check:** Quick compile check inside Springer Nature `sn-jnl` template prior to final Editorial Manager PDF sign-off.
