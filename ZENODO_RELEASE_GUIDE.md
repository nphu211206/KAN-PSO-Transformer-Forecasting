# Zenodo Permanent DOI Archival & Release Guide

**Manuscript Title:** *PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting*  
**Target Journal:** *Neural Computing and Applications* (Springer Nature London, Q1)  
**Author:** Phu Toan Nguyen  
**ORCID iD:** [0009-0005-0529-2131](https://orcid.org/0009-0005-0529-2131)  
**GitHub Repository:** [https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting](https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting)  
**Permanent Archival DOI Placeholder:** `10.5281/zenodo.xxxxxxx`

---

## 1. Automated GitHub-to-Zenodo Integration

1. Sign in to [Zenodo](https://zenodo.org) via **GitHub (`nphu211206`)** and link ORCID `0009-0005-0529-2131`.
2. In Zenodo **GitHub settings** (`https://zenodo.org/account/settings/github/`), toggle `nphu211206/KAN-PSO-Transformer-Forecasting` to **ON**.
3. Commit and push the verified release from `github_release/` to `forecasting main`:
   ```bash
   cd github_release
   git add -A
   git commit -m "release(ncaa-v1.0.0): verified Option A benchmark, 600 DPI figures, and manuscript package"
   git push forecasting main
   git tag -f v1.0.0
   git push -f forecasting v1.0.0
   ```
4. Optionally publish a formal GitHub Release for tag `v1.0.0` (title: `PSO-TCWavKAN v1.0.0: Initial Scientific Release for NCAA`).
5. Within 30–60 seconds, Zenodo mints the permanent **Concept DOI** and **Version DOI**.

---

## 2. Final DOI Synchronization in Manuscript & Docs

Upon receiving the real numeric Zenodo ID (e.g., `10.5281/zenodo.14892015`):
1. Replace `zenodo.xxxxxxx` across `paper_ncaa.tex`, `paper_v91_FINAL.md`, `COVER_LETTER_NCAA.md`, `DECLARATIONS_NCAA.md`, `build_submission_docs.py`, and `README.md`.
2. Re-run:
   ```bash
   python generate_paper_v91_docx.py
   python build_submission_docs.py
   python verify_latex_package.py
   ```

