# Zenodo Permanent DOI Archival & Release Guide

**Manuscript Title:** *PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting*  
**Target Journal:** *Neural Computing and Applications* (Springer Nature London, Q1)  
**Author:** Phu Toan Nguyen  
**ORCID iD:** [0009-0005-0529-2131](https://orcid.org/0009-0005-0529-2131)  
**Verified GitHub Repository:** [https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting](https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting)  
**Permanent Archival DOI Scheme:** `10.5281/zenodo.XXXXXXX`

---

## 1. Overview and Springer Nature Policy Compliance

In strict compliance with **Springer Nature Code and Data Availability Guidelines**, all open-source algorithms, pre-trained weights, hyperparameter configuration dictionaries, and experimental logs must be permanently archived with a persistent, citable Digital Object Identifier (DOI). 

While GitHub provides active source control and code collaboration, repositories can be edited or deleted. **Zenodo** (hosted by CERN and supported by the European Commission's OpenAIRE program) guarantees 100% immutable, permanent preservation on long-term tape storage for a minimum of 20 years.

---

## 2. Automated Step-by-Step GitHub-to-Zenodo Integration

The recommended and standard route for code archival is Zenodo's automated GitHub release webhook.

### Step 1: Connect GitHub Account to Zenodo
1. Navigate to [https://zenodo.org](https://zenodo.org).
2. Click **Log in** in the top-right corner.
3. Select **Log in with GitHub** and authorize Zenodo using your GitHub credentials (`nphu211206`).
4. Ensure your ORCID (`0009-0005-0529-2131`) is linked in your Zenodo profile settings.

### Step 2: Enable Webhook for Repository
1. In Zenodo, navigate to the **GitHub** tab under your account menu: [https://zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/).
2. Locate the repository: `nphu211206/KAN-PSO-Transformer-Forecasting`.
3. Toggle the switch to **ON** (Enabled).  
   *Zenodo now automatically registers a webhook for all future Git releases.*

### Step 3: Stage and Push Clean Release Artifacts
Ensure the local staging directory `github_release/` is pushed to the GitHub repository:
```bash
git add .
git commit -m "feat: complete master submission package and clean reproduction suite"
git push origin master
```

### Step 4: Publish a Formal GitHub Release
1. In your GitHub repository, click on **Releases** -> **Draft a new release**.
2. Set **Tag version** to `v1.0.0` (target: `master`).
3. Set **Release title** to `PSO-TCWavKAN v1.0.0: Initial Scientific Release for NCAA`.
4. Attach the benchmark results archive: `ncaa_option_a_results.zip`.
5. Click **Publish release**.

### Step 5: Verify Automated Zenodo Minting
1. Return to [https://zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/).
2. Within 30–60 seconds, Zenodo automatically ingests the release snapshot and mints:
   - A **Concept DOI** (representing all versions).
   - A **Version DOI** (representing specifically `v1.0.0`, e.g., `10.5281/zenodo.14892015`).
3. Under the Zenodo record, review metadata:
   - **Title**: `PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting`
   - **Creator**: `Nguyen, Phu Toan` (ORCID: `0009-0005-0529-2131`)
   - **License**: `MIT License`

---

## 3. Alternative: Direct Manual Archive Deposit

If preferred, you may deposit the archival bundle manually without waiting for a GitHub release:

1. Compress the audited clean release directory:
   ```bash
   python -c "import shutil; shutil.make_archive('PSO_TCWavKAN_v1.0.0_Zenodo_Archive', 'zip', 'github_release')"
   ```
2. Navigate to [https://zenodo.org/deposit/new](https://zenodo.org/deposit/new).
3. Upload `PSO_TCWavKAN_v1.0.0_Zenodo_Archive.zip` and `ncaa_option_a_results.zip`.
4. Fill in the required metadata:
   - **Upload type**: Software
   - **Title**: *PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal Dynamics Forecasting*
   - **Authors**:
     - Family Name: `Nguyen`
     - Given Name: `Phu Toan`
     - Affiliation: `Faculty of Information Technology, Hanoi University of Business and Technology (HUBT), Hanoi, Vietnam`
     - ORCID: `0009-0005-0529-2131`
   - **Description**: Verbatim Abstract from `paper_ncaa.tex`.
   - **Keywords**: `Kolmogorov–Arnold Networks (KAN)`, `Continuous Wavelet Transform`, `Particle Swarm Optimization`, `Power Transformer Thermal Dynamics`, `Reversible Instance Normalization`, `Long-Term Time Series Forecasting`.
   - **Related Identifiers**:
     - Identifier: `https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting`
     - Relation: `isSupplementTo` / `isCompiledBy`
5. Click **Save draft** -> **Publish**. Zenodo assigns the permanent DOI immediately.

---

## 4. Manuscript Synchronisation

During initial manuscript submission to *Neural Computing and Applications*, standard Springer Nature practice allows:
```latex
\subsection*{Code Availability}
The complete PyTorch source code, CLPSO optimization routines, model configuration files, and pre-trained checkpoints supporting the findings of this study are available in an open-access GitHub repository at \url{https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting} and permanently archived via Zenodo with persistent DOI \url{https://doi.org/10.5281/zenodo.xxxxxxx}.
```
Upon editorial acceptance and prior to final proof galley sign-off, replace `zenodo.xxxxxxx` with the exact numerical identifier issued by Zenodo.

