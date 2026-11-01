# Submission Declarations – Electrical Engineering (Archiv für Elektrotechnik, Springer Nature)

**Manuscript Title:** PSO-TCWavKAN: A Swarm-Optimized Temporal Convolutional Wavelet Kolmogorov-Arnold Network with Multi-Kernel Trend Decomposition and Asymptotic Linear Structural Recovery for Resilient Power Transformer Thermal Dynamics Forecasting  
**Author:** Nguyen Phu Toan  
**Affiliation:** Department of Information Technology, Hanoi University of Business and Technology, 29A, Lane 124 Vinh Tuy Street, Vinh Tuy Ward, Hai Ba Trung District, Hanoi, Vietnam  
**Corresponding Author:** Nguyen Phu Toan (Email: nphu211206@gmail.com | ORCID: 0009-0005-0529-2131)  

---

**Availability of data and materials:** The ETTh1 and ETTh2 benchmark datasets used in this study are publicly available components of the canonical Electricity Transformer Temperature benchmark established by Zhou et al. (Informer repository: https://github.com/zhouhaoyi/ETDataset). No private, proprietary, or confidential data were used. The release includes the aggregate benchmark and ablation CSV matrices, the statistical reproduction script, the Table 7 parameter-count reproduction script for the four released models, and the Figure 3 source-data JSON and plotting script; the Figure 3 diagnostic script additionally requires the public efficient-kan package and the ETT CSV files. Per-seed raw training logs, model checkpoints, per-seed noise trajectories, predecessor-diagnostic prediction arrays for Figures 5, 6, 9, and 10, and the Table 7 profiling logs were generated in Kaggle Cloud environments and are available from the author on reasonable request.

**Code availability:**
- **Project Name:** KAN-PSO-Transformer-Forecasting (PSO-TCWavKAN v9.1)
- **Project Home Page:** https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting
- **Release Contents:** v9.1 implementation, aggregate result CSV matrices, `reproduce_statistical_tests.py`, `reproduce_table7_params.py`, `plot_grid_dilemma_v2.py`, `figures/dilemma_v2_data.json`, and publication figures
- **Operating System:** Platform independent (verified on Windows 11 and Linux Ubuntu; experiments executed on Kaggle Cloud with an NVIDIA Tesla T4 GPU)
- **Programming Language:** Python 3.10 – 3.13
- **Primary Dependencies:** PyTorch >= 2.0.0, NumPy, SciPy, scikit-learn, pandas, Matplotlib
- **License:** MIT License for the proposed model architecture and benchmark code
- **Restrictions on Use:** None for academic or commercial research, subject to the MIT License and the ETT dataset terms

**Competing interests:** The author declares that there are no financial, professional, personal, or non-financial competing interests directly or indirectly related to the work submitted for publication.

**Funding:** The author received no specific grant or external financial support from any funding agency in the public, commercial, or not-for-profit sectors for this research.

**Authors' contributions:** Nguyen Phu Toan (NPT): conceptualization, mathematical formulation, software implementation, experiment execution, formal analysis, validation, data curation, visualization, writing – original draft preparation, writing – review and editing, and project administration. The author read and approved the final manuscript.

**Acknowledgements:** Computational resources for the 16-configuration benchmark, the 192 A1–A4 ablation training runs, and the exploratory noise diagnostic were provided by the Kaggle Cloud GPU environment (NVIDIA Tesla T4). A5 is the five-seed benchmark reference rather than a separate three-seed ablation run; the manuscript and repository document this provenance.

**Ethics approval and consent to participate:** Not applicable. This study uses open-source physical sensor telemetry and involves no human participants, tissue samples, or animal experiments.

**Consent for publication:** Not applicable. The manuscript contains no individual person's personal identifying data, images, or details in any form.
