# Cover Letter — Journal of Engineering and Applied Science

Dear Editor,

I submit "Training Consistency and Grid-Update Stability in Metaheuristic-Tuned
Kolmogorov-Arnold Networks for Transformer-Temperature Forecasting" for
consideration as a Research article in the *Journal of Engineering and Applied
Science*.

**Why this manuscript is suitable for JEAS.**
The study addresses a practical engineering reliability question within the
journal's stated scope of electrical engineering, machine intelligence and
optimization: whether metaheuristic hyperparameter optimization (HPO) makes
Kolmogorov-Arnold Networks (KANs) reliable enough for transformer oil-temperature
forecasting, and when KAN Grid Updates destabilize training rather than improve
it. The Electricity Transformer Temperature (ETT) benchmark is a widely used,
publicly verifiable test bed for condition-monitoring and maintenance-planning
research. The engineering interpretation is expressed in recorded oil-temperature
units and translated into explicit deployment safeguards, consistent with the
application-oriented reporting standards demonstrated by published JEAS articles
on hybrid machine-learning methods.

**What the study contributes.**
The contribution is a reproducible reliability framework, not a claim that KAN
universally outperforms established baselines. The manuscript provides: (i) a
provenance-tracked audit of historical KAN HPO across six search algorithms,
(ii) a controlled training-consistency intervention showing a 35.5% mean MSE
reduction when HPO and final training share the same objective, (iii) paired
Grid Update schedule and Adam-state reset interventions with event-level
displacement logging, (iv) a controlled negative result for KAN residual
learning, (v) a five-seed canonical confirmation on ETTh1 and ETTh2, (vi) a
uniform-checkpoint diagnostic campaign with saved predictions, residual
analysis and compute measurements, and (vii) a 120-run canonical multi-horizon
comparison against DLinear and PatchTST at horizons 96, 192, 336 and 720.
All 380 experiment records are frozen and integrity-checked; every manuscript
claim is mapped to a primary JSON evidence source and an independent SHA-256
audit. The reproducibility package will be archived at [ZENODO_DOI] and the
living source repository is at https://github.com/nphu211206/kan-grid-update-reliability.

**Scope of claims.**
DLinear has lower mean test MSE than corrected KAN in all eight canonical
dataset-horizon cells and in 38 of 40 K10 paired comparisons; the gap narrows
at ETTh2/720. The paper reports this result honestly rather than filtering it.
Repeated Grid Updates produce high variance and catastrophic outcomes; clearing
Adam state alone is not a sufficient remedy. These are engineering-relevant
failure modes, and identifying them reproducibly is the paper's primary value.
The study is intentionally univariate and bounded to two hourly ETT datasets.
Feature-importance plots are not included because the univariate OT predictor
cannot support that inference; the corresponding visual evidence covers
actual-versus-predicted curves, residual densities, empirical error CDFs and
per-step RMSE growth across all 96 forecast steps.

**Ethical and policy declarations.**
The study uses public benchmark data and involves no human participants, animals
or private data. No external funding was received. The author declares no
competing interests. OpenAI Codex was used as an assistive coding and drafting
tool; all study design decisions, numerical claims and final wording were
reviewed against frozen evidence by the author, who retains full responsibility.
This manuscript has not been published or submitted elsewhere.

The author has read and approved this submission.

Sincerely,

Nguyen Phu Toan  
Department of Information Technology  
Hanoi University of Business and Technology  
29A, Lane 124 Vinh Tuy Street, Vinh Hung Ward, Hanoi, Vietnam  
nphu211206@gmail.com  
ORCID: 0009-0005-0529-2131
