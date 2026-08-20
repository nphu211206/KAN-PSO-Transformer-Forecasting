# Training Consistency and Grid-Update Stability in Metaheuristic-Tuned Kolmogorov-Arnold Networks for Transformer-Temperature Forecasting

**Author:** Nguyen Phu Toan

**Affiliation:** Department of Information Technology, Hanoi University of
Business and Technology, 29A, Lane 124 Vinh Tuy Street, Vinh Hung Ward, Hanoi,
Vietnam

**Corresponding author:** Nguyen Phu Toan, nphu211206@gmail.com

**ORCID:** 0009-0005-0529-2131

## Abstract

Kolmogorov-Arnold Networks (KANs) introduce spline-specific configuration and
grid-adaptation choices that complicate model selection for engineering time
series. This study examines whether metaheuristic hyperparameter optimization
(HPO) yields a reliable KAN advantage and when Grid Update destabilizes
training. The evidence comprises 380 completed experiment records on the
univariate oil-temperature target of ETTh1 and ETTh2. Historical HPO, controlled
training-consistency corrections, Grid Update schedules and Adam-state
interventions are kept separate from three five-seed campaigns using the
official hourly ETT 12/4/4-month protocol. The final campaign compares DLinear,
PatchTST and one frozen corrected-KAN configuration at horizons 96, 192, 336
and 720.

Under the pilot protocol, HPO-consistent training reduces the corrected KAN
mean ETTh1/h96 test MSE from 0.175458 to 0.113224, but DLinear remains lower at
0.101641. Under uniform validation-best checkpointing, DLinear is lower than
corrected KAN in all ten h96 dataset-seed comparisons. Across the canonical
multi-horizon campaign, DLinear has the lowest mean MSE in all eight
dataset-horizon cells and is lower in 38 of 40 paired comparisons with corrected
KAN; the separation narrows at ETTh2/h720. Moderate Grid Update has a worse
mean than no update on both datasets, and clearing Adam state after an update
does not provide reliable stabilization. Corrected KAN also requires about
7.2-7.6 times the DLinear training time in the h96 campaign. The evidence does
not support universal KAN or metaheuristic superiority. It instead identifies
validation-consistent training as necessary and repeated Grid Update as a
deployment risk under the tested configurations.

**Keywords:** Kolmogorov-Arnold Networks; time-series forecasting; hyperparameter
optimization; B-splines; Grid Update; reproducibility; transformer temperature

## 1. Introduction

Forecasting the operating temperature of electrical transformers is an
engineering task with direct relevance to condition monitoring, maintenance
planning and load management. The Electricity Transformer Temperature (ETT)
benchmark provides hourly and minute-level measurements of oil temperature and
external load variables and has become a standard test bed for long-horizon
forecasting following its release with Informer [3]. Its public and
reproducible nature makes it suitable for a study
that emphasizes both model performance and experimental reliability.

Kolmogorov-Arnold Networks (KANs) [1,2] offer a different parameterization from an
ordinary multilayer perceptron. Rather than applying a fixed activation at a
node and learning scalar edge weights, a KAN learns univariate functions on
edges. In the efficient-kan implementation used here, those functions are
represented by B-spline coefficients and a data-dependent grid. The grid can be
updated during training so that spline support follows the activation
distribution. This adaptivity is attractive, but it changes the effective
parameter geometry while Adam is accumulating first- and second-moment
estimates.

This interaction creates the practical problem studied in this paper. A grid
update can improve local representation, yet frequent updates can disrupt
optimization and produce large validation variance. We call this tension the
Grid Update Dilemma. The question is not whether a grid should always be
updated, but under which training conditions an update is beneficial and when
it becomes a source of instability.

Metaheuristic HPO is a natural tool for exploring KAN-specific discrete and
continuous choices. However, the existence of a large search space does not
imply that a metaheuristic produces a better forecasting model. A configuration
can score well under a proxy objective and then be trained inconsistently in
the final evaluation. A best single run can also conceal seed variance. These
risks are especially important for engineering applications where a failure
mode matters as much as a mean error.

This paper therefore asks five questions:

1. Does making final KAN training consistent with the HPO objective materially
   change forecasting performance?
2. How does Grid Update frequency affect validation and test stability?
3. Does clearing Adam state after a Grid Update provide a reliable remedy?
4. Do the conclusions survive the official canonical ETT split rather than a
   development-oriented 60/20/20 split and across standard forecast horizons?
5. How sensitive are the conclusions to the checkpoint-selection contract, and
   what do prediction-level and compute diagnostics reveal?

The contributions are deliberately empirical. We provide (i) a provenance-
tracked audit of historical KAN HPO, (ii) a controlled training-consistency
study, (iii) paired Grid Update and optimizer-state interventions, (iv) a
controlled negative result for DLinear-plus-KAN residual learning, (v) a
five-seed canonical confirmation on two hourly ETT datasets, and (vi) a
separately pre-specified uniform-checkpoint diagnostic campaign with saved
predictions, original-unit residual analysis and compute measurements, and
(vii) a 120-run canonical multi-horizon comparison against DLinear and
PatchTST. The contribution
is not a claim that KAN universally outperforms linear forecasting.

## 2. Related Work

### 2.1 KAN and spline parameterization

KAN was introduced by Liu et al. [1] as a neural architecture motivated by the
Kolmogorov-Arnold representation theorem. Subsequent work has extended the
architecture, proposed scalable implementations and studied its expressivity,
training behavior and applications in scientific and engineering tasks [2]. The
efficient-kan implementation used in this study is a pinned implementation of
the spline-based layer [15] and is not replaced by a new spline formulation.

The present work is related to, but distinct from, papers that combine KAN with
PSO or other optimization methods for a particular application. PSO-optimized
GRU-KAN has been studied for oil-well production forecasting [13], while
QAMO-PSO has been proposed for tailoring KAN parameters [14]. Those studies
establish that KAN-metaheuristic combinations are an active research area. Our
focus is narrower: training-objective consistency, Grid Update stability and
reproducibility under controlled forecasting protocols.

### 2.2 ETT forecasting and linear baselines

Informer [3] and later long-horizon forecasting studies established common ETT
evaluation conventions. PatchTST [4] uses patching to improve long-range temporal
modeling, while DLinear [5] showed that a simple decomposition followed by linear
heads can outperform more elaborate architectures in many settings.
iTransformer [6], TimesNet [7] and Autoformer [12] further illustrate the range
of transformer and temporal-modeling designs evaluated on ETT. These
baselines are important here because a KAN improvement is meaningful only when
compared with a strong, same-protocol non-KAN model.

Forecast evaluation also depends on how observations are ordered and assigned
to development and test periods. Out-of-sample forecasting guidance treats the
test period as future information that must remain unavailable during model
selection [20]. Cross-validation can be valid for autoregressive prediction
only under assumptions about residual dependence and the construction of the
training folds [21]. More recent best-practice guidance likewise emphasizes
temporal ordering, scale-aware metrics, explicit forecast horizons and baseline
comparison [22]. We therefore use chronological splits, fit the scaler only on
training rows, keep every forecast target inside its assigned split and report
pilot and canonical protocols separately. Random fivefold cross-validation,
although common in some engineering regression papers, would answer a different
question and could leak future temporal structure into model development.

### 2.3 Metaheuristic HPO and reporting

Particle Swarm Optimization [8], Grey Wolf Optimization [10], Differential
Evolution [9], Genetic Algorithms and Whale Optimization [11] are
population-based search methods
that can handle mixed discrete/continuous configuration spaces. Their use in
this study is descriptive and exploratory: the historical campaign identifies
a configuration landscape, but it is not used to claim statistically
significant superiority of one optimizer over another. The historical campaign
has three runs per method and unequal evaluation counts in some records, so
best-run and optimizer-ranking claims are intentionally excluded.

HPO is not only the choice of a search algorithm. It also includes the search
space, evaluation budget, validation objective, stochastic replication and the
rule used to select the final checkpoint [23,24]. Consequently, a configuration
found with validation-best fitness should not be assessed under an unreported
final-epoch rule and treated as the same optimization contract. This principle
motivates the training-consistency intervention and the explicit disclosure of
the checkpoint implementation deviation in the canonical campaign. It also
limits the historical metaheuristic evidence to landscape description: because
budgets were not perfectly equal, the observed search outcomes cannot identify
an intrinsically superior optimizer.

### 2.4 Comparable JEAS engineering studies

Recent JEAS papers show the journal's application-oriented expectations for
hybrid machine-learning studies. A PSO-SVR pavement study reports fivefold
cross-validation, multiple tree baselines, residual diagnostics, variable
importance and three-dimensional interaction plots [16]. A related PSO-GBM
study combines a workflow diagram, correlation analysis, hyperparameter
sensitivity and comparison against linear, ensemble and neural baselines [17].
For engineering time series, an MVMD-GNN-Transformer study includes component
ablation, noise robustness, attention-weight visualization and testing on
multiple datasets [18]. The closest KAN forecasting article compares seven
models, reports training and inference cost, repeats experiments five times and
ablates both the KAN layer and its error-correction component [19].

These papers establish useful reporting patterns, but they answer different
questions. Feature-importance plots are appropriate for their multivariate
predictors, whereas the present study deliberately forecasts a single OT
channel. Our corresponding evidence is therefore not post-hoc feature ranking;
it is seed-level stability, intervention logging, protocol separation and
reproducible provenance. This distinction prevents visual conventions from
being copied into a setting where they would not be scientifically meaningful.

## 3. Methodology

![**Fig. 1. Study workflow and evidence boundary.** Data identity, protocol
separation, HPO-consistent training, causal interventions and frozen claim
generation are shown as distinct stages.](figures/study_workflow.png)

### 3.1 Forecasting task

All modern experiments use the ETT `OT` column as a univariate target. Given a
history window x(t-L+1:t), the model predicts the next P standardized values
y(t+1:t+P). All canonical settings use L = 96; K08/K09 use P = 96, while K10
uses P in {96, 192, 336, 720}.
MSE and MAE are reported in standardized space, with the scaler fitted only on
training rows.

The project contains two explicit protocols. The pilot protocol,
`pilot_60_20_20_v1`, uses chronological 60/20/20 rows and does not share
history across split boundaries. The canonical protocol,
`ett_canonical_12_4_4_v1`, follows the official hourly ETT boundaries: train
rows [0, 8640), validation rows [8640, 11520) and test rows [11520, 14400),
with the preceding history window available at validation and test boundaries.
The protocols are never combined in a single comparison table.

**Table 1. Experimental protocol boundaries.**

| Protocol | Training boundary | Validation boundary | Test boundary | Seeds | Purpose |
|---|---|---|---|---:|---|
| `pilot_60_20_20_v1` | First 60% of rows | Next 20% | Final 20% | 1000-1002 | Development and controlled intervention |
| `ett_canonical_12_4_4_v1` | Rows 0-8639 | Rows 8640-11519 | Rows 11520-14399 | 1000-1004 | Pre-specified canonical campaign |

**Table 2. ETT data identity in the canonical campaign.**

| Dataset | Rows | Sampling | Recorded variables | Target | Train/val/test windows | SHA-256 prefix |
|---|---:|---|---|---|---|---|
| ETTh1 | 17420 | Hourly | HUFL, HULL, MUFL, MULL, LUFL, LULL, OT | OT | 8449 / 2785 / 2785 | `f18de3ad269c` |
| ETTh2 | 17420 | Hourly | HUFL, HULL, MUFL, MULL, LUFL, LULL, OT | OT | 8449 / 2785 / 2785 | `a3dc2c597b92` |

An independent CSV audit found no missing numerical values, duplicate
timestamps or interruptions in the hourly sequence in either file. The
canonical protocol uses the first 14400 rows according to the published ETT
12/4/4-month convention; the remaining rows are not used to enlarge the test
set after model development. The canonical validation and test inputs include
the preceding 96 historical observations, but their forecast targets remain
strictly inside their assigned split.

The target distribution is non-stationary across these chronological blocks
(Fig. 2). ETTh1 mean OT decreases from 17.1283 in training to 4.8499 in test;
ETTh2 rises from 26.8720 in training to 36.1304 in validation and then falls to
14.4074 in test. This is a genuine deployment-like distribution shift rather
than a reason to refit the scaler. Refitting on validation or test observations
would leak future distribution information and is therefore prohibited.

![**Fig. 2. Canonical ETT target shift.** Hourly OT values and the fixed
train, validation and test boundaries are shown for both datasets.](figures/ett_canonical_distribution_shift.png)

### 3.2 KAN model

The KAN receives a flattened history window and predicts the horizon directly.
Each spline layer combines a base activation path and a B-spline path. The
implementation uses the pinned efficient-kan source with the documented
constructor settings `grid_range=[-4,4]` and
`enable_standalone_scale_spline=False`. Every modern record stores the source
hash, implementation hash, environment metadata, parameter count and protocol
manifest.

For layer l, the output coordinate j is a sum of learned univariate
edge functions,

$$
x_j^{(\ell+1)}=\sum_i \phi_{ji}^{(\ell)}\!\left(x_i^{(\ell)}\right).
$$

In the tested implementation, each edge function combines a base activation
and a spline expansion,

$$
\phi(x)=w_b b(x)+w_s\sum_{r=0}^{G+k-1}c_r B_{r,k}(x),
$$

where b is SiLU, G is the grid size, k is the spline order, B(r,k) is a
B-spline basis function and c(r) is a learned coefficient.
Unlike an ordinary linear edge weight, this parameterization gives each input-
output connection a locally supported nonlinear response. It also makes the
training geometry depend on knot placement, so grid adaptation is not merely
another scalar hyperparameter.

During `update_grid`, the implementation obtains adaptive knots from empirical
activation quantiles and uniform knots over the observed range. It blends them
as

$$
g_{new}=\epsilon_g g_{uniform}+(1-\epsilon_g)g_{adaptive},
$$

then recomputes spline coefficients by least squares so that the spline output
on the probe batch is approximately preserved. Preservation on that finite
batch does not imply preservation on all inputs, and the coefficient
reparameterization occurs while Adam's moments still refer to the previous
coordinates. This provides the mechanism motivating the Grid Update and
optimizer-state interventions.

For a test set containing N forecast elements, the reported metrics are

$$
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2,
\qquad
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|.
$$

Both metrics are computed after standardization with training-only statistics.
Consequently, values from the pilot and canonical split protocols are internally
comparable within a protocol but are not treated as interchangeable benchmark
scores.

For engineering interpretation, standardized test MSE is also converted to
root-mean-square error in the recorded OT units. For seed (j), this derived
quantity is

$$
\mathrm{RMSE}_{\mathrm{OT},j}=s_{\mathrm{OT}}\sqrt{\mathrm{MSE}_j},
$$

where s_OT is the population standard deviation learned by the
training-only scaler. No predictions are refitted or reevaluated for this
conversion.

**Table 3. Canonical h96 configurations and checkpoint rules.**

| Condition | Main configuration | Trainable parameters | Epochs | Executed checkpoint rule |
|---|---|---:|---:|---|
| DLinear | kernel 25; learning rate 0.001; no weight decay | 18624 | 30 | Validation-best |
| Corrected KAN | width 96-8-96; grid 3; spline order 2; batch 16; learning rate 0.0012595; regularization 9.403e-5; no Grid Update | 9216 | 30 | Validation-best |
| No-update KAN | width 96-10-96; grid 3; spline order 3; batch 32; learning rate 0.020633; no Grid Update | 13440 | 30 | Validation-best |
| Moderate-update KAN | same architecture as no-update; five updates during initial five epochs; retained Adam state | 13440 | 30 | Validation-best |

Table 3 describes K09, in which every condition uses the same
minimum-validation-MSE checkpoint rule. The earlier K08 campaign retained the
final epoch for DLinear, no-update KAN and moderate-update KAN while corrected
KAN used validation-best selection. K08 is retained as versioned
checkpoint-contract evidence and is never pooled with K09.

### 3.3 Historical HPO search

The historical campaign explores 14 configuration dimensions: grid size,
spline order, logarithmic grid epsilon, warmup ratio, update frequency,
hidden width, layer count, logarithmic learning rate, logarithmic weight decay,
batch-size index, spline noise scale, base scale, spline scale and logarithmic
regularization. WOA, PSO, GWO, DE, GA and Random Search each have three runs.
HPO fitness is the mean validation MSE over the final three proxy epochs,
called the Stabilization Sliding Window (SSW).

For configuration lambda and a proxy run ending at epoch T, the
historical objective is

$$
F_{\mathrm{SSW}}(\lambda)=\frac{1}{3}
\sum_{e=T-2}^{T}\mathrm{MSE}_{\mathrm{val}}^{(e)}(\lambda),
\qquad \lambda^{*}=\arg\min_{\lambda\in\Lambda}F_{\mathrm{SSW}}(\lambda).
$$

This objective rewards a stable validation tail rather than a favorable single
epoch. It does not make the six search algorithms directly rankable because
the historical campaign has only three runs per method and unequal realized
search budgets.

**Table 4. Historical Kaggle HPO search space.**

| Dimension | Type and bound | Decoding or role |
|---|---|---|
| `grid_size` | integer 3-12 | Number of spline intervals |
| `spline_order` | integer 2-4 | Piecewise polynomial order |
| `log_grid_eps` | continuous -3.0 to -0.3 | `grid_eps = 10^x` |
| `warmup_ratio` | continuous 0-0.6 | Fraction of the 10 HPO epochs eligible for updates |
| `update_every` | integer 0-4 | Grid Update interval inside warmup |
| `hidden_size` | integer 8-48 | Width of each hidden KAN layer |
| `n_layers` | integer 1-2 | Number of hidden layers |
| `log_lr` | continuous -4.0 to -1.5 | `learning_rate = 10^x` |
| `log_wd` | continuous -5.0 to -2.0 | `weight_decay = 10^x` |
| `batch_idx` | integer 0-3 | Maps to batch size 16, 32, 64 or 128 |
| `scale_noise` | continuous 0.05-0.5 | Spline initialization noise |
| `scale_base` | continuous 0.2-2.0 | Base-path initialization scale |
| `scale_spline` | continuous 0.2-2.0 | Spline-path initialization scale |
| `log_reg` | continuous -6.0 to -2.0 | `regularization = 10^x` |

The historical search is preserved as landscape evidence. Its local legacy
module uses different bounds from the 11-hour Kaggle script, and the historical
records report ten HPO epochs. These provenance differences are recorded and
are not silently collapsed into the later controlled protocol.

### 3.4 Training-consistency intervention

The historical final-training path omitted the regularization used during HPO
and selected the final epoch rather than the validation-best epoch. The
corrected direct KAN condition uses the historical WOA configuration, includes
the same KAN regularization, restores the lowest validation-MSE checkpoint and
clips gradients at norm 1.0. Stage 1 compares this condition with regularized
final-epoch training and with/without clipping.

The residual branch adds a KAN residual to a DLinear forecast. It includes a
validation gate and is treated as a negative-result experiment if the residual
is rejected at epoch zero. No test metric is used for this decision.

### 3.5 Grid Update and optimizer-state interventions

Four schedule families are represented in the pilot ablation: none,
conservative, moderate and aggressive. Each update records probe MSE before
and after the update, state displacement, grid displacement, spline parameter
displacement and optimizer-state summaries.

For an update event, the audit derives the immediate probe displacement
(Delta E, defined as post-update probe MSE minus pre-update probe MSE), the
Euclidean grid displacement, and the Euclidean spline-parameter displacement,
formally

$$
\Delta E_t=E_t^{+}-E_t^{-},\qquad
D_{g,t}=\lVert g_t^{+}-g_t^{-}\rVert_2,\qquad
D_{\theta,t}=\lVert \theta_t^{+}-\theta_t^{-}\rVert_2.
$$

These event-level quantities
do not claim global causality by themselves. They establish that an update was
executed, measure its immediate numerical effect on a fixed probe, and connect
that intervention to the later validation trajectory.

The causal campaign crosses moderate/aggressive schedules with two policies:
retain Adam state or clear `optimizer.state` immediately after a successful
Grid Update. The causal audit excludes epoch-zero events, where Adam has not
yet accumulated moments. Under the reset policy, parameters and the updated
grid are retained while Adam's moment state is cleared,

$$
(m_t,v_t)\leftarrow(0,0),\qquad
(\theta_t,g_t)\leftarrow(\theta_t^{+},g_t^{+}).
$$

The audit verifies 39 non-initial events per policy in both ETTh1 and ETTh2
campaigns.

### 3.6 Canonical confirmation and K09 checkpoint audit

K08 was pre-specified with fixed datasets, protocol, horizon, conditions and
five seeds. It contains DLinear, corrected KAN, no Grid Update and moderate Grid
Update with retained Adam state. There was no post-hoc architecture search or
test-driven variant addition. Its 40 records are stored separately from the
180-record pilot evidence.

One K08 execution deviation is material to interpretation. The preregistration
specified minimum validation MSE for selection, but the runner applied that
rule only to corrected KAN. DLinear, no-update KAN and moderate-update KAN used
their final epoch. The archive was preserved unchanged and reported under its
executed condition-specific contracts.

K09 was then separately pre-specified and frozen before execution to answer the checkpoint question rather
than repair K08 in place. It repeats the same 2 datasets x 4 conditions x 5
seeds under one minimum-validation-MSE rule. Each of its 40 records saves the
selected checkpoint, standardized and inverse-scaled predictions, inference
latency and peak GPU memory. K08 and K09 use different campaign and source
hashes and are analyzed as versioned evidence, never pooled replications.

### 3.7 Prediction-level and compute diagnostics

For seed (j), test origin (n) and forecast step (h), the original-unit
residual is

$$
e_{j,n,h}=\hat{y}^{\mathrm{OT}}_{j,n,h}-y^{\mathrm{OT}}_{n,h}.
$$

Forecast-step error is aggregated over all (N=2785) test origins within each
seed and then summarized across the five seeds,

$$
\mathrm{RMSE}_{j,h}=\sqrt{\frac{1}{N}\sum_{n=1}^{N} e_{j,n,h}^{2}}.
$$

Actual-versus-predicted curves use the fixed midpoint test origin
(n=1392), chosen by index before inspecting model errors, and show the mean
prediction plus/minus sample SD across all five seeds. Residual densities and
absolute-error empirical distributions use every saved test prediction from
all five seeds. Inference latency is the recorded milliseconds per sample;
peak GPU memory covers model training after loader construction and before
evaluation. These measurements characterize the Tesla T4 execution and are not
claimed as hardware-independent deployment benchmarks.

### 3.8 Canonical multi-horizon campaign

K10 is a separately pre-specified campaign whose protocol and source were
frozen before execution. It crosses ETTh1
and ETTh2 with horizons 96, 192, 336 and 720, five seeds and three conditions:
DLinear, corrected KAN and PatchTST. The lookback remains 96 and every model
uses the minimum-validation-MSE checkpoint from a 30-epoch budget. No Grid
Update condition is repeated in K10 because K09 remains the dedicated h96
mechanism campaign.

The corrected KAN transfers the validation-selected WOA configuration without
new horizon-specific tuning. PatchTST uses one fixed three-layer configuration
with patch length 16, stride 8, model width 64 and four attention heads. This
design tests fixed model contracts rather than allowing a new test-informed
search at each horizon. Each of the 120 tasks saves predictions in standardized
and recorded OT units, the selected checkpoint, inference latency, peak GPU
memory and source/artifact hashes.

### 3.9 Reproducibility and statistical policy

JSON records are primary evidence; summaries, figures and manuscript tables are
derived. Each campaign has a source combined SHA-256 and each record has dataset
and environment metadata. For three-seed campaigns we report descriptive mean,
sample SD, median, minimum and maximum. The K08 and K09 canonical campaigns
each use five seeds and report the same descriptive statistics plus paired seed
differences. We do
not add post-hoc significance claims to compensate for small stochastic sample
sizes.

The complete mapping from manuscript claims to primary JSON evidence is
maintained in the supplementary claim-evidence matrix. Archive, source and
dataset identities are independently checked before any publication artifact
is regenerated.

This reporting policy follows the broader recommendation that computational
experiments expose run-to-run variation, model-selection details and resource
budgets instead of presenting a favorable maximum as if it were a stable model
property [25]. Seed-level records are retained even when an update is
catastrophic. The five canonical seeds improve visibility of dispersion but do
not by themselves justify asymptotic tests or broad population claims.

### 3.10 AI-assisted tooling

OpenAI Codex was used as an assistive tool for code review, manuscript drafting,
formatting and consistency checks. It was not treated as an author. All study
design decisions, source-code changes, numerical claims, references and final
wording were reviewed against the frozen project evidence by the author, who
retains full responsibility for the work.

## 4. Experimental Results

### 4.1 Historical HPO landscape

The historical HPO fitness means are WOA 0.157116, PSO 0.157152, GWO 0.164565,
DE 0.175855, GA 0.178361 and Random Search 0.182008. These values describe the
configuration landscape under the historical proxy objective. They do not
establish that WOA is statistically superior: each method has only three runs,
the search is stochastic and historical evaluation budgets are not perfectly
uniform.

![**Fig. 3. Historical HPO validation-fitness distributions.** Each point is a
completed run; boxes summarize the three recorded runs per search method.](figures/hpo_fitness_three_runs.png)

### 4.2 Pilot baselines and Grid Update ablation

Under `pilot_60_20_20_v1` at h96, DLinear has mean test MSE
`0.101641 +/- 0.000031`, Linear `0.102633 +/- 0.000057`, NLinear
`0.105369 +/- 0.000041`, PatchTST `0.113584 +/- 0.003713`, RepeatLast
`0.129382`, KAN-WOA `0.175458 +/- 0.062499` and Vanilla KAN
`0.204331 +/- 0.015099`.

The fixed-configuration Grid Update ablation shows a narrow operating region:

**Table 5. Pilot Grid Update ablation at ETTh1/h96.**

| Schedule | Mean test MSE | Sample SD | Updates/run |
|---|---:|---:|---:|
| None | 0.182001 | 0.018613 | 0 |
| Conservative | 0.133661 | 0.009492 | 1 |
| Moderate | 1.175128 | 1.505580 | 5 |
| Aggressive | 1.908393 | 2.656384 | 10 |

One sparse update improves the fixed KAN condition, whereas repeated updates
produce very large variance and catastrophic runs. This is an empirical
stability result for the tested configuration, not a universal statement about
all KAN implementations.

![**Fig. 4. Pilot Grid Update ablation at ETTh1/h96.** Individual seed MSEs and
the mean are shown on a logarithmic vertical axis so that the catastrophic
moderate and aggressive outcomes remain visible.](figures/grid_update_ablation_h96.png)

### 4.3 Training consistency and residual negative result

The corrected direct KAN condition reduces mean ETTh1/h96 MSE from 0.175458 to
0.113224, a 35.47% reduction. Validation-best checkpointing is the primary
correction; clipping provides a smaller additional mean change. Corrected KAN
is close to PatchTST under the same pilot h96 setting but remains weaker than
DLinear.

All 12 Stage 1 hybrid records and all 6 Stage 2 residual-target records select
epoch zero and restore residual gain to zero. The final hybrid predictions are
therefore DLinear predictions. This is a controlled negative result: the KAN
residual can reduce training loss while validation rejects the added component.

### 4.4 Optimizer-state causal campaign

On ETTh1/h96, moderate retain gives `0.211416 +/- 0.040933` and moderate reset
gives `0.266130 +/- 0.057011`. Reset is worse for all three paired seeds in
this schedule. Under aggressive updates, both policies are unstable; reset
has mean `3.719170 +/- 6.117718` versus retain `1.430756 +/- 1.631831`, with
the reset mean dominated by one catastrophic run.

The event logs confirm that reset was executed: all 39 non-initial reset events
had Adam parameter-state count zero after the intervention, while retained
conditions preserved state. The outcome does not support clearing Adam state as
a sufficient stabilization remedy.

![**Fig. 5. Grid Update dynamics.** Left: validation trajectories under four
schedule families. Right: immediate probe-MSE displacement after conservative,
moderate and aggressive updates.](figures/grid_update_dynamics.png)

### 4.5 ETTh2 pilot confirmation

On ETTh2/h96 under the pilot protocol, DLinear has mean MSE `0.235637 +/-
0.000122` and corrected KAN `0.270004 +/- 0.010761`. No-update KAN has
`0.284032 +/- 0.007912`, moderate retain `0.308285 +/- 0.025466` and moderate
reset `0.354125 +/- 0.063449`. This confirms the mean-level direction that
reset is not a reliable remedy, while also showing that Grid Update behavior is
dataset-dependent.

### 4.6 Uniform-checkpoint canonical confirmation

K09 uses the official hourly ETT 12/4/4 split, five seeds and one
minimum-validation-MSE checkpoint rule for every condition. Table 6 reports
the resulting same-contract comparison. K08 is retained only as a versioned
checkpoint-sensitivity campaign and is not pooled with K09.

**Table 6. Canonical h96 test MSE (mean +/- sample SD over five seeds).**

| Dataset | DLinear | Corrected KAN | No update | Moderate update |
|---|---:|---:|---:|---:|
| ETTh1 | 0.064489 +/- 0.001843 | 0.164913 +/- 0.014927 | 0.277007 +/- 0.039970 | 0.417187 +/- 0.159815 |
| ETTh2 | 0.127925 +/- 0.000598 | 0.195536 +/- 0.028679 | 0.329198 +/- 0.071826 | 0.369897 +/- 0.081077 |

K08-to-K09 changes are informative but not monotonic in test space. Uniform
validation selection reduces ETTh1 no-update MSE from 0.398114 to 0.277007 and
moderate-update MSE from 1.695296 to 0.417187. It reduces ETTh2 moderate-update
MSE from 0.580647 to 0.369897, while ETTh2 no-update MSE changes from 0.286589
to 0.329198. DLinear changes from 0.060650 to 0.064489 on ETTh1 and from
0.128562 to 0.127925 on ETTh2. Thus validation-best checkpointing removes much
of the extreme late-epoch Grid Update damage, but it is not a mechanism for
optimizing unseen test error and does not reverse the model ranking.
The complete paired seed-level contract comparison is supplied in the
reproducibility appendix; it is descriptive because K08 and K09 are separately
versioned source campaigns rather than interchangeable replications.

![**Fig. 6. Uniform-checkpoint forecasts at the fixed midpoint test origin.**
Actual OT is compared with the five-seed mean prediction; shading denotes
sample SD across seeds.](figures/k09_actual_vs_predicted.png)

The fixed-origin curves are descriptive rather than selected examples: origin
1392 is the midpoint of the 2785 test origins and is shared across datasets,
models and seeds. DLinear follows the observed short-term variation more
closely in this window, while corrected KAN is smoother and shows a pronounced
positive bias on ETTh1.

![**Fig. 7. Uniform-checkpoint residual and absolute-error distributions.**
Signed densities are peak-normalized; ECDFs use all saved test predictions
from all five seeds.](figures/k09_residual_diagnostics.png)

The residual densities show that corrected KAN errors are broader and more
positively shifted, especially on ETTh1. Its absolute-error empirical curves
also lie to the right of DLinear throughout most of the displayed range. These
prediction-level results are consistent with, rather than substituted for,
the aggregate MSE comparison.

![**Fig. 8. Original-unit RMSE across the 96-step forecast horizon.** Curves
show the five-seed mean; shading denotes sample SD across seeds.](figures/k09_forecast_step_rmse.png)

DLinear has the lowest mean original-unit RMSE at every one of the 96 forecast
steps on both datasets. Error rises with lead time for all four conditions,
but the separation from corrected and Grid Update KAN variants persists across
the complete horizon rather than being driven by a few terminal steps.

**Table 7. Uniform-checkpoint compute diagnostics on Tesla T4.**

| Dataset | Condition | Training s | Inference ms/sample | Peak GPU MB | Parameters | Selected epoch |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | DLinear | 11.88 +/- 1.87 | 0.01924 +/- 0.00044 | 17.94 | 18624 | 20.6 +/- 3.6 |
| ETTh1 | Corrected KAN | 85.83 +/- 4.53 | 0.10042 +/- 0.00308 | 17.76 | 9216 | 24.2 +/- 2.6 |
| ETTh1 | No-update KAN | 38.71 +/- 0.15 | 0.06445 +/- 0.00297 | 18.09 | 13440 | 7.6 +/- 5.4 |
| ETTh1 | Moderate-update KAN | 40.04 +/- 0.38 | 0.06482 +/- 0.00105 | 23.55 | 13440 | 7.6 +/- 2.2 |
| ETTh2 | DLinear | 10.96 +/- 0.05 | 0.01908 +/- 0.00021 | 17.94 | 18624 | 28.6 +/- 1.1 |
| ETTh2 | Corrected KAN | 83.73 +/- 0.19 | 0.09921 +/- 0.00057 | 17.76 | 9216 | 7.2 +/- 2.3 |
| ETTh2 | No-update KAN | 38.74 +/- 0.05 | 0.06313 +/- 0.00074 | 18.09 | 13440 | 20.0 +/- 4.9 |
| ETTh2 | Moderate-update KAN | 39.83 +/- 0.17 | 0.06365 +/- 0.00079 | 23.55 | 13440 | 7.2 +/- 4.4 |

Corrected KAN uses fewer trainable parameters than DLinear but takes 7.2 times
longer on ETTh1 and 7.6 times longer on ETTh2. Its measured inference latency
is approximately 5.2 times DLinear on both datasets. Moderate Grid Update also
raises peak recorded GPU memory from 18.09 to 23.55 MB relative to the same
no-update architecture.

![**Fig. 9. Compute cost and validation-selected checkpoints.** Bars show mean
training time and sample SD; dots show all five selected epochs.](figures/k09_compute_checkpoints.png)

**Table 8. Uniform-checkpoint error in recorded OT units.**

| Dataset | Condition | OT RMSE | OT MAE |
|---|---|---:|---:|
| ETTh1 | DLinear | 2.3302 +/- 0.0332 | 1.7196 +/- 0.0232 |
| ETTh1 | Corrected KAN | 3.7235 +/- 0.1683 | 3.0371 +/- 0.1520 |
| ETTh1 | No-update KAN | 4.8188 +/- 0.3636 | 4.0121 +/- 0.3133 |
| ETTh1 | Moderate-update KAN | 5.8265 +/- 1.2155 | 4.8517 +/- 1.0997 |
| ETTh2 | DLinear | 4.1435 +/- 0.0097 | 3.1429 +/- 0.0074 |
| ETTh2 | Corrected KAN | 5.1120 +/- 0.3698 | 4.0207 +/- 0.2907 |
| ETTh2 | No-update KAN | 6.6150 +/- 0.7259 | 5.0357 +/- 0.4974 |
| ETTh2 | Moderate-update KAN | 7.0152 +/- 0.7324 | 5.1980 +/- 0.5152 |

In operational terms, corrected KAN increases uniform-checkpoint RMSE by
approximately 1.39 recorded OT units on ETTh1 and 0.97 on ETTh2 relative to
DLinear. Moderate Grid Update increases the mean further on both datasets.

DLinear beats corrected KAN in all ten K09 dataset-seed pairs. Its mean MSE is
60.89% lower on ETTh1 and 34.58% lower on ETTh2. Moderate update is worse than
no update in four of five ETTh1 seeds and three of five ETTh2 seeds, with mean
MSE deltas of 0.140179 and 0.040700 respectively.

The absolute MSE values differ from the pilot results because the split and
history-boundary rules differ. This protocol sensitivity is reported rather
than hidden.

### 4.7 Canonical multi-horizon comparison

K10 extends the canonical comparison without pooling K08 or K09. Table 9 and
Fig. 10 summarize the 120 validation-best runs. DLinear has the lowest mean MSE
in all eight dataset-horizon cells. It has lower seed-level MSE than corrected
KAN in 20/20 ETTh1 and 18/20 ETTh2 horizon-seed pairs, and lower MSE than
PatchTST in 18/20 pairs on each dataset.

**Table 9. Canonical multi-horizon test MSE over five seeds.**

| Dataset | Horizon | DLinear | PatchTST | Corrected KAN |
|---|---:|---:|---:|---:|
| ETTh1 | 96 | 0.064489 +/- 0.001843 | 0.121910 +/- 0.023330 | 0.164914 +/- 0.014927 |
| ETTh1 | 192 | 0.079831 +/- 0.001271 | 0.126775 +/- 0.018393 | 0.190365 +/- 0.027766 |
| ETTh1 | 336 | 0.098667 +/- 0.002689 | 0.168590 +/- 0.043239 | 0.227936 +/- 0.011707 |
| ETTh1 | 720 | 0.195691 +/- 0.027741 | 0.212319 +/- 0.043358 | 0.269458 +/- 0.035737 |
| ETTh2 | 96 | 0.127925 +/- 0.000598 | 0.243137 +/- 0.052484 | 0.195537 +/- 0.028679 |
| ETTh2 | 192 | 0.179582 +/- 0.000461 | 0.272309 +/- 0.021118 | 0.285171 +/- 0.035656 |
| ETTh2 | 336 | 0.229555 +/- 0.001062 | 0.318983 +/- 0.085294 | 0.260597 +/- 0.016748 |
| ETTh2 | 720 | 0.318278 +/- 0.001366 | 0.338608 +/- 0.025483 | 0.328821 +/- 0.014056 |

![**Fig. 10. Canonical multi-horizon forecasting accuracy.** Points show
five-seed mean standardized MSE; shading denotes sample SD within each
dataset-horizon-model cell.](figures/k10_canonical_multihorizon.png)

The ranking is not equally separated at every horizon. On ETTh1, PatchTST is
between DLinear and corrected KAN in 19/20 paired comparisons. On ETTh2 its
relationship with corrected KAN is mixed, and at h720 all three mean MSEs lie
within 0.021 of one another. Corrected KAN nevertheless does not attain the
lowest mean in any registered cell.

![**Fig. 11. Multi-horizon compute profiles on Tesla T4.** Lines show mean
training time and full-test-loader inference latency over five seeds for each
fixed model contract.](figures/k10_multihorizon_compute.png)

The compute ordering is stable across horizons. DLinear requires about
11.2-12.2 s per run, PatchTST 46.6-55.7 s and corrected KAN 78.7-87.6 s.
Corrected KAN inference remains approximately 4.4-5.3 times slower per sample
than DLinear, while PatchTST is approximately 2.9-3.3 times slower. Parameter
counts grow with output horizon, but corrected KAN remains slower despite
using fewer parameters than DLinear in every registered cell.

![**Fig. 12. Canonical 720-step forecasts at the fixed midpoint test origin.**
Actual OT is compared with five-seed mean predictions; shading denotes sample
SD and the midpoint origin is fixed before model comparison.](figures/k10_h720_forecasts.png)

The h720 curves expose a shared long-horizon limitation: the direct models
capture smooth level and periodic structure but attenuate abrupt excursions
and regime shifts. ETTh1 predictions remain above the extended low-OT episode,
while ETTh2 PatchTST collapses much of the oscillation toward a smooth level.
This figure is illustrative of one pre-fixed origin and does not replace the
all-origin MSE results.

## 5. Discussion

### 5.1 What HPO helps and what it does not

The training-consistency intervention demonstrates that a KAN can be
substantially underestimated when HPO and final evaluation optimize different
objectives. This is a useful methodological result: HPO-selected regularization
and validation-best selection should be part of the declared training contract.
The correction does not reverse the model ranking against DLinear, especially
under the canonical protocol. K10 extends that conclusion beyond h96: DLinear
retains the lowest mean at all eight registered dataset-horizon cells, although
the ETTh2/h720 separation is small and two paired seeds favor corrected KAN.

The result also illustrates why model selection belongs to the reported model
definition. HPO guidance [23,24] treats the validation objective and budget as
part of the optimization procedure; changing checkpoint selection after search
can erase or exaggerate the apparent benefit of the selected hyperparameters.
For this reason, the paper reports K08 under its executed rule and K09 under the
uniform intended rule instead of silently normalizing or overwriting an archive
after observing test outcomes. The fact that K09 improves several Grid Update
conditions without making them competitive is evidence that checkpoint choice
matters but does not explain away the architecture and update-schedule gaps.

### 5.2 Grid Update as a deployment risk

The results show that Grid Update is not simply an on/off accuracy feature.
Sparse adaptation can help a fixed pilot configuration, but repeated updates
create large state displacement and high seed variance. The canonical results
strengthen the operational warning: under K09, moderate updates have worse mean
MSE than no update on both datasets and lose four of five ETTh1 and three of
five ETTh2 paired comparisons. A deployment should
therefore log update events, validate update schedules and treat catastrophic
seed behavior as a first-class risk.

### 5.3 Why Adam reset is not enough

Resetting Adam removes accumulated moment tensors, and the implementation logs
that removal. Yet the forecast outcomes do not improve reliably. This means
that the instability cannot be addressed by a one-line optimizer-state reset
alone. The result does not prove that Adam moments have no causal contribution;
it shows that reset is insufficient under the tested KAN and schedule.

### 5.4 Engineering implications

For transformer-temperature forecasting, a simple DLinear baseline should be
established before deploying KAN. If KAN is selected for interpretability or
nonlinear modeling, the training pipeline should use validation-best
checkpointing, explicit regularization, conservative or disabled Grid Updates,
seed replication and update-event diagnostics. These controls reduce the risk
that an apparently strong single run is deployed as a reliable model.

An engineering acceptance gate can be made explicit without changing the frozen
results: a candidate KAN should be retained only if it is competitive with the
same-protocol DLinear baseline on every declared horizon, remains stable across
the registered seeds, and produces no non-finite or catastrophic Grid Update
event. Before deployment, the selected checkpoint and scaler should be stored
with the source and dataset hashes, while inverse-scaled OT error should be
reviewed against the operational tolerance of the transformer asset. K10
directly applies this gate across four horizons, and corrected KAN fails the
mean-accuracy criterion in all eight cells. DLinear therefore remains the
default reliability model; the KAN can be
kept as a monitored research alternative rather than silently promoted from a
favorable pilot run. This is a decision rule derived from the observed failure
mechanisms, not a newly measured acceptance threshold.

### 5.5 Relation to comparable engineering studies

The full-text JEAS benchmark [16-19] commonly emphasizes model architecture,
prediction curves, feature or attention interpretation, sensitivity analysis
and a decision-facing use case. The present study meets the architecture,
ablation, baseline, compute and engineering-consequence expectations, while
adding a stronger evidence trail through source hashes, dataset hashes, frozen
JSON records and protocol-aware claim auditing. It does not reproduce
multivariate feature-importance graphics because a univariate OT input cannot
support that inference. Its interpretive contribution is instead mechanistic:
Grid Update events are tied to immediate displacement, subsequent validation
behavior and optimizer-state interventions.

The visual inventory is evidence-driven. K09 supplies the underlying arrays
needed for actual-versus-predicted curves, residual distributions and
forecast-step diagnostics, and those figures aggregate all five seeds rather
than selecting the most favorable run. Feature-importance plots still require
a multivariate predictor and a specified attribution method. Treating the seven
recorded ETT columns as seven model inputs after the fact would misrepresent the
frozen univariate experiment. The figures therefore cover dataset shift, HPO
fitness, seed-level ablation, event displacement, prediction behavior,
residuals, lead-time error, canonical multi-horizon accuracy, h720 forecast
behavior and compute/checkpoint variation.

### 5.6 Limitations

The modern experiments are univariate and use two related ETT datasets rather
than a broad set of industrial domains. The historical HPO comparison has only
three runs per search method and is not a fair inferential optimizer ranking.
The canonical campaigns have five seeds and K10 covers four standard horizons,
but only three fixed model contracts are compared across those horizons. The
conclusions concern the tested efficient-kan configuration, protocols and
schedules; they are not universal claims about all KAN variants or datasets.

K09 resolves the checkpoint inconsistency for the four canonical conditions
and adds prediction arrays, checkpoints, latency and peak GPU memory. K10
resolves the canonical multi-horizon coverage gap for DLinear, PatchTST and
corrected KAN, but it does not resolve external validity: ETTh1 and ETTh2 are
related benchmark datasets, and no independent utility, manufacturing or
field-deployment dataset is tested. The inference measurements
come from one Tesla T4 software environment and are comparative rather than
hardware-independent latency guarantees. FLOPs, energy consumption and
end-to-end data-pipeline cost were not logged. Finally, five seeds expose
dispersion and paired direction but remain too few for broad distributional or
post-hoc significance claims.

## 6. Conclusion

This study provides a reproducible account of KAN HPO and Grid Update behavior
for engineering time-series forecasting. Making final training consistent with
the HPO objective substantially improves direct KAN performance under the
pilot protocol. However, the uniform-checkpoint canonical experiment shows
that DLinear remains more accurate at every h96 forecast step and in all ten
K09 dataset-seed comparisons against corrected KAN. The 120-run K10 campaign
extends the same comparison across four horizons: DLinear has the lowest mean
in all eight cells and lower MSE in 38/40 paired comparisons, while the
ETTh2/h720 gap narrows. Repeated Grid Updates retain
worse mean error under both datasets, while the K08-to-K09 comparison shows
that checkpoint selection can reduce late-epoch damage without reversing the
ranking. Clearing Adam state is not a sufficient remedy. The resulting contribution is a stability
and reproducibility framework for deciding when KAN configuration search is
worth deploying, not a claim of universal KAN superiority.

## List of abbreviations

**ETT:** Electricity Transformer Temperature; **HPO:** hyperparameter
optimization; **KAN:** Kolmogorov-Arnold Network; **MAE:** mean absolute error;
**MSE:** mean squared error; **OT:** oil temperature; **PSO:** Particle Swarm
Optimization; **SSW:** Stabilization Sliding Window; **WOA:** Whale Optimization
Algorithm.

## Declarations

### Availability of data and materials

The ETTh1 and ETTh2 data are public components of the Electricity Transformer
Temperature benchmark distributed with the Informer repository [3]. The exact
dataset files used in this study are identified by SHA-256 hashes in the
experiment registry and reproducibility manifest. No private or proprietary
data were used.

### Code availability

- Project name: kan-grid-update-reliability
- Project home page: https://github.com/nphu211206/kan-grid-update-reliability
- Archived version: [ZENODO_DOI] (Zenodo DOI — to be inserted after deposit)
- Operating system: Platform independent (tested on Windows 11 and Kaggle Linux)
- Programming language: Python 3.10–3.12
- Other requirements: See `requirements-canonical.txt` and `requirements-publication.txt`
- License: MIT License (project-owned code); efficient-kan: upstream MIT;
  ETTDataset: CC BY-ND 4.0 (raw data not redistributed)
- Any restrictions to use by non-academics: None

Each frozen result record stores its source hash and protocol identifier. The
archival Zenodo DOI will be inserted after the deposit passes the project hash audit.

### Competing interests

The author declares that there are no competing interests.

### Funding

The author received no external funding for this research.

### Authors' contributions

Nguyen Phu Toan: conceptualization, methodology, software, validation, formal
analysis, investigation, data curation, visualization, writing—original draft,
writing—review and editing, project administration and supervision. The author
read and approved the final manuscript.

### Acknowledgements

Not applicable.

### Ethics approval and consent to participate

Not applicable; the study uses public machine-learning benchmark data and
involves no human participants or animals.

### Consent for publication

Not applicable.

## References

1. Liu, Z. et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.
2. Liu, Z. et al. (2024). KAN 2.0: Kolmogorov-Arnold Networks Meet Science. arXiv:2408.10205.
3. Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. Proceedings of AAAI, 35(12), 11106-11115. DOI: 10.1609/aaai.v35i12.17325.
4. Nie, Y. et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR. arXiv:2211.14730.
5. Zeng, A. et al. (2023). Are Transformers Effective for Time Series Forecasting? Proceedings of AAAI, 37(9), 11121-11128. DOI: 10.1609/aaai.v37i9.26317.
6. Liu, Y. et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. ICLR. arXiv:2310.06625.
7. Wu, H. et al. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR. arXiv:2210.02186.
8. Kennedy, J. and Eberhart, R. (1995). Particle swarm optimization. Proceedings of ICNN, 1942-1948.
9. Storn, R. and Price, K. (1997). Differential evolution: A simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization, 11(4), 341-359. DOI: 10.1023/A:1008202821328.
10. Mirjalili, S., Mirjalili, S.M. and Lewis, A. (2014). Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61. DOI: 10.1016/j.advengsoft.2013.12.007.
11. Mirjalili, S. and Lewis, A. (2016). The Whale Optimization Algorithm. Advances in Engineering Software, 95, 51-67. DOI: 10.1016/j.advengsoft.2016.01.008.
12. Wu, H. et al. (2022). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS. arXiv:2106.13008.
13. Qiu, B., Zhang, J., Yang, Y., Qin, G., Zhou, Z. and Ying, C. (2024). Research on Oil Well Production Prediction Based on GRU-KAN Model Optimized by PSO. Energies, 17(21), 5502. DOI: 10.3390/en17215502.
14. Agrawal, U.K. and Panda, N. (2025). Quantum-inspired adaptive mutation operator enabled PSO (QAMO-PSO) for parallel optimization and tailoring parameters of Kolmogorov-Arnold network. The Journal of Supercomputing, 81(14). DOI: 10.1007/s11227-025-07810-w.
15. Blealtan. efficient-kan: An efficient implementation of Kolmogorov-Arnold Networks. GitHub repository, pinned project source recorded in the supplementary manifest.
16. Alnaqbi, A., Zeiada, W. and Al-Khateeb, G. (2025). A hybrid machine learning method of support vector regression with particle swarm optimization for predicting IRI in continuously reinforced concrete pavement. Journal of Engineering and Applied Science, 72, article 128. DOI: 10.1186/s44147-025-00706-9.
17. Alnaqbi, A., Al-Khateeb, G.G. and Zeiada, W. (2025). Predictive modeling of longitudinal cracking in CRCP using PSO-tuned gradient boosting machines. Journal of Engineering and Applied Science, 72, article 58. DOI: 10.1186/s44147-025-00623-x.
18. Guo, Y., Chen, Y., Wang, H. and Liu, Y. (2024). Data-driven multivariate time series prediction of in-vehicle equipment failure rates. Journal of Engineering and Applied Science, 71, article 210. DOI: 10.1186/s44147-024-00543-2.
19. Yang, J., Zhang, B., Tong, L., Cao, S., Zhao, B., Tang, Z., Wang, G., Gao, H. and Sun, S. (2025). An error correction ACKLSTM model combining data-driven and deep learning for wind power prediction. Journal of Engineering and Applied Science, 72, article 191. DOI: 10.1186/s44147-025-00771-0.
20. Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. International Journal of Forecasting, 16(4), 437-450. DOI: 10.1016/S0169-2070(00)00065-0.
21. Bergmeir, C., Hyndman, R.J. and Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. Computational Statistics & Data Analysis, 120, 70-83. DOI: 10.1016/j.csda.2017.11.003.
22. Hewamalage, H., Ackermann, K. and Bergmeir, C. (2023). Forecast evaluation for data scientists: common pitfalls and best practices. Data Mining and Knowledge Discovery, 37, 788-832. DOI: 10.1007/s10618-022-00894-5.
23. Yang, L. and Shami, A. (2020). On hyperparameter optimization of machine learning algorithms: theory and practice. Neurocomputing, 415, 295-316. DOI: 10.1016/j.neucom.2020.07.061.
24. Bischl, B. et al. (2023). Hyperparameter optimization: foundations, algorithms, best practices, and open challenges. WIREs Data Mining and Knowledge Discovery, 13(2), e1484. DOI: 10.1002/widm.1484.
25. Dodge, J., Gururangan, S., Card, D., Schwartz, R. and Smith, N.A. (2019). Show Your Work: Improved Reporting of Experimental Results. Proceedings of EMNLP-IJCNLP, 2185-2197. DOI: 10.18653/v1/D19-1224.
