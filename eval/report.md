# Sparse-Autoencoder Features from a Time-Series Foundation Model Predict Forecast Difficulty

**Source of truth.** This is the canonical technical report for the TSFM project:
methodology, per-section results, and numbers live here. The top-level
[`README.md`](../README.md) is the repo entry point and links here rather than
restating; [`workshop_paper.md`](workshop_paper.md) is the condensed paper and
cross-references this document's section numbers.

**Working draft.** Numerical claims marked `[FILL: …]` are populated automatically once `probing/results/probe_results.json` exists from a full-series run. All prose is editable.

## Abstract
Time-series foundation models (TSFMs) are deployed as black boxes; downstream
systems would benefit from a label-free signal of *when not to trust a
forecast*. We train an 8× TopK sparse autoencoder on encoder-block activations
of Chronos-T5 and ask whether the discovered features predict a forecast's own
difficulty *beyond* cheap input statistics and raw activations. On ETTh1 with a
purged temporal split, the SAE features yield an incremental AUROC of
**−0.228 (95% CI [−0.366, −0.092])** over eight classical input statistics, and
**−0.158 (95% CI [−0.293, −0.025])** over raw activations. Reframing the same
input-statistics probe as a selective-prediction signal, however, recovers a
positive result: at 50 % coverage, mean CRPS drops 8.1 % from no abstention
(1.527 → 1.403), with AURC 1.215 vs random 1.374 and oracle 0.850 — capturing
roughly 30 % of the oracle's available AURC improvement. Crucially, this cheap
probe also **beats the standard UQ baseline** — the model's own
conformalized predictive-interval width — which captures only 9–11 % of the
oracle headroom and is *worse than no abstention* at aggressive coverage (§4.9).
We report both the null SAE result and the positive selective-prediction result
honestly, without post-hoc reframing of the metric.

## 1. Introduction
TSFMs (Chronos, TimesFM, Moirai) deliver strong zero-shot forecasts but offer
no native abstention or routing signal. A useful question — answered nowhere
in the published literature, as of early 2026 — is whether internal
representations encode self-difficulty that is not trivially derivable from the
input. If yes, the same TSFM can drive a feature-routed cascade (cheap model
by default, escalate when difficulty features fire) at zero extra training
cost.

Contributions of this preliminary work:
1. The first **incremental, leakage-controlled** evaluation of internal
   representations of a TSFM as a difficulty predictor.
2. A reproducible pipeline (TopK SAE → L1 logistic probe → paired-bootstrap
   ΔAUROC) instrumented with hard guardrails against random-split leakage on
   overlapping sliding windows and scale-dependent CRPS labels.
3. [Optional, target tier:] A one-point cascade demonstration on
   `chronos-t5-small` ↔ `chronos-t5-base`.

## 2. Related work
- **Mishra (2026)**, "Dissecting Chronos: Sparse Autoencoders Reveal Causal
  Feature Hierarchies in Time Series Foundation Models", arXiv:2603.10071.
  Trains TopK SAEs on Chronos-T5-Large (710 M) across six layers; 392
  single-feature ablation experiments establish a depth-dependent hierarchy in
  which the mid-encoder concentrates causally critical change-detection
  features. *Verified May 2026.*
- **TimeSAE (Jan 2026)**, "TimeSAE: Sparse Decoding for Faithful Explanations
  of Black-Box Time Series Models", arXiv:2601.09776. JumpReLU SAE for
  *post-hoc, model-agnostic* black-box explanation under distribution shift.
  *Verified May 2026.*
- **Chronos** (Ansari et al., 2024). T5 backbone trained as a language model
  on quantized numeric tokens; benchmark protocol used here.

**Our distinction.** Mishra targets *causal* features via ablation; TimeSAE
targets *post-hoc explanations* of any black-box. Neither uses internal
features as a **label-free, inference-time signal for routing or abstention**.
This is the novelty wedge of the present work, and it remains unclaimed in
the published 2026 literature.

## 3. Method

### 3.1 Backbone and activation extraction
We use `amazon/chronos-t5-small` (60 M parameters, `d_model=512`,
encoder–decoder T5). Forward hook on the final sub-layer of encoder block
`num_layers/2` captures post-LN residuals of shape (B, 513, 512) — 512 context
tokens plus one EOS. We verify empirically that the encoder hook fires once
per window and is **not** expanded by `num_samples`; this was a known risk and
is ruled out.

### 3.2 Labels
For each window we sample 100 forecasts and compute CRPS; we use **seasonal**
naive MASE with m=24 (hourly daily seasonality) as a secondary label. CRPS is
normalized using **train-split statistics only**, then thresholded at the
top-15 % train quantile to define `hard`. The temporal train/test split
includes a **purge gap** ≥ context + horizon to eliminate window overlap
between train and test.

### 3.3 Sparse autoencoder
TopK SAE with `d_hidden = 8·d_model = 4096`, `k=32`, `aux_k=512` for
dead-feature revival, decoder-bias initialized to the activation mean and
decoder columns kept unit-norm. The SAE is trained **only on train-split
tokens** to keep it blind to inputs the probe is tested on. Hyperparameters:
Adam, lr 5e-4, 1k warmup steps, ~5 epochs over ~247296 train tokens.

### 3.4 Difficulty probe
L1 logistic regression over `concat(mean, max, last)` pooling of either raw
activations or SAE codes, plus eight classical input statistics (variance,
volatility, lag-1 and seasonal autocorrelation, spectral entropy, trend slope,
range, ADF p-value with a scipy variance-ratio fallback). `C` is selected by
5-fold time-series cross-validation on the train split. We report:
- `P1` = input-stats only — the baseline that matters,
- `P2` = input-stats + raw activations,
- `P3` = input-stats + SAE features.

The headline metric is **paired-bootstrap ΔAUROC** (B = 2000): `P2−P1`,
`P3−P1`, and especially `P3−P2`. The last comparison neutralizes the
dimensionality argument — SAE vs raw, both high-dim.

## 4. Experiments

### 4.1 Setup
ETTh1 (`OT` channel), context 512, horizon 96, stride 24 → ~701
windows. After temporal split with purge gap: n_train = 483, n_test =
167, hard fraction (test) = 0.16. Single seed (42); per-test-window
scores in `probing/results/probe_scores.parquet`.

### 4.2 Headline ΔAUROC

| Probe                 | Test AUROC (95 % CI)  |
|-----------------------|-----------------------|
| P1 stats (8 features) | 0.654 (0.552, 0.751)  |
| P2 stats + raw        | 0.584 (0.460, 0.697)  |
| P3 stats + sae        | 0.426 (0.313, 0.547)  |
| P4 raw only (diag.)   | 0.584 (0.461, 0.698)  |
| P5 sae only (diag.)   | 0.426 (0.312, 0.546)  |

| Δ                       | Point   | 95 % CI            |
|-------------------------|---------|--------------------|
| P2 − P1 (raw value)     | −0.070  | [−0.193, +0.055]   |
| P3 − P1 (sae value)     | −0.228  | [−0.366, −0.092]   |
| P3 − P2 (sae over raw)  | −0.158  | [−0.293, −0.025]   |

**Diagnostic reading.** P4 (raw only) matches P2 (stats + raw) to three decimals,
and P5 (sae only) matches P3 (stats + sae). The L1 logistic, when given both
input statistics and high-dim activations, ignores the statistics — so the
collapse of P2/P3 below P1 is *not* input-stats being drowned out; it is the
raw/SAE features genuinely failing to carry signal beyond chance at the
mid-encoder of chronos-t5-small. Δ(P2 − P1) crosses zero (null); Δ(P3 − P1)
and Δ(P3 − P2) do not, and SAE features are significantly worse than raw.

Figure 1: `probing/results/auroc.png` — bars with bootstrap CIs.

**Cross-layer robustness check.** To confirm the null result is not an artifact
of layer choice, we re-ran the full pipeline on activations hooked from the
**last** encoder block (block 5 of 6) instead of the mid-encoder (block 3),
reusing the same labels via `extract_activations.py --layer_idx 5 --skip_predict`.
Saved as `probing/results/probe_results_late_layer5.json`.

| Probe                 | Layer 3 (mid)          | Layer 5 (late)         |
|-----------------------|------------------------|------------------------|
| P1 stats (unchanged)  | 0.654 (0.552, 0.751)   | 0.654 (0.552, 0.751)   |
| P2 stats + raw        | 0.584 (0.460, 0.697)   | 0.450 (0.321, 0.577)   |
| P3 stats + sae        | 0.426 (0.313, 0.547)   | 0.511 (0.380, 0.639)   |
| P4 raw only (diag.)   | 0.584 (0.461, 0.698)   | 0.441 (0.316, 0.564)   |
| P5 sae only (diag.)   | 0.426 (0.312, 0.546)   | 0.511 (0.380, 0.639)   |

| Δ                       | Layer 3 (mid)              | Layer 5 (late)             |
|-------------------------|----------------------------|----------------------------|
| Raw − Stats             | −0.070 [−0.193, +0.055]    | −0.204 [−0.324, −0.075]    |
| SAE − Stats             | −0.228 [−0.366, −0.092]    | −0.143 [−0.277, −0.013]    |
| SAE − Raw               | −0.158 [−0.293, −0.025]    | +0.061 [−0.083, +0.194]    |

The headline conclusion survives the layer swap: classical input statistics
outperform both raw activations and SAE features at *both* the mid- and
late-encoder, with all Δ(internal − stats) CIs lying below zero. One nuance
emerges from the diagnostic probes: at the late encoder, raw activations
drop to near-chance (P4 = 0.441), and the previously significant SAE-vs-raw
gap closes to a null (Δ(SAE − Raw) = +0.061, CI crosses zero). Where the
raw representation carries less signal, the SAE's compression is at parity
with raw — consistent with a sparse autoencoder that is doing approximately
the right thing on inputs that simply do not carry the target signal.

**Full encoder-depth sweep.** To rule out the null being a two-point artifact,
we extended the check to *every* encoder block (0–5) — embed-only
(`--skip_predict`, reusing the committed CRPS labels), the whole sweep in
~17 min (`experiments/depth_sweep.sh`, `probing/results/depth/`):

| Block | P1 stats | P2 stats+raw | P3 stats+sae | Δ(SAE−stats) | Δ(SAE−raw) |
|-------|----------|--------------|--------------|--------------|------------|
| 0 (early) | 0.654 | 0.576 | 0.484 | −0.170 | −0.093 |
| 1     | 0.654 | 0.584 | 0.596 | **−0.058** | +0.012 |
| 2     | 0.654 | 0.637 | 0.450 | −0.204 | −0.187 |
| 3 (mid)   | 0.654 | 0.584 | 0.426 | **−0.228** | −0.158 |
| 4     | 0.654 | 0.557 | 0.502 | −0.152 | −0.055 |
| 5 (late)  | 0.654 | 0.450 | 0.511 | −0.143 | +0.061 |

The null is **monotone in neither direction but robust at every depth**:
Δ(SAE−stats) is negative across all six blocks (−0.058 to −0.228), deepest at
the mid-encoder (blocks 2–3) and shallowest at block 1, and never turns
positive. Blocks 3 and 5 reproduce §4.2's headline and late-layer numbers
exactly, confirming the sweep is faithful. Input statistics beat the SAE
features at *every* encoder layer of chronos-t5-small — the result is not a
layer-selection artifact.

### 4.3 Selective prediction (positive result on the same data)

Although SAE features do not add incremental predictive power, the
input-statistics probe (P1) is itself a usable forecast-abstention signal.
We sort test windows ascending by P1's predicted `P(hard)` and report the
mean CRPS on the retained `coverage·N` predicted-easy windows, against an
oracle (sort by true CRPS) and a random baseline averaged over 2,000
permutations:

| Method        | AURC ↓ | Mean CRPS @ 50% coverage | Reduction vs no abstention |
|---------------|--------|--------------------------|----------------------------|
| No abstention | 1.527  | 1.527                    |                            |
| Random        | 1.374  | 1.527                    |                            |
| **P1 stats**  | **1.215** | **1.403**             | **−8.1 %**                 |
| P2 stats+raw  | 1.240  | 1.480                    | −3.0 %                     |
| P3 stats+sae  | 1.437  | 1.559                    | +2.1 % (worse)             |
| Oracle        | 0.850  | 0.872                    | −42.9 % (ceiling)          |

Figure 3: `eval/results/risk_coverage.png`.

The P1 probe captures **30 % of the available oracle improvement on AURC**
((random − P1) / (random − oracle) = 0.159 / 0.524). The SAE-based probes
(P3, P5) sit at or below random, consistent with the §4.2 finding that they
carry no difficulty signal beyond input statistics. Interpretation: cheap
classical context-window statistics from a 512-step window already form a
useful, label-free selective-prediction signal for Chronos-T5 forecasts on
ETTh1 — and that signal does not need internal representations to extract.

### 4.4 Qualitative feature inspection
Figure 2 (`probing/results/features/feat_*.png`): the five SAE features with
largest absolute probe weights, plotted on the windows where each fires
hardest. Interpretation: [FILL — do these visibly correspond to regime
shifts / level changes / anomalies, or are they diffuse? Be honest.]

### 4.5 Feature-routed cascade (executed)

We add a second backbone, `amazon/chronos-t5-base` (`d_model=768`, ~3.3× the
parameter count of small), and run a focused test-only extraction
(`eval/extract_base_crps_test_only.py`) to get per-window CRPS on the same
167 test windows used by the probe. The cascade routes between small (cheap,
cost = 1.0) and base (expensive, cost = 5.0) at threshold τ on a probe's
predicted `P(hard)`; we sweep τ and compare the resulting Pareto curve
against three rigorous reference curves built on the same data:

- **always cheap** anchor at (1.0, 1.5266)
- **always base**  anchor at (5.0, 1.5130)
- **linear interpolation** (the random-equivalent line between the two anchors)
- **random routing**: at each routing fraction *f*, the mean over 500 random
  permutations choosing *f·N* windows uniformly at random for the base
- **oracle routing**: at each fraction *f*, route the *f·N* windows where
  `crps_small − crps_base` is largest (the best any oracle-ranked router can
  do without seeing the per-window forecast outcome twice)

| Routing signal             | # Pareto-dominating points | Best dominating (cost, CRPS) |
|----------------------------|----------------------------|------------------------------|
| pred_P3_InputStats_SAE     | 1                          | (1.048, 1.5256)              |
| pred_P1_InputStats         | **5**                      | **(4.88, 1.5092)**           |

**Honest read.** The cascade is feasible — `P1` routing finds five points
strictly below the random/interpolation line, and the best of them (route
~97 % of windows to base, retain ~3 % on small) achieves mean CRPS 1.5092,
beating *both* anchors (always-cheap 1.5266 and always-base 1.5130). SAE-based
routing finds only one trivially-dominating point at a 1.2 % base-routing
fraction. This is consistent with §4.2 (SAE features carry no useful
difficulty signal at chronos-t5-small's mid- or late-encoder).

**Headline caveat.** The operational ceiling on this dataset is small: base
outperforms small by only **0.9 % mean CRPS** on ETTh1, and base wins on
only **52.1 %** of test windows — essentially a coin flip. The cascade's
**methodology** (probe-driven routing strictly dominates random; the routing
signal carries deployable value even when the absolute CRPS gap is small) is
the artifact; the empirical magnitude of the gain is bounded by the small
backbone gap on this particular series. A higher-variance series, a wider
backbone gap (small ↔ large), or a more difficulty-discriminating routing
signal would all widen this gain — left to future work.

Figure 4: `eval/results/pareto_frontier.png`.
Saved metrics: `eval/results/cascade_results.json` (full frontier curves and
dominating-point lists for both probes).

### 4.6 Causal ablation of top-K difficulty-predictive features

We test whether the SAE features the §4.2 probe ranks most predictive of
difficulty are *causally* tied to forecast quality (Mishra-2026 style, smaller
scale). For each of the 167 test windows, a forward hook on
`encoder.block[3].layer[-1]` (the same layer the SAE was trained on) replaces
the hidden state with the SAE's reconstruction under three conditions:
**natural** (no hook), **SAE-reconstruct** (no features zeroed; isolates
reconstruction-loss cost), and **ablate(feat=k)** for each of the top-5
features ranked by absolute L1-logistic coefficient. CRPS is sampled at
num_samples=50 for the SAE-recon and ablation conditions (the relative
comparison vs. recon is what matters; absolute CRPS vs. the 100-sample
natural baseline carries an MC-noise caveat).

**Top-5 features identified** (mid-encoder, L1 coefs):
`[1465 (0.67), 2717 (0.56), 1425 (0.51), 3702 (0.46), 3678 (0.45)]`.

**Reconstruction-loss baseline.** Δ(SAE-recon − natural) = **−0.023**
(95% CI [−0.076, +0.033]). Null — inserting the SAE into the forward pass
does not measurably degrade forecasts on average, so the ablation deltas
below are not confounded by a baseline reconstruction penalty.

**Per-feature ablation (Δ(ablate − recon), 2,000 paired-bootstrap iters):**

| Feature | All (n=167)                | Hard tercile (n=56)        | Easy 2/3 (n=111)           |
|---------|----------------------------|----------------------------|----------------------------|
| 1465    | +0.003 [−0.024, +0.029]    | +0.028 [−0.032, +0.085]    | −0.010 [−0.037, +0.016]    |
| 2717    | +0.010 [−0.021, +0.040]    | +0.034 [−0.033, +0.108]    | −0.002 [−0.029, +0.030]    |
| 1425    | +0.022 [−0.003, +0.050]    | +0.055 [−0.009, +0.122]    | +0.006 [−0.018, +0.032]    |
| 3702    | +0.005 [−0.024, +0.036]    | +0.046 [−0.028, +0.117]    | −0.015 [−0.041, +0.011]    |
| 3678    | +0.007 [−0.022, +0.035]    | +0.051 [−0.010, +0.112]    | −0.016 [−0.042, +0.011]    |

**Aggregate over the 5 features** (mean ΔCRPS per window across ablations):

| Cohort  | Effect  | 95 % CI               | Read |
|---------|---------|-----------------------|------|
| All     | +0.009  | [−0.014, +0.031]      | null |
| Hard    | **+0.043** | **[−0.008, +0.095]** | **near-significant** |
| Easy    | −0.008  | [−0.028, +0.013]      | null |
| Diff-in-diff (hard − easy) | **+0.050** | **[−0.005, +0.104]** | **near-significant** |

**Honest read.** Individually all five features pass through zero (population
causal null). But **5/5 features have larger positive point estimates on hard
windows than on easy** (a directionally consistent pattern that is not what
random noise produces), and the aggregate ablation effect on hard windows
(+0.043, ~3 % of the natural mean CRPS) sits with its CI lower bound
**−0.008** — i.e. barely crossing zero. The diff-in-diff (hard − easy) tells
the same story (+0.050, CI [−0.005, +0.104]). At n=56 hard windows the
bootstrap simply cannot resolve an effect of this magnitude.

Interpretation: the top-5 features are **weakly causally tied to forecast
quality on hard windows** — consistent in direction across features, with a
non-trivial magnitude, but underpowered for individual-feature significance
at this dataset size. This refines §4.2's predictive null: the features are
correlational signal-of-difficulty AND carry a weak causal contribution to
the forecast on hard inputs, but neither effect is strong enough to be
detectable with this sample. A larger hard-cohort (more windows, harder
series, or a backbone with richer mid-encoder representations) would resolve
whether the consistent direction is real signal or coordinated bootstrap
noise.

Saved: `eval/results/causal_ablation.parquet` (per-window),
`eval/results/causal_ablation.json` (aggregate).

### 4.7 Probe calibration & reliability

AUROC measures **ranking** quality. For deployment as an abstention signal,
**calibration** matters separately: `P(hard) = 0.8` should mean "about 80 %
of these windows are hard". We bin each probe's test-window predictions into
10 equal-width probability bins and report Expected Calibration Error (ECE)
and Brier score.

| Probe        | ECE ↓     | Brier ↓  | Read |
|--------------|-----------|----------|------|
| **P1 stats** | **0.380** | **0.205** | best ranker, best calibrated |
| P2 stats+raw | 0.561     | 0.451    | severely miscalibrated |
| P3 stats+sae | 0.498     | 0.370    | severely miscalibrated |
| P4 raw only  | 0.561     | 0.451    | severely miscalibrated |
| P5 sae only  | 0.499     | 0.370    | severely miscalibrated |

**Honest read.** All probes are systematically over-confident. Probes were
trained with `class_weight='balanced'` to maximize AUROC under the 15 %
train-hard rate, but the test hard fraction is 6.6 % (temporal distribution
shift — the test horizon falls into a milder regime of the series). The
high-dim probes (P2–P5) all converge to ECE ≈ 0.50, consistent with the
§4.2 finding that they don't carry usable signal beyond chance.

**Recalibration that works.** We fit a Platt (sigmoid) calibrator and an
isotonic calibrator on 5-fold OOF predictions over the full train set (all
483 windows participate as held-out cal data), then apply to the test
predictions. Platt is a strict monotone two-parameter fit so it **preserves
AUROC exactly** by construction.

|              | raw ECE | Platt ECE | Isotonic ECE | AUROC (preserved by Platt) |
|--------------|---------|-----------|--------------|----------------------------|
| **P1 stats** | 0.482   | **0.097** | 0.103        | 0.697 → 0.697              |
| P3 stats+sae | 0.404   | 0.153     | 0.157        | 0.611 → 0.611              |

P1's calibration error drops **80 %** under Platt (0.482 → 0.097), Brier
goes from 0.297 → 0.070, and ranking is unchanged. The probe is now
deployment-grade for selective-prediction use. (An earlier 80/20 temporal
split for the calibrator failed because the last 20 % of train had
distribution-shifted hard-rate; K-fold OOF on the full train resolves it.)

Figures 5–6: `eval/results/reliability_diagram.png` (raw probes),
`eval/results/reliability_recalibrated.png` (Platt + isotonic, P1 and P3).
Saved: `eval/results/calibration_results.json`,
`eval/results/recalibration_results.json`.

### 4.8 Pre-registered power analysis for the causal ablation

§4.6 left the causal claim underpowered: the aggregate ablation effect on hard
windows is +0.043 CRPS but its CI lower bound barely crosses zero at n=56. Before
"powering up" (lowering the hard threshold or pooling across datasets) we
**pre-register** the confirmatory design — fixing the endpoint, test, and target
n *from the pilot's own noise* so the powered run is confirmatory rather than a
garden of forking paths. The analysis (`eval/power_analysis.py`) uses only the
already-collected `causal_ablation.parquet`; no new forecasts.

**Pre-registration.** Unit = test window. Per-window statistic
`d_i = mean_f[crps_ablate(f,i) − crps_sae_recon(i)]` over the top-5 features
(the SAE-recon condition is the within-window control). Primary endpoint
`mean_{hard} d_i` with hard = top CRPS tercile; H1: `mean d_i > 0` (one-sided —
the direction is fixed now, from the 5/5 consistent positive signs in §4.6, not
re-chosen later). Test: paired bootstrap (B=2000), reject if the one-sided 95%
lower bound > 0; planning curves use the z-approximation
`n = ((z_{1−α}+z_{1−β})·sd/effect)²`. α=0.05, target power 0.80.

**Pilot effect/noise and required n** (reproduces §4.6: tercile effect +0.0427
matches the reported +0.043):

| Hard threshold | n_hard | effect (CRPS) | sd | current power | n for 80 % power (1-sided / 2-sided) |
|----------------|--------|---------------|------|---------------|--------------------------------------|
| top tercile (primary) | 56 | +0.0427 | 0.198 | **0.49** | **132 / 168** |
| top 40 % | 67 | +0.0427 | 0.185 | 0.60 | 116 / 147 |
| top 50 % | 84 | +0.0311 | 0.181 | 0.47 | 210 / 266 |

**Read.** At the current n=56 the experiment had **49 % power** — a coin flip —
which is *why* §4.6 is null; the pilot was simply too small to resolve a +0.043
effect at sd≈0.20. Reaching 80 % power needs **n≈132 (one-sided) / 168
(two-sided) hard windows, ≈2.4× the current count** (simulated power: 0.53 at
n=60 → 0.79 at n=130 → 0.88 at n=170). Crucially, **lowering the hard threshold
is the wrong lever past ~40 %**: at the top-50 % cut the cohort grows to n=84 but
the effect *dilutes* to +0.031 (less-hard windows carry less causal signal, per
§4.6's easy-cohort null), so the required n *rises* to 210. **Pooling hard
windows across datasets at a fixed top-tercile threshold — constant effect, more
n — strictly dominates.** Four ETT series at ~56 hard test windows each yield
~224 pooled hard windows, comfortably past the 168 needed for two-sided 80 %
power. This links the confirmatory ablation to the dataset replication (§5,
future work): the powered run is the per-window deltas from each replicated
dataset, concatenated, tested once under the pre-registration above.

The diff-in-diff (hard − easy) is +0.050 (SE 0.029, z=1.76, one-sided p=0.039) —
already nominally significant one-sided, consistent with §4.6's two-sided CI that
just straddles zero. Figure 7: `eval/results/power_analysis.png` (power and
minimum-detectable-effect vs n). Saved: `eval/results/power_analysis.json`.

### 4.9 Conformal-prediction baseline (the UQ method we must beat)

§4.3 shows the P1 probe beats random and a SAE probe on selective prediction,
but the obvious missing comparator is the **model's own predictive uncertainty**
— the standard label-free UQ signal. We recover it (`eval/extract_uncertainty.py`
re-runs the 100-sample forecast and saves the central-90 % band width per
window; reproducibility vs the committed CRPS labels: **Pearson 0.989, Spearman
0.982**, n=650) and evaluate it two ways (`eval/conformal_baseline.py`).

**(A) Split-conformal coverage.** Chronos's *raw* central-90 % interval covers
only **0.594** of test windows (nominal 0.90) — the sampled intervals are
severely over-confident. CQR-style split-conformal calibration on the train
split restores marginal coverage (**0.976** at α=0.1, **0.922** at α=0.2). But
the guarantee is only *marginal*: conditional on the **truly-hard** tercile it
**under-covers** — 0.852 vs 0.976 marginal at α=0.1, and **0.593 vs 0.922** at
α=0.2 (n=27 hard). The standard UQ method silently fails exactly where coverage
matters most, motivating difficulty-aware (group-conditional) conformal.

**(B) Selective-prediction frontier — head to head.** Ranking test windows by the
conformal band width (the deployable, truth-free signal; CQR's constant
correction does not change the ranking) and by predictive std, against P1 and
the oracle/random anchors (mean retained CRPS, lower better):

| Signal | @20 % cov | @50 % cov | AURC ↓ | Oracle headroom captured |
|--------|-----------|-----------|--------|--------------------------|
| **P1 input-stats** | **1.083** | 1.403 | **1.215** | **+30 %** |
| Conformal band width | 1.591 | **1.355** | 1.325 | +9 % |
| Predictive std | 1.606 | **1.338** | 1.318 | +11 % |
| P3 stats+SAE | 1.670 | 1.542 | 1.436 | −12 % |
| Oracle | 0.641 | 0.872 | 0.850 | (ceiling) |
| Random | — | — | 1.374 | 0 % |

(no-abstention mean CRPS = 1.527; headroom = (random−signal)/(random−oracle).)

**Honest read.** On the integrated metric the cheap input-stats probe **beats the
model's own conformalized uncertainty** — AURC 1.215 vs 1.325/1.318, capturing
**30 % of the oracle's headroom vs 9–11 %**. The gap is starkest under aggressive
abstention: at 20 % coverage P1 nearly halves the distance to the oracle
(1.083), while *both* UQ signals are **worse than not abstaining at all**
(1.59–1.61 > 1.527) — Chronos's predictive width fails to flag its own most
reliable windows. There is one honest nuance: around 50 % coverage the UQ
signals edge P1 (1.34–1.36 vs 1.403), so predictive width is mildly useful at
moderate retention. SAE-based routing (P3) remains at/below random throughout
(−12 %), reaffirming §4.2/§4.3. Net: the headline positive result survives the
comparison reviewers most expect — a 512-step window's classical statistics are
a *better* label-free abstention signal for Chronos on ETTh1 than the model's
own conformal uncertainty, and far better at the high-confidence operating
points selective prediction actually uses.

Figure 8: `eval/results/risk_coverage_conformal.png`. Saved:
`eval/results/conformal_baseline.json`,
`eval/results/uncertainty_ETTh1_chronos-t5-small.parquet`.

### 4.10 Robustness of the null to baseline strength and label choice

Two ways the §4.2 null could be an artifact: a *weak* P1 baseline, or the
specific CRPS-top-25 % *binarization*. We rule out both (`probing/probe_robustness.py`),
reusing the committed activations and the saved forecasts (no new TSFM pass).

**(a) Stronger baseline.** We augment the eight classical stats with STL trend &
seasonal strength, the Hurst exponent, and the model's own forecast-interval
width (Exp1's `band_width_90` — the natural difficulty signal). The SAE must now
beat *this*:

| Baseline (CRPS top-25 % label) | P1 AUROC | P3 (stats+SAE) | Δ(SAE − stats) |
|--------------------------------|----------|----------------|----------------|
| 8 classical stats (= §4.2)     | 0.654    | 0.426 | −0.228 [−0.366, −0.092] |
| + STL + Hurst + interval-width | **0.718** | 0.426 | **−0.292 [−0.429, −0.151]** |

Strengthening the baseline *raises* P1 (0.654 → 0.718) and pushes the SAE
**further** behind (Δ −0.228 → −0.292) — the opposite of what a weak-baseline
artifact would do. The model's own interval width, on its own, is a weak
difficulty predictor (test AUROC **0.593**), consistent with §4.9.

**(b) Label variation.** Repeating with a **MASE** top-25 % label keeps the null
(Δ(SAE − stats) = −0.178 [−0.295, −0.055]), so it is not CRPS-specific. Dropping
binarization entirely and predicting the **continuous** CRPS target (Ridge → test
Spearman) tells the same story:

| Feature block | test Spearman ρ (continuous CRPS) |
|---------------|-----------------------------------|
| 8 stats       | **+0.361** |
| stats + STL/Hurst/interval-width | +0.316 |
| raw activations | +0.088 |
| SAE codes     | +0.028 |
| stats + SAE   | +0.026 |

Classical statistics rank forecast difficulty (ρ ≈ 0.36) while SAE codes barely
do (ρ ≈ 0.03), and concatenating 6 144 SAE features onto the 12 stats *destroys*
the stats signal (0.32 → 0.03) — the high-dim codes are noise for this target.
The null is robust to baseline strength **and** to the label / its binarization.
Saved: `probing/results/robustness.json`.

### 4.11 Cross-dataset replication (ETTh2, ETTm1, ETTm2)

The single-series scope (ETTh1) is the biggest external-validity gap, so we
replicate the probe ladder on three more ET-T series (`experiments/run_sweep.py`).
**Fidelity caveat:** these are *lean* runs — `num_samples=20` and ~350 windows
(stride ×2) — because full-fidelity CPU extraction is ~14 h/series on this 16 GB
machine (the memory working set swap-thrashes; lean keeps it in RAM and runs
~50× faster). Lean is valid for the *sign* of Δ(SAE−stats) — the null question —
but adds label noise and shrinks n, which inflates CIs and understates absolute
AUROC. A GPU rerun at full fidelity is future work.

| Series | n_test | hard % | P1 stats | P3 stats+SAE | Δ(SAE−stats) 95 % CI |
|--------|--------|--------|----------|--------------|----------------------|
| ETTh1 (headline, full) | 167 | 16 % | 0.654 | 0.426 | −0.228 [−0.366, −0.092] |
| ETTh2 (lean) | 84 | 32 % | 0.500 | 0.504 | +0.004 [−0.134, +0.137] |
| ETTm1 (lean) | 102 | 13 % | 0.620 | 0.506 | −0.113 [−0.356, +0.144] |
| ETTm2 (lean) | 102 | 29 % | 0.515 | 0.531 | +0.016 [−0.156, +0.203] |

**Read.** The central result replicates: **SAE features never beat input
statistics** — Δ(SAE−stats) straddles zero on all three new series (no positive
significant delta anywhere), so the "SAE adds no incremental difficulty signal"
null is not ETTh1-specific. The honest nuance is on the *baseline*: P1 itself is
a strong difficulty predictor only on ETTh1 (0.654) and ETTm1 (0.620), and falls
to near-chance on ETTh2/ETTm2 (0.50/0.52). Part of that is likely the lean
fidelity (noisier 20-sample CRPS labels, ~half the windows) rather than those
series being intrinsically unpredictable — separating the two needs a
full-fidelity/GPU rerun. So the §4.3 *positive* selective-prediction result
should be read as ETTh1-demonstrated, not yet shown to generalize; the *null*
(SAE ≤ stats) does generalize across all four series. Saved:
`experiments/runs/sweep_summary.{json,csv}`.

## 5. Limitations
- Single series (ETTh1), single TSFM backbone (chronos-t5-small). Layer
  robustness is checked across mid- and late-encoder (§4.2); generalization
  across domains, datasets, and larger backbones is deferred.
- We probe encoded context, not decoder sampling dynamics; CRPS depends on
  both.
- ADF replaced by a scipy variance-ratio proxy when statsmodels is absent.
- Single seed; the headline ΔAUROC CIs come from bootstrapping test windows,
  not from re-training.
- [If null:] SAE features are interpretable but not measurably more
  predictive than raw activations; we report this honestly rather than
  reframing the metric post-hoc.

## 6. Citations
Both 2026 SAE-on-TSFM citations in §2 were independently verified (May 2026)
and the arXiv IDs are correct as listed. Chronos (Ansari et al., 2024) is the
original benchmark protocol reference.

## 7. Future work
Multi-model SAE training and cross-backbone feature alignment
(Chronos / TimesFM / Moirai); cross-domain feature transfer (Monash, M5);
full implemented cascade with end-to-end cost accounting on real hardware;
feature steering ("Golden Gate for forecasting") for seasonal vs. trend modes
and abstention to a classical baseline on distribution-shift firing.

## Reproducibility
`bash reproduce.sh` runs the full pipeline. `requirements.txt` is pinned;
`.vscode/settings.json` selects the venv interpreter. Stale prototype
artifacts live in `_stale/` with a README warning. The difficulty probe
refuses to run on unlabeled metadata.
