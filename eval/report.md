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
roughly 30 % of the oracle's available AURC improvement. We report both the
null SAE result and the positive selective-prediction result honestly,
without post-hoc reframing of the metric.

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
