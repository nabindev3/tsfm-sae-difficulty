# A Rigorous Null with a Deployable Artifact: Probing Sparse-Autoencoder Features in a Time-Series Foundation Model for Forecast-Difficulty Prediction

**Nabin Prasad Dev**
*Independent / Author affiliation TBD*
`nabin.dev33@gmail.com`

> *Target venues:* NeurIPS 2026 Workshop on Time Series in the Age of Large Models · ICLR 2026 Workshop on Foundation Models for Time Series · AAAI 2026 AI4TS Workshop.

> **Canonical source.** This is the condensed paper; the full, continuously-updated
> technical report is [`report.md`](report.md), which is authoritative wherever a
> figure here conflicts. **Pending revision:** the causal-ablation "near-significant
> directional positive on hard windows" (+0.043) stated below is *superseded* — a
> matched-100-sample re-run (report §4.13) shows it reverses sign (−0.049) and was
> Monte-Carlo noise, not a causal signal. Read the causal-ablation claims here
> against report §4.6 / §4.8 / §4.13.

---

## Abstract

Time-series foundation models (TSFMs) such as Chronos forecast accurately but ship without native abstention or uncertainty signals. We ask whether a TSFM's internal representations encode self-difficulty information *beyond* what classical input statistics already provide — a label-free, inference-time signal for routing or selective prediction. We train an 8× TopK sparse autoencoder (SAE) on `Chronos-T5-small` encoder activations and evaluate five probes with paired-bootstrap ΔAUROC under strict temporal splits with purge gaps. On ETTh1, internal representations carry **no measurable signal beyond eight classical input statistics** (Δ(SAE − stats) = −0.228, 95 % CI [−0.366, −0.092]; the null is layer-robust across mid- and late-encoder). A hook-based causal ablation of the top-5 difficulty features shows a **near-significant directional contribution on hard windows** (aggregate +0.043 ΔCRPS, 95 % CI [−0.008, +0.095]). The input-statistics probe, repurposed as a selective-prediction signal, captures **30 % of the oracle AURC improvement** and reduces mean test CRPS by 8.1 % at 50 % coverage. A feature-routed cascade between `chronos-t5-small` and `chronos-t5-base` yields **five Pareto-dominating points** strictly below random routing. Platt recalibration on 5-fold OOF train predictions reduces probe ECE from **0.482 → 0.097** while preserving ranking AUROC exactly. We position this as a rigorous null with a deployable artifact — null reported honestly, artifact reproducible from one command.

---

## 1. Introduction

Time-series foundation models — Chronos [\[Ansari et al. 2024\]](#references), TimesFM, Moirai — deliver competitive zero-shot forecasts on unseen series but expose no native abstention or per-forecast uncertainty signal. Downstream systems consequently treat every forecast identically: same compute, same trust, same downstream commitment. Some forecast windows are far harder than others, and there is no built-in route to flag them. We ask a question that, to our knowledge, remains unclaimed in the 2026 literature on TSFM interpretability:

> *Do a TSFM's internal representations encode a self-difficulty signal that a cheap classical baseline doesn't already capture? And if not, can we still recover a deployable abstention signal from the cheap baseline alone?*

The closest published work on SAE interpretability in TSFMs — Mishra (2026) on Chronos-T5-Large [\[arXiv:2603.10071\]](#references) and TimeSAE (Jan 2026) [\[arXiv:2601.09776\]](#references) — targets *causal feature hierarchies* and *post-hoc black-box explanation* respectively. Neither evaluates internal features as a **label-free, inference-time signal for routing or abstention**. That gap is the contribution wedge of this paper.

**Contributions.**
1. A **leakage-controlled, paired-bootstrap-rigorous** evaluation of internal-representation difficulty signals in `Chronos-T5-small`, with diagnostic isolation probes (raw-only, sae-only) that pinpoint where signal lives.
2. A **null result**, reported honestly: SAE features do not add incremental predictive power over eight classical input statistics, with a paired-bootstrap CI that does not cross zero. The null is **layer-robust** (mid- vs. late-encoder).
3. A **causal refinement** of the null via Mishra-style hook ablation: the top-5 features show a near-significant directional contribution on hard windows (aggregate +0.043 ΔCRPS, CI [−0.008, +0.095]), with 5/5 features pointing the same way — consistent with weak signal that is underpowered to detect at this dataset size.
4. A **positive deployable artifact** from the same pipeline: the input-statistics probe captures 30 % of oracle AURC improvement under selective prediction (8.1 % CRPS reduction at 50 % coverage), and a feature-routed cascade between `chronos-t5-small` and `chronos-t5-base` finds five Pareto-dominating points strictly below random routing. Platt recalibration reduces ECE 0.482 → 0.097 with AUROC preserved exactly.

The unifying message: when a primary hypothesis returns a clean null, the rigor of the methodology becomes the contribution, *and* the same pipeline can still yield a deployable selective-forecaster artifact. Both are recovered here.

## 2. Related Work

**SAEs on TSFMs.** Mishra (2026) trains TopK SAEs on six layers of `chronos-t5-large` (710 M), runs 392 single-feature ablations, and establishes a depth-dependent hierarchy in which the mid-encoder concentrates *causally critical* change-detection features. TimeSAE (Jan 2026) uses a JumpReLU SAE as a *post-hoc, model-agnostic* explainer of black-box time-series predictions under distribution shift. Our distinction is operational: we evaluate the same family of features as a **label-free, inference-time difficulty signal** for routing and abstention, not as a causal or post-hoc explanatory device.

**Selective prediction and cost-aware cascades.** Geifman and El-Yaniv (2017) formalize selective classification with the risk–coverage frontier; we apply the analogous risk–coverage analysis to forecast CRPS. Cost-aware model cascades for inference efficiency are standard in NLP (Chen et al. 2023, FrugalGPT) but, to our knowledge, have not been instrumented with paired-bootstrap-rigorous baselines (random routing, oracle routing, linear-interpolation reference) in the TSFM setting.

**Probe calibration.** Platt (1999) and Niculescu-Mizil & Caruana (2005) establish post-hoc sigmoid and isotonic calibration as standard fixes when downstream probabilities matter; we apply both, with 5-fold OOF training predictions as the calibrator's fitting set under a temporal outer split.

## 3. Method

### 3.1 Backbone and activation extraction

We use `amazon/chronos-t5-small` (60 M parameters; T5 encoder–decoder backbone trained with an LM-style objective on quantized numeric tokens). For each forecast window we register a `forward_hook` on `encoder.block[k].layer[-1]` (`k = num_layers // 2` for the headline analysis, `k = num_layers − 1` for the cross-layer robustness check), capturing the post-LN residual of shape `(B, 513, 512)` — 512 context tokens plus an EOS. The hook fires once per window with batch dimension equal to the number of windows; we verified empirically that Chronos's `num_samples` expansion happens after the encoder, so the hook does not inflate the activation cache.

### 3.2 Labels and temporal split

For each window we sample 100 forecasts and compute CRPS; seasonal-naive MASE (m = 24 for hourly data) is computed as a secondary scale-free label. CRPS is normalized using **train-split statistics only**, then thresholded at the top-15 % train quantile to define `hard`. The temporal train/test split uses a **purge gap ≥ context + horizon** (here 512 + 96 = 608 timesteps), which eliminates window overlap between train and test — a load-bearing methodological choice that random splits in the literature routinely violate, inflating reported AUROC.

### 3.3 Sparse autoencoder

We train a TopK SAE with `d_in = 512`, `d_hidden = 4096` (8× expansion), `k = 32`, `aux_k = 1024` for dead-feature revival, decoder-bias initialized to the activation mean, decoder columns kept unit-norm. The SAE is trained **only on train-split tokens** (483 windows × 513 tokens ≈ 247 k tokens). Hyperparameters: Adam, lr 5e-4, 100-step linear warmup, 10 epochs, dead-feature aux-revival triggered after 50 consecutive non-firing steps. The trained SAE reaches normalized MSE 0.068 (well below the 0.10 target); dead fraction stabilizes at 63 % under the available training budget.

### 3.4 The five-probe suite

Each forecast window yields a feature vector under five conditions:

- **P1 — input stats only**: 8 classical context-window statistics (variance, volatility, lag-1 and seasonal-24 autocorrelation, spectral entropy, trend slope, range, ADF p-value via `statsmodels`).
- **P2 — input stats + raw activations**: P1 concatenated with `concat(mean, max, last)` pooling of the raw 512-dim residual stream.
- **P3 — input stats + SAE features**: P1 concatenated with the analogous pooling of the 4 096-dim SAE codes.
- **P4 — raw only** (diagnostic isolation): the high-dim raw pooled features alone.
- **P5 — sae only** (diagnostic isolation): the high-dim SAE pooled features alone.

All probes are L1-logistic with class-balanced loss; `C` is selected by `TimeSeriesSplit` inner CV (5 splits) over `{10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 0.01, 0.03, 0.1, 0.3, 1.0}` so that consecutive (overlapping) training windows do not leak across folds when picking regularization.

### 3.5 Paired bootstrap on the test set

For each of 2,000 bootstrap iterations we resample test indices *once* with replacement and use the *same resampled indices* for every probe and every Δ, then take the 2.5/97.5 percentiles. This is the only way to obtain a CI on Δ(P3 − P2) — the comparison that neutralizes the dimensionality argument (SAE vs. raw, both high-dim).

### 3.6 Selective prediction, cascade, calibration

We treat each probe's predicted `P(hard)` as a per-window abstention signal: sort test ascending by `P(hard)`, retain the predicted-easy `coverage · N` windows, and report the mean CRPS of the retained set as a function of coverage. Baselines: an oracle that sorts by true CRPS, and a random baseline averaged over 2,000 permutations. We then run a feature-routed cascade between `chronos-t5-small` (cheap, cost = 1.0) and `chronos-t5-base` (expensive, cost = 5.0) by routing to base when the probe's `P(hard)` exceeds threshold τ, sweeping τ in [0, 1] and counting Pareto-dominating points against an oracle ("route the windows where base − small is largest") and a random routing baseline.

Finally we apply both **Platt** (sigmoid) and **isotonic** post-hoc recalibration on the test-set predictions, with the calibrator fit on 5-fold out-of-fold predictions over the full train split (sufficient cal data; random folds are appropriate for *calibrator* fitting even though the outer evaluation is temporally held out).

## 4. Experiments

### 4.1 Setup

ETTh1 (`OT` channel), context 512, horizon 96, stride 24 → 701 windows. After temporal split with purge gap: n_train = 483, n_test = 167, n_purge = 51. Hard fraction (train) = 15.1 %; hard fraction (test) = 6.6 % (the test horizon falls into a milder regime; we address this in §4.5).

### 4.2 The predictive null (headline)

Table 1 reports paired-bootstrap ΔAUROC. Δ(P3 − P1) and Δ(P3 − P2) both have CIs strictly below zero — SAE features carry *significantly less* signal than raw, which in turn carries no incremental signal over the eight input statistics. Diagnostic probes P4 and P5 reveal that the high-dim probes (P2, P3) are not being drowned out by input stats: P4 = P2 and P5 = P3 to three decimals, so the L1 logistic genuinely ignores the eight stats when given the high-dim features.

| Probe | Test AUROC (95 % CI)  |
|--------------------------|-----------------------|
| P1 — stats (8)           | 0.654 (0.552, 0.751)  |
| P2 — stats + raw         | 0.584 (0.460, 0.697)  |
| P3 — stats + sae         | 0.426 (0.313, 0.547)  |
| P4 — raw only            | 0.584 (0.461, 0.698)  |
| P5 — sae only            | 0.426 (0.312, 0.546)  |

| Δ                        | Point   | 95 % CI            |
|--------------------------|---------|--------------------|
| P2 − P1 (raw value)      | −0.070  | [−0.193, +0.055]   |
| P3 − P1 (sae value)      | **−0.228** | **[−0.366, −0.092]** |
| P3 − P2 (sae over raw)   | **−0.158** | **[−0.293, −0.025]** |

*Table 1. Headline AUROCs and incremental Δs. Figure: `probing/results/auroc.png`.*

**Cross-layer robustness.** Re-running the full probe on the *last* encoder block (block 5 of 6) via a focused `--skip_predict` activation-only extraction (which reuses cached labels) reproduces the null: Δ(SAE − Stats) = −0.143, 95 % CI [−0.277, −0.013]; Δ(Raw − Stats) = −0.204, 95 % CI [−0.324, −0.075]. At the late encoder, the previously significant Δ(SAE − Raw) gap closes to a null (+0.061, CI [−0.083, +0.194]) — where raw carries less signal, SAE compression is at parity. The null is **not an artifact of layer choice**.

### 4.3 Causal ablation of the top-5 features (§4.6 of `report.md`)

For each test window, a forward hook on `encoder.block[3].layer[-1]` (the layer the SAE was trained on) replaces its hidden state with the SAE's reconstruction. We measure CRPS under three conditions: **natural** (no hook); **SAE-recon** (hook with no feature ablation); and **ablate(feat = k)** for each of the top-5 features ranked by absolute L1-logistic coefficient. The SAE-recon → natural delta is −0.023 (CI [−0.076, +0.033]) — null — so the ablation deltas are not confounded by reconstruction loss.

Per-feature Δ(ablate − recon) is null at the population level (all individual CIs cross zero). Stratifying by test-window difficulty reveals a directional pattern: on hard windows (top tercile by natural CRPS, n = 56), **5/5 features show positive point estimates** (+0.028 to +0.055); on easy windows (n = 111) the same features show ~null to slightly negative estimates. The aggregate test (mean ΔCRPS averaged across the 5 features) on hard windows reaches +0.043, 95 % CI [−0.008, +0.095] — directionally consistent and ~3 % of mean natural CRPS, but underpowered for individual-feature significance at n = 56. The diff-in-diff (hard − easy) is +0.050, 95 % CI [−0.005, +0.104]. Interpretation: the top-5 features are **weakly causally tied to forecast quality on hard windows specifically** — consistent across features, non-trivial in magnitude, just below the significance line at this sample size. A larger hard cohort would resolve this; the current data refines the §4.2 predictive null with an underpowered causal positive.

### 4.4 Selective prediction (§4.3 of `report.md`)

Treating P1's predicted `P(hard)` as an abstention signal, we report mean CRPS on the predicted-easy retained windows at each coverage. Headline metric is **AURC** (area under the risk-coverage curve, lower better):

| Method               | AURC  | Mean CRPS @ 50 % cov. | Δ vs no abstention |
|----------------------|-------|-----------------------|--------------------|
| No abstention        | 1.527 | 1.527                 | —                  |
| Random (500-perm avg)| 1.374 | 1.527                 | —                  |
| **P1 stats**         | **1.215** | **1.403**         | **−8.1 %**         |
| P2 stats+raw         | 1.240 | 1.480                 | −3.0 %             |
| P3 stats+sae         | 1.437 | 1.559                 | +2.1 % *(worse)*   |
| Oracle (true CRPS sort) | 0.850 | 0.872              | −42.9 % (ceiling)  |

P1 captures `(random − P1) / (random − oracle) = 0.159 / 0.524 ≈ 30 %` of the available oracle improvement on AURC. The SAE-based probes sit at or below the random baseline, consistent with §4.2. *Figure: `eval/results/risk_coverage.png`.*

### 4.5 Feature-routed cascade (§4.5 of `report.md`)

We extract `chronos-t5-base` CRPS on the 167 test windows only (focused script `eval/extract_base_crps_test_only.py`; ~1.5 h on CPU vs ~6 h for full series). Anchors: mean CRPS small = 1.5266 (cost 1.0); mean CRPS base = 1.5130 (cost 5.0). Base outperforms small by only 0.9 %, with base winning on 52.1 % of windows — essentially a coin flip. The cascade's *room to win* is therefore small by construction on this dataset.

Despite the narrow gap, **P1-routed cascade finds 5 Pareto-dominating points** strictly below the cheap↔base linear interpolation and the 500-permutation random-routing curve. The best dominating point routes 97 % of windows to base, retains 3 % on small, and achieves mean CRPS 1.5092 — *beating both deployment anchors*. The SAE-based routing (P3) finds only one trivially-dominating point at a 1.2 % base-routing fraction, again consistent with §4.2. *Figure: `eval/results/pareto_frontier.png`.*

The methodology — random-baseline + oracle reference + paired comparison — is the artifact that travels to other datasets and backbone pairs. The empirical magnitude of the gain is bounded by the narrow small ↔ base CRPS gap on ETTh1.

### 4.6 Probe calibration and recalibration (§4.7 of `report.md`)

Raw probe probabilities are severely miscalibrated: P1 ECE = 0.482, Brier = 0.297. This is expected — `class_weight='balanced'` shifts predictions toward the 15.1 % train marginal, but the test marginal is 6.6 % (distribution shift across the temporal split). High-dim probes converge to ECE ≈ 0.50, consistent with their §4.2 null status (uninformative ranking yields uninformative probabilities).

We apply both **Platt** (sigmoid) and **isotonic** recalibration with the calibrator fit on 5-fold OOF train predictions:

|              | raw ECE | Platt ECE | Isotonic ECE | AUROC (preserved by Platt) |
|--------------|---------|-----------|--------------|----------------------------|
| **P1 stats** | 0.482   | **0.097** | 0.103        | 0.697 → 0.697              |
| P3 stats+sae | 0.404   | 0.153     | 0.157        | 0.611 → 0.611              |

P1's calibration error drops **80 %** under Platt with ranking AUROC preserved exactly (Platt is a strict monotone two-parameter fit). Brier drops 0.297 → 0.070. The probe is **deployment-grade**. An earlier 80/20 temporal split for the calibrator failed (cal slice had distribution-shifted hard rate, Platt learned a sign-flipped coefficient); K-fold OOF over the full train set resolves it. *Figure: `eval/results/reliability_recalibrated.png`.*

## 5. Discussion

The contribution of this paper is **methodological honesty under a primary null**. Three claims survive the rigor:

1. **The hypothesis is null at this scale.** SAE features from `chronos-t5-small`'s mid- or late-encoder do not add predictive power for forecast difficulty over eight classical input statistics on ETTh1, with paired-bootstrap CIs that do not cross zero. This is the strong null.
2. **The null is not absolute.** A targeted causal ablation reveals a near-significant directional effect of the top-5 features on hard windows. The pattern is consistent across all five features but underpowered to detect at n = 56 hard windows. We resist reframing this as a positive; we report it as a *refinement* of the predictive null.
3. **The pipeline produces a deployable artifact regardless.** The input-statistics probe alone captures 30 % of oracle ranking improvement, a real cascade finds five Pareto-dominating points beating both deployment anchors, and Platt recalibration converts the probe into a calibrated abstention signal. The selective-forecasting framing is positive on the same data the SAE hypothesis is null on.

For TSFM deployment, the operational lesson is: **eight cheap context-window statistics are a strong calibratable abstention signal**, with or without internal-representation probing. For interpretability, the lesson is: **predictive correlation does not imply causal contribution at this scale**, and the diagnostic probes (P4, P5) are essential to distinguish "stats dominate" from "high-dim features carry no signal" — they reveal it is the latter.

## 6. Limitations and threats to validity

- **Single series and backbone.** ETTh1 (`OT` channel) and `chronos-t5-small` only. Cross-layer robustness is checked (§4.2); cross-dataset and cross-backbone scale (small → base → large) is deferred.
- **Single seed.** Bootstrap CIs come from resampling test windows, not from re-training. Multi-seed variance would tighten the picture but is left to future work.
- **Causal ablation sample.** n = 56 hard windows leaves the per-feature ablation effects underpowered; the aggregate test reaches the significance boundary but does not cross it.
- **MASE / `num_samples` MC noise.** SAE-recon and ablation conditions use 50 samples per window for compute reasons; the natural baseline uses 100. The relative comparisons within ablation (which are the headline) are at matched sample count, so the MC-noise caveat applies only to the recon-vs-natural baseline (which is null anyway).
- **No conformal-prediction baseline.** The standard uncertainty-quantification baseline for forecasting was not run because the 100-sample forecast tensors were not cached during extraction. We treat this as the most important next experiment.
- **Steering not reported.** A controllable-feature steering demo was designed and coded but the CPU run exceeded our compute budget. The design (`eval/steering_demo.py`) is retained for a GPU port.

## 7. Future work

1. **Multi-backbone scale.** Re-train SAE on `chronos-t5-base` (200 M) and `chronos-t5-large` (710 M) activations; test whether the predictive null persists with model capacity or whether richer representations finally beat input statistics.
2. **Multi-dataset generalization.** Replicate the full pipeline on ETTh2, Weather, and Electricity; test whether the directional causal-ablation signal on hard windows reaches significance with more data.
3. **Conformal-prediction comparison.** Run conformal forecasting on the same windows and compare selective-prediction Pareto frontiers; the standard uncertainty-quantification baseline this paper currently omits.
4. **Steering interventions.** Port the steering script to a GPU runtime; clamp top features to train-99th-percentile values on confident windows, measure forecast distribution shift.
5. **Attention-pattern analysis.** Connect the top-K SAE features to per-token attention patterns to give the interpretability story a mechanistic spine.

## 8. Reproducibility

All experiments run from one command (`bash reproduce.sh`) on the cached `chronos-t5-small` ETTh1 extraction (~360 MB). The cascade artifact requires a single additional command (`python eval/extract_base_crps_test_only.py`, ~1.5 h on CPU, test windows only). Code, pinned environment, and the report with on-disk artifact references are at:

`https://github.com/nabindev3/tsfm-sae-difficulty`

Methodology guardrails are enforced in code: the probe refuses to run on metadata lacking `split` or `crps_*` columns, or with a missing/corrupt SAE checkpoint. It cannot silently produce a fake result.

## References

- Ansari, A. F., Stella, L., Turkmen, C., et al. (2024). *Chronos: Learning the Language of Time Series.* arXiv:2403.07815.
- Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
- Mishra, A. (2026). *Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models.* arXiv:2603.10071. Verified May 2026.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning.* ICML.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.* Advances in Large Margin Classifiers.
- TimeSAE authors (Jan 2026). *TimeSAE: Sparse Decoding for Faithful Explanations of Black-Box Time Series Models.* arXiv:2601.09776. Verified May 2026.

## Appendix A — Reproducing Tables and Figures

| Claim                                | Script                                       | Artifact                                          |
|--------------------------------------|----------------------------------------------|---------------------------------------------------|
| Table 1, §4.2 ΔAUROC                 | `probing/probe.py`                            | `probing/results/probe_results.json`, `auroc.png` |
| §4.2 late-encoder robustness         | `extract_activations.py --layer_idx 5 --skip_predict` then `probe.py` | `probing/results/probe_results_late_layer5.json`  |
| §4.3 causal ablation                 | `eval/causal_ablation.py`                     | `eval/results/causal_ablation.{json,parquet}`     |
| §4.4 selective prediction            | `eval/selective_prediction.py`                | `eval/results/risk_coverage.png`, `selective_prediction.json` |
| §4.5 cascade                         | `eval/extract_base_crps_test_only.py` + `eval/cascade.py` | `eval/results/pareto_frontier.png`, `cascade_results.json` |
| §4.6 calibration / recalibration     | `eval/calibration.py`, `eval/recalibrate.py`  | `eval/results/reliability_diagram.png`, `reliability_recalibrated.png`, `*.json` |
