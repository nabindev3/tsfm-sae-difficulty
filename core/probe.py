"""The unified probe ladder — modality-agnostic.

This is the component the whole project lives or dies on: both modalities MUST
report the *same* ladder so the cross-modal claim is apples-to-apples.

The ladder (matching what both legacy repos already computed):

    P1_cheap        cheap baseline only      (lexical stats | classical TS stats)
    P2_cheap_raw    cheap + raw activations  (the crucial MIDDLE rung)
    P3_cheap_sae    cheap + SAE codes
    P4_raw_only     raw activations only      (diagnostic: where does signal live?)
    P5_sae_only     SAE codes only            (diagnostic)

Headline deltas (with paired-bootstrap CIs):
    Δ(P2 - P1)  incremental power of raw activations over cheap
    Δ(P3 - P1)  incremental power of SAE over cheap
    Δ(P3 - P2)  incremental power of SAE OVER RAW  <-- the money number

The probe is an L1-penalized logistic regression with C chosen by inner CV.
The caller passes the CV splitter, so the LLM modality hands in a stratified
KFold and the TSFM modality hands in a TimeSeriesSplit — same code, correct
leakage control for each.

Nothing here imports torch or any model: features come in as numpy arrays.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .stats import LadderResult, ProbeResult, paired_bootstrap_auroc

# Canonical rung names. Adapters fill in cheap_features / raw_agg / sae_agg.
P1, P2, P3, P4, P5 = (
    "P1_cheap",
    "P2_cheap_raw",
    "P3_cheap_sae",
    "P4_raw_only",
    "P5_sae_only",
)

DEFAULT_C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]

# The three headline comparisons, as (numerator, reference) rung pairs.
HEADLINE_PAIRS = [(P2, P1), (P3, P1), (P3, P2)]


def build_ladder(
    cheap: np.ndarray,
    raw_agg: np.ndarray,
    sae_agg: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the five feature matrices from the three building blocks."""
    return {
        P1: cheap,
        P2: np.concatenate([cheap, raw_agg], axis=1),
        P3: np.concatenate([cheap, sae_agg], axis=1),
        P4: raw_agg,
        P5: sae_agg,
    }


def run_probe_ladder(
    features: dict[str, np.ndarray],
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    c_grid: list[float] = DEFAULT_C_GRID,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[LadderResult, dict[str, np.ndarray]]:
    """Fit every rung, pick C by inner CV, paired-bootstrap the AUROCs & deltas.

    Parameters
    ----------
    features   : {rung_name: (N, d) matrix}; use `build_ladder` for the standard 5.
    y          : (N,) binary difficulty labels.
    train_mask : (N,) bool.
    test_mask  : (N,) bool.
    cv_splits  : pre-materialized inner-CV folds over the TRAIN rows, e.g.
                 list(StratifiedKFold(...).split(...)) for the LLM modality or
                 list(TimeSeriesSplit(...).split(...)) for the TSFM modality.
                 Passing folds (not a splitter) keeps this modality-agnostic.

    Returns
    -------
    LadderResult, and {rung_name: test-set P(hard) predictions} for downstream
    calibration / selective-prediction / cascade stages.
    """
    y = np.asarray(y)
    y_train, y_test = y[train_mask], y[test_mask]
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        raise ValueError("Empty train or test split.")

    preds: dict[str, np.ndarray] = {}
    point_auroc: dict[str, float] = {}
    best_C: dict[str, float] = {}

    for name, X in features.items():
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])

        # random_state pins liblinear's internal coordinate-descent ordering;
        # without it the same seed yields slightly different fits run-to-run.
        base = LogisticRegression(
            penalty="l1", solver="liblinear",
            class_weight="balanced", max_iter=2000, random_state=seed,
        )
        # n_jobs=-1 parallelizes the (C × fold) grid of fits across threads; the
        # threading backend avoids fork deadlocks on Apple Silicon. The liblinear
        # "n_jobs has no effect" notice refers to the solver's *internal*
        # threading (irrelevant here), so we scope-suppress that one message.
        gs = GridSearchCV(base, {"C": c_grid}, scoring="roc_auc",
                          cv=list(cv_splits), n_jobs=-1)
        with warnings.catch_warnings(), parallel_backend("threading"):
            warnings.filterwarnings("ignore", message=".*n_jobs.*liblinear.*")
            gs.fit(X_tr, y_train)
        p = gs.predict_proba(X_te)[:, 1]
        preds[name] = p
        point_auroc[name] = (
            float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else 0.0
        )
        best_C[name] = float(gs.best_params_["C"])

    # Paired bootstrap: AUROC CIs for every rung + delta CIs for the headline pairs.
    pairs = [(a, b) for a, b in HEADLINE_PAIRS if a in features and b in features]
    auroc_ci, delta_ci = paired_bootstrap_auroc(
        y_test, preds, pairs, n_boot=n_boot, seed=seed
    )

    n_test = int(test_mask.sum())
    probes = {
        name: ProbeResult(
            name=name,
            auroc=point_auroc[name],
            ci_low=auroc_ci[name][0],
            ci_high=auroc_ci[name][1],
            n_test=n_test,
            best_C=best_C[name],
        )
        for name in features
    }

    deltas = {}
    for a, b in pairs:
        key = f"{a}-{b}"
        deltas[key] = {
            "point": point_auroc[a] - point_auroc[b],
            "ci_low": delta_ci[key][0],
            "ci_high": delta_ci[key][1],
        }

    result = LadderResult(
        n_total=len(y),
        n_train=int(train_mask.sum()),
        n_test=n_test,
        hard_fraction=float(y_test.mean()),
        probes=probes,
        deltas=deltas,
    )
    return result, preds
