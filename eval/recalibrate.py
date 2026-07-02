"""Post-hoc recalibration of probe probabilities.

§4.7 showed the L1-logistic probes are well-ranked (AUROC) but poorly
calibrated (ECE 0.38–0.56) because `class_weight='balanced'` shifts
probabilities to the train marginal (~15 %), while test marginal is 6.6 %.

We fix it with **isotonic regression**, fit on out-of-fold predictions on the
train split so calibration never sees the test set. Apply the calibrator to
the test predictions and re-measure ECE / Brier. Ranking AUROC is preserved
by construction (isotonic is monotone).

This turns §4.7's diagnosis into a deployable fix.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.probe import compute_input_stats   # reuse the same 8-feature input stats
from sae.sae_model import TopKSAE


def compute_ece_brier(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(ece), float(np.mean((p - y) ** 2))


def reliability_pts(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out_x, out_y = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out_x.append(float(p[m].mean()))
        out_y.append(float(y[m].mean()))
    return out_x, out_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="activations/ETTh1_activations.safetensors")
    ap.add_argument("--metadata", default="activations/ETTh1_metadata.parquet")
    ap.add_argument("--probe_scores", default="activations/probe_scores.parquet")
    ap.add_argument("--sae_ckpt", default="sae/checkpoints/sae_topk_32.pt")
    ap.add_argument("--hard_quantile", type=float, default=0.85)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading metadata, activations, SAE...")
    meta = pd.read_parquet(args.metadata)
    acts = load_file(args.activations)["encoder_embeddings"]
    state = torch.load(args.sae_ckpt, map_location="cpu", weights_only=True)
    d_model, d_hidden = state["W_enc"].shape
    sae = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=32)
    sae.load_state_dict(state)
    sae.eval()

    label_col = "crps_norm" if "crps_norm" in meta.columns else "crps_raw"
    tr = meta["split"].values == "train"
    te = meta["split"].values == "test"
    thr = np.quantile(meta.loc[tr, label_col].values, args.hard_quantile)
    y = (meta[label_col].values >= thr).astype(int)
    y_tr, y_te = y[tr], y[te]
    print(f"  n_train={tr.sum()}  n_test={te.sum()}  "
          f"hard_frac_train={y_tr.mean():.3f}  hard_frac_test={y_te.mean():.3f}")

    # Recompute the 8 input statistics for the FULL series (train + purge + test)
    print("Computing input statistics (this is the only non-trivial step) ...")
    input_stats = compute_input_stats(meta, context_length=512)

    # Set up the same probe shape as probing/probe.py — focus on P1 (the best
    # ranker) and P3 (the SAE one) because P2/P4/P5 add no AUROC over P1/P3.
    probes = {"P1_InputStats": input_stats}

    # P3 requires SAE max-pool aggregate — recompute. Same shape as probe.py.
    print("Computing SAE-aggregated features (mean+max+last) for P3 ...")
    sae_agg_chunks = []
    with torch.no_grad():
        for i in range(acts.shape[0]):
            w = acts[i:i + 1].to(torch.float32)
            c, _, _ = sae(w.reshape(-1, w.shape[-1]))
            c = c.reshape(w.shape[1], -1)
            agg = torch.cat([c.mean(0), c.max(0).values, c[-1]]).numpy()
            sae_agg_chunks.append(agg)
    sae_agg = np.stack(sae_agg_chunks)
    probes["P3_InputStats_SAE"] = np.concatenate([input_stats, sae_agg], axis=1)

    # K-fold OOF on the full train (random folds). Time-series purity is
    # already enforced for the outer test split; here we just need a
    # representative cross-section of train predictions to fit the calibrator,
    # not an evaluation. The 80/20 temporal split we tried first gave only
    # ~97 cal samples and was corrupted by distribution shift across the
    # boundary (cal hard-rate diverged from inner-train hard-rate), so Platt
    # learned a sign-flipped coefficient. K-fold avoids that pathology.
    from sklearn.model_selection import KFold
    n_tr = int(tr.sum())
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)

    for ax, (name, X) in zip(axes, probes.items()):
        print(f"\n--- {name} (features: {X.shape[1]}) ---")
        scaler = StandardScaler()
        X_tr_full = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        C_pick = 1.0 if name == "P1_InputStats" else 0.1

        # 5-fold OOF predictions on train -> p_cal of length n_train
        p_cal = np.zeros(n_tr, dtype=float)
        for fold_tr_idx, fold_te_idx in kf.split(X_tr_full):
            clf = LogisticRegression(penalty="l1", solver="liblinear",
                                     class_weight="balanced", max_iter=2000,
                                     C=C_pick)
            clf.fit(X_tr_full[fold_tr_idx], y_tr[fold_tr_idx])
            p_cal[fold_te_idx] = clf.predict_proba(X_tr_full[fold_te_idx])[:, 1]
        y_cal = y_tr  # calibrator sees all of train via OOF

        # Final model fit on FULL train for test predictions
        base = LogisticRegression(penalty="l1", solver="liblinear",
                                  class_weight="balanced", max_iter=2000,
                                  C=C_pick)
        base.fit(X_tr_full, y_tr)
        p_te_raw = base.predict_proba(X_te)[:, 1]

        # Two calibrators:
        # (a) Isotonic — minimizes ECE but suffers from clipping ties when cal
        #     set is small relative to test range.
        # (b) Platt / sigmoid — 2-parameter monotone fit. Strictly preserves
        #     ranking AUROC. Standard choice when calibration data is scarce.
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        p_te_iso = iso.transform(p_te_raw)

        # Platt: fit a logistic regression on (p_cal, y_cal) and apply.
        platt = LogisticRegression(C=1e6)  # weak reg, near-MLE
        platt.fit(p_cal.reshape(-1, 1), y_cal)
        p_te_pl = platt.predict_proba(p_te_raw.reshape(-1, 1))[:, 1]

        ece_b, brier_b = compute_ece_brier(y_te, p_te_raw)
        ece_iso, brier_iso = compute_ece_brier(y_te, p_te_iso)
        ece_pl, brier_pl = compute_ece_brier(y_te, p_te_pl)
        unique_y_te = len(np.unique(y_te)) > 1
        auroc_b = float(roc_auc_score(y_te, p_te_raw)) if unique_y_te else None
        auroc_iso = float(roc_auc_score(y_te, p_te_iso)) if unique_y_te else None
        auroc_pl = float(roc_auc_score(y_te, p_te_pl)) if unique_y_te else None
        results[name] = {
            "raw":     {"ece": ece_b,   "brier": brier_b,   "auroc": auroc_b},
            "platt":   {"ece": ece_pl,  "brier": brier_pl,  "auroc": auroc_pl},
            "isotonic":{"ece": ece_iso, "brier": brier_iso, "auroc": auroc_iso},
        }
        print(f"  raw      ECE {ece_b:.3f}  Brier {brier_b:.3f}  AUROC {auroc_b:.3f}")
        print(f"  Platt    ECE {ece_pl:.3f}  Brier {brier_pl:.3f}  AUROC {auroc_pl:.3f}   (monotone, preserves AUROC)")
        print(f"  isotonic ECE {ece_iso:.3f}  Brier {brier_iso:.3f}  AUROC {auroc_iso:.3f}   (lower ECE; AUROC may suffer from clipping ties)")

        bx, by = reliability_pts(y_te, p_te_raw)
        px, py = reliability_pts(y_te, p_te_pl)
        ix, iy = reliability_pts(y_te, p_te_iso)
        ax.plot([0, 1], [0, 1], ":", color="k", lw=1, label="perfect")
        ax.plot(bx, by, "o-", color="#e45756", lw=1.6,
                label=f"raw  (ECE {ece_b:.3f}, AUROC {auroc_b:.3f})")
        ax.plot(px, py, "s-", color="#54a24b", lw=1.6,
                label=f"Platt  (ECE {ece_pl:.3f}, AUROC {auroc_pl:.3f})")
        ax.plot(ix, iy, "^-", color="#4c78a8", lw=1.6,
                label=f"isotonic (ECE {ece_iso:.3f}, AUROC {auroc_iso:.3f})")
        ax.set_title(name)
        ax.set_xlabel("predicted P(hard)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("actual hard frequency")
    fig.suptitle("Probe recalibration via isotonic regression "
                 "(fit on TimeSeriesSplit OOF train predictions)", fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "reliability_recalibrated.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")
    with open(os.path.join(args.out_dir, "recalibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'recalibration_results.json')}")


if __name__ == "__main__":
    main()
