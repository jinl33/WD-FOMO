#!/usr/bin/env python3
"""Evaluate fold-clean threshold tuning and ensembles for Task9 ds004199 FCD.

This uses only held-out fold predictions. For each outer fold, thresholds (and
ensemble weights) are selected using predictions from the other folds, then
applied to the held-out fold. This avoids choosing a threshold directly on the
same fold being reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(
    "/nfs/turbo/umms-wilms1/FOMO/inference_outputs/task9_ds004199_t1flair_cv_20260428"
)
OUT_JSON = ROOT / "nested_threshold_ensemble_eval_20260429.json"
MODELS = ["cleandift_s23", "cleandift_s33", "mmunetvae", "unet_b", "unet_xl"]
THRESHOLDS = np.linspace(0.05, 0.95, 181)
WEIGHTS = np.linspace(0.0, 1.0, 21)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float((y == pred).mean())


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    pos = y == 1
    neg = y == 0
    tpr = float(((pred == 1) & pos).sum() / max(1, pos.sum()))
    tnr = float(((pred == 0) & neg).sum() / max(1, neg.sum()))
    return 0.5 * (tpr + tnr)


def precision(y: np.ndarray, pred: np.ndarray) -> float:
    tp = ((y == 1) & (pred == 1)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    return float(tp / max(1, tp + fp))


def recall(y: np.ndarray, pred: np.ndarray) -> float:
    tp = ((y == 1) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum()
    return float(tp / max(1, tp + fn))


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    pr = precision(y, pred)
    rc = recall(y, pred)
    return 0.0 if pr + rc == 0 else float(2 * pr * rc / (pr + rc))


def eval_scores(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "accuracy": accuracy(y, pred),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "precision": precision(y, pred),
        "recall": recall(y, pred),
        "f1": f1(y, pred),
        "auroc": float(roc_auc_score(y, scores)),
        "threshold": float(threshold),
    }


def summarize(per_fold: list[dict]) -> dict:
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc"]
    return {
        "per_fold": per_fold,
        "mean": {metric: mean(row[metric] for row in per_fold) for metric in metrics},
        "std": {metric: pstdev(row[metric] for row in per_fold) for metric in metrics},
    }


def load_fold_predictions(model_name: str) -> list[list[dict]]:
    rows = []
    for path in sorted(ROOT.glob(f"{model_name}_fold*_test.json")):
        payload = json.loads(path.read_text())
        rows.append(sorted(payload["predictions"], key=lambda row: row["id"]))
    if not rows:
        raise SystemExit(f"No fold prediction files found for {model_name}")
    return rows


def main() -> None:
    fold_rows = {model: load_fold_predictions(model) for model in MODELS}
    out: dict[str, dict] = {"nested_threshold": {}, "nested_ensemble": {}}

    for fold_idx in range(len(fold_rows[MODELS[0]])):
        ids = [row["id"] for row in fold_rows[MODELS[0]][fold_idx]]
        for model in MODELS[1:]:
            other = [row["id"] for row in fold_rows[model][fold_idx]]
            if other != ids:
                raise RuntimeError(f"Subject mismatch in fold {fold_idx} for {model}")

    for model in MODELS:
        per_fold = []
        for fold_idx in range(len(fold_rows[model])):
            train_rows = []
            for other_idx, rows in enumerate(fold_rows[model]):
                if other_idx != fold_idx:
                    train_rows.extend(rows)
            test_rows = fold_rows[model][fold_idx]

            y_train = np.array([row["ground_truth"] for row in train_rows], dtype=int)
            s_train = np.array([row["prob_fcd"] for row in train_rows], dtype=float)
            y_test = np.array([row["ground_truth"] for row in test_rows], dtype=int)
            s_test = np.array([row["prob_fcd"] for row in test_rows], dtype=float)

            best = max(
                (eval_scores(y_train, s_train, thr) for thr in THRESHOLDS),
                key=lambda row: (row["balanced_accuracy"], row["f1"]),
            )
            test_metrics = eval_scores(y_test, s_test, best["threshold"])
            test_metrics["fold"] = fold_idx
            test_metrics["chosen_threshold"] = best["threshold"]
            per_fold.append(test_metrics)
        out["nested_threshold"][model] = summarize(per_fold)

    ens_name = "ours23_33"
    per_fold = []
    s23_rows = fold_rows["cleandift_s23"]
    s33_rows = fold_rows["cleandift_s33"]
    for fold_idx in range(len(s23_rows)):
        tr23, tr33 = [], []
        for other_idx in range(len(s23_rows)):
            if other_idx != fold_idx:
                tr23.extend(s23_rows[other_idx])
                tr33.extend(s33_rows[other_idx])
        te23 = s23_rows[fold_idx]
        te33 = s33_rows[fold_idx]

        if [row["id"] for row in tr23] != [row["id"] for row in tr33]:
            raise RuntimeError(f"Training subject mismatch in fold {fold_idx}")
        if [row["id"] for row in te23] != [row["id"] for row in te33]:
            raise RuntimeError(f"Test subject mismatch in fold {fold_idx}")

        y_train = np.array([row["ground_truth"] for row in tr23], dtype=int)
        y_test = np.array([row["ground_truth"] for row in te23], dtype=int)
        s23_train = np.array([row["prob_fcd"] for row in tr23], dtype=float)
        s33_train = np.array([row["prob_fcd"] for row in tr33], dtype=float)
        s23_test = np.array([row["prob_fcd"] for row in te23], dtype=float)
        s33_test = np.array([row["prob_fcd"] for row in te33], dtype=float)

        best = None
        best_key = None
        for w_s23 in WEIGHTS:
            w_s33 = 1.0 - w_s23
            train_scores = w_s23 * s23_train + w_s33 * s33_train
            for threshold in THRESHOLDS:
                row = eval_scores(y_train, train_scores, threshold)
                key = (row["balanced_accuracy"], row["f1"])
                if best is None or key > best_key:
                    best = {
                        "w_s23": float(w_s23),
                        "w_s33": float(w_s33),
                        "threshold": float(threshold),
                        **row,
                    }
                    best_key = key

        test_scores = best["w_s23"] * s23_test + best["w_s33"] * s33_test
        test_metrics = eval_scores(y_test, test_scores, best["threshold"])
        test_metrics["fold"] = fold_idx
        test_metrics["chosen_threshold"] = best["threshold"]
        test_metrics["w_s23"] = best["w_s23"]
        test_metrics["w_s33"] = best["w_s33"]
        per_fold.append(test_metrics)

    out["nested_ensemble"][ens_name] = summarize(per_fold)

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
