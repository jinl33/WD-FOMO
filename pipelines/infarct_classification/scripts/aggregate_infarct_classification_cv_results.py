#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

def auroc(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--emit_csv", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.input_root.glob(f"{args.model_name}_fold*_test.json"))
    if not paths:
        raise SystemExit(f"No fold JSONs found for {args.model_name} under {args.input_root}")

    fold_rows = [json.loads(p.read_text()) for p in paths]

    fold_aurocs = []
    pooled_labels, pooled_scores = [], []
    csv_rows = []
    for row in fold_rows:
        preds = row["predictions"]
        y = [p["ground_truth"] for p in preds if p["ground_truth"] in (0, 1)]
        s = [p["prob_case"] if "prob_case" in p else p.get("prob_fcd") for p in preds if p["ground_truth"] in (0, 1)]
        fold_aurocs.append(auroc(y, s))
        pooled_labels.extend(y)
        pooled_scores.extend(s)
        for p in preds:
            if p["ground_truth"] not in (0, 1):
                continue
            csv_rows.append(
                {
                    "model": args.model_name,
                    "fold": row["fold_idx"],
                    "subject_id": p["id"],
                    "label": p["ground_truth"],
                    "score": p.get("prob_case", p.get("prob_fcd")),
                }
            )

    finite_aurocs = [a for a in fold_aurocs if not np.isnan(a)]
    if len(finite_aurocs) < len(fold_aurocs):
        print(
            f"WARNING: {len(fold_aurocs) - len(finite_aurocs)} of {len(fold_aurocs)} folds had a "
            "single-class validation set (AUROC undefined). Excluded from the fold mean/SD below."
        )

    summary = {
        "model": args.model_name,
        "n_folds": len(fold_rows),
        "fold_files": [str(p) for p in paths],
        "fold_auroc": fold_aurocs,
        "auroc_fold_mean": float(np.mean(finite_aurocs)) if finite_aurocs else float("nan"),
        "auroc_fold_sd": float(np.std(finite_aurocs, ddof=1)) if len(finite_aurocs) > 1 else 0.0,
        "auroc_pooled": auroc(pooled_labels, pooled_scores),
        "n_subjects_pooled": len(pooled_labels),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {args.output_json}")
    print(
        f"{args.model_name}: fold-mean AUROC {summary['auroc_fold_mean']:.4f} +/- "
        f"{summary['auroc_fold_sd']:.4f}  |  pooled AUROC {summary['auroc_pooled']:.4f}"
    )

    if args.emit_csv:
        args.emit_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.emit_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "fold", "subject_id", "label", "score"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved: {args.emit_csv}")

if __name__ == "__main__":
    main()
