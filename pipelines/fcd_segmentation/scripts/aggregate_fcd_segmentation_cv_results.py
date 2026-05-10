#!/usr/bin/env python3
"""Aggregate Task10 per-fold JSON outputs into a CV summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fold_files = sorted(args.input_dir.glob(f"{args.model_name}_fold*_test.json"))
    if not fold_files:
        raise SystemExit(
            f"No fold JSONs found for {args.model_name} in {args.input_dir}"
        )

    folds = []
    all_predictions = []
    for path in fold_files:
        payload = json.loads(path.read_text())
        folds.append(
            {
                "fold_idx": int(payload["fold_idx"]),
                "n_subjects": int(payload["n_subjects"]),
                "mean_dice": float(payload["mean_dice"]),
                "std_dice": float(payload["std_dice"]),
                "median_dice": float(payload["median_dice"]),
                "path": str(path),
            }
        )
        all_predictions.extend(payload.get("predictions", []))

    per_subject_dice = np.array([row["dice"] for row in all_predictions], dtype=float)
    fold_means = np.array([row["mean_dice"] for row in folds], dtype=float)
    summary = {
        "model_name": args.model_name,
        "n_folds": int(len(folds)),
        "n_subjects": int(len(all_predictions)),
        "mean_dice": float(np.mean(per_subject_dice)),
        "std_dice": float(np.std(per_subject_dice)),
        "median_dice": float(np.median(per_subject_dice)),
        "mean_fold_dice": float(np.mean(fold_means)),
        "std_fold_dice": float(np.std(fold_means)),
        "folds": sorted(folds, key=lambda x: x["fold_idx"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
