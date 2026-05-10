#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "auroc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", type=Path, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.glob(f"{args.model_name}_fold*_test.json"))
    if not paths:
        raise SystemExit(
            f"No fold JSONs found for {args.model_name} under {args.input_root}"
        )

    fold_rows = [json.loads(path.read_text()) for path in paths]
    summary = {
        "model": args.model_name,
        "n_folds": len(fold_rows),
        "fold_files": [str(path) for path in paths],
        "per_fold": [],
        "metrics": {},
    }

    for row in fold_rows:
        summary["per_fold"].append(
            {
                "fold_idx": row.get("fold_idx"),
                **{metric: row.get(metric) for metric in METRICS},
                "n_subjects": row.get("n_subjects"),
            }
        )

    for metric in METRICS:
        values = [
            float(row[metric]) for row in fold_rows if row.get(metric) is not None
        ]
        summary["metrics"][metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
