#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", type=Path, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--split_method_name",
        type=str,
        default="nested_balanced_kfold",
    )
    p.add_argument("--manifest_tsv", type=Path, default=None)
    return p.parse_args()


def load_manifest_rows(manifest_tsv: Path) -> list[dict]:
    rows = []
    with manifest_tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") != "ok":
                continue
            row["lesion_mm3_1mm"] = float(row["lesion_mm3_1mm"])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No usable rows found in {manifest_tsv}")
    return rows


def stratify_labels(rows: list[dict], num_bins: int) -> np.ndarray:
    volumes = np.array([row["lesion_mm3_1mm"] for row in rows], dtype=float)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    edges = np.unique(np.quantile(volumes, quantiles))
    volume_bins = np.digitize(volumes, edges, right=False)
    return np.array(
        [f"v{vb}|{row['official_split']}" for vb, row in zip(volume_bins, rows)]
    )


def fold_summary(case_ids: list[str], row_map: dict[str, dict]) -> dict:
    vols = np.array([row_map[cid]["lesion_mm3_1mm"] for cid in case_ids], dtype=float)
    return {
        "n": len(case_ids),
        "volume_mean": float(vols.mean()),
        "volume_median": float(np.median(vols)),
        "volume_min": float(vols.min()),
        "volume_max": float(vols.max()),
        "official_split_counts": {
            key: int(val)
            for key, val in sorted(
                {
                    split: sum(
                        1 for cid in case_ids if row_map[cid]["official_split"] == split
                    )
                    for split in {"train", "test"}
                }.items()
            )
            if val > 0
        },
    }


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir
    manifest_tsv = args.manifest_tsv or (task_dir / "manifest_ds004199_t1flairseg.tsv")
    rows = load_manifest_rows(manifest_tsv)
    row_map = {row["case_id"]: row for row in rows}

    ids = np.array([row["case_id"] for row in rows])
    outer_labels = stratify_labels(rows, num_bins=3)

    outer_splitter = StratifiedKFold(
        n_splits=args.num_folds, shuffle=True, random_state=args.random_state
    )
    nested = []
    summary = {
        "split_method_name": args.split_method_name,
        "num_folds": args.num_folds,
        "random_state": args.random_state,
        "outer_stratification": "lesion_volume_tertiles + official_split",
        "inner_stratification": "lesion_volume_halves + official_split",
        "folds": [],
    }

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splitter.split(ids, outer_labels)
    ):
        outer_train_ids = ids[outer_train_idx]
        outer_test_ids = ids[outer_test_idx]
        outer_train_rows = [row_map[cid] for cid in outer_train_ids]
        inner_labels = stratify_labels(outer_train_rows, num_bins=2)

        inner_splitter = StratifiedKFold(
            n_splits=args.num_folds,
            shuffle=True,
            random_state=args.random_state + 100 + fold_idx,
        )
        inner_splits = list(inner_splitter.split(outer_train_ids, inner_labels))
        inner_val_idx = fold_idx % args.num_folds
        inner_train_idx, inner_val_idx_arr = inner_splits[inner_val_idx]

        train_ids = sorted(outer_train_ids[inner_train_idx].tolist())
        val_ids = sorted(outer_train_ids[inner_val_idx_arr].tolist())
        test_ids = sorted(outer_test_ids.tolist())

        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Train/val overlap in fold {fold_idx}")
        if set(train_ids) & set(test_ids):
            raise RuntimeError(f"Train/test overlap in fold {fold_idx}")
        if set(val_ids) & set(test_ids):
            raise RuntimeError(f"Val/test overlap in fold {fold_idx}")

        nested.append({"train": train_ids, "val": val_ids, "test": test_ids})
        summary["folds"].append(
            {
                "fold_idx": fold_idx,
                "train": fold_summary(train_ids, row_map),
                "val": fold_summary(val_ids, row_map),
                "test": fold_summary(test_ids, row_map),
            }
        )

    splits_path = task_dir / "splits.pkl"
    splits = pickle.load(open(splits_path, "rb"))
    splits.setdefault(args.split_method_name, {})
    splits[args.split_method_name][args.num_folds] = nested
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    out_json = (
        task_dir / f"split_summary_{args.split_method_name}_{args.num_folds}fold.json"
    )
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
