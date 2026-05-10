#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", type=Path, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--inner_val_fraction", type=float, default=0.2)
    p.add_argument("--manifest_tsv", type=Path, default=None)
    p.add_argument("--split_method_name", type=str, default="nested_kfold")
    return p.parse_args()


def load_manifest_volumes(manifest_tsv: Path) -> dict[str, float]:
    rows = list(csv.DictReader(manifest_tsv.open(), delimiter="\t"))
    out: dict[str, float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        out[row["case_id"]] = float(row["lesion_mm3_1mm"])
    if not out:
        raise RuntimeError(f"No usable lesion volumes found in {manifest_tsv}")
    return out


def volume_bins(volumes: np.ndarray, num_folds: int) -> tuple[np.ndarray | None, dict]:
    for n_bins in (4, 3, 2):
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(volumes, quantiles))
        if len(edges) == 0:
            continue
        bins = np.digitize(volumes, edges, right=False)
        _, counts = np.unique(bins, return_counts=True)
        if counts.min() >= num_folds:
            return bins, {
                "method": "volume_stratified",
                "n_bins": int(len(np.unique(bins))),
                "edges": [float(x) for x in edges.tolist()],
                "counts": {
                    int(k): int(v) for k, v in zip(*np.unique(bins, return_counts=True))
                },
            }
    return None, {"method": "random_fallback"}


def inner_split(
    case_ids: list[str],
    volume_map: dict[str, float],
    val_fraction: float,
    random_state: int,
) -> tuple[list[str], list[str], dict]:
    ids = np.array(case_ids)
    vols = np.array([volume_map[i] for i in ids], dtype=float)
    labels, meta = volume_bins(vols, max(2, int(round(1.0 / val_fraction))))
    if labels is not None:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=random_state
        )
        train_idx, val_idx = next(splitter.split(ids, labels))
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=random_state
        )
        train_idx, val_idx = next(splitter.split(ids))
    return ids[train_idx].tolist(), ids[val_idx].tolist(), meta


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir
    splits_path = task_dir / "splits.pkl"
    if not splits_path.exists():
        raise FileNotFoundError(splits_path)

    manifest_tsv = args.manifest_tsv or (task_dir / "manifest_ds004199_t1flairseg.tsv")
    volume_map = load_manifest_volumes(manifest_tsv)

    splits = pickle.load(open(splits_path, "rb"))
    if "kfold" not in splits or args.num_folds not in splits["kfold"]:
        raise RuntimeError(f"Expected outer kfold[{args.num_folds}] in {splits_path}")

    outer_folds = splits["kfold"][args.num_folds]
    nested_folds = []
    inner_meta = []
    for fold_idx, fold in enumerate(outer_folds):
        outer_train = sorted(fold["train"])
        outer_test = sorted(fold["val"])
        train_ids, val_ids, meta = inner_split(
            outer_train,
            volume_map=volume_map,
            val_fraction=args.inner_val_fraction,
            random_state=args.random_state + fold_idx,
        )
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Inner train/val overlap in fold {fold_idx}")
        if set(train_ids) & set(outer_test):
            raise RuntimeError(f"Train/test overlap in fold {fold_idx}")
        if set(val_ids) & set(outer_test):
            raise RuntimeError(f"Val/test overlap in fold {fold_idx}")

        nested_folds.append(
            {
                "train": sorted(train_ids),
                "val": sorted(val_ids),
                "test": outer_test,
            }
        )
        inner_meta.append(
            {
                "fold_idx": fold_idx,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "n_test": len(outer_test),
                "volume_split": meta,
            }
        )

    splits.setdefault(args.split_method_name, {})
    splits[args.split_method_name][args.num_folds] = nested_folds
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    summary = {
        "split_method_name": args.split_method_name,
        "num_folds": args.num_folds,
        "inner_val_fraction": args.inner_val_fraction,
        "random_state": args.random_state,
        "folds": inner_meta,
    }
    out_json = (
        task_dir / f"split_summary_{args.split_method_name}_{args.num_folds}fold.json"
    )
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
