#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", type=Path, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--delete_stale_zero_byte_npys", action="store_true")
    return p.parse_args()

def _volume_bins(volumes: np.ndarray, num_folds: int):
    for n_bins in (5, 4, 3, 2):
        edges = np.unique(np.quantile(volumes, np.linspace(0, 1, n_bins + 1)[1:-1]))
        if len(edges) == 0:
            continue
        bins = np.digitize(volumes, edges, right=False)
        _, counts = np.unique(bins, return_counts=True)
        if counts.min() >= num_folds:
            return bins, {
                "stratification": "lesion_volume",
                "n_bins": int(len(np.unique(bins))),
                "edges_mm3": [float(x) for x in edges.tolist()],
            }
    return None, {"stratification": "random_fallback"}

def _stratified_train_val_test_split(ids: np.ndarray, labels, random_state: int) -> dict:
    total = int(len(ids))
    train_target = int(round(total * 0.8))
    val_target = int(round(total * 0.1))
    test_target = max(1, total - train_target - val_target)

    if labels is None:
        first_splitter = ShuffleSplit(n_splits=1, test_size=test_target, random_state=random_state)
        trainval_idx, test_idx = next(first_splitter.split(ids))
        trainval_ids = ids[trainval_idx]
        second_splitter = ShuffleSplit(
            n_splits=1,
            test_size=val_target,
            random_state=random_state + 1,
        )
        train_idx, val_idx = next(second_splitter.split(trainval_ids))
    else:
        first_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_target,
            random_state=random_state,
        )
        trainval_idx, test_idx = next(first_splitter.split(ids, labels))
        trainval_ids = ids[trainval_idx]
        trainval_labels = labels[trainval_idx]
        second_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_target,
            random_state=random_state + 1,
        )
        train_idx, val_idx = next(second_splitter.split(trainval_ids, trainval_labels))

    return {
        "train": sorted(trainval_ids[train_idx].tolist()),
        "val": sorted(trainval_ids[val_idx].tolist()),
        "test": sorted(ids[test_idx].tolist()),
    }

def build_splits(rows, num_folds: int, random_state: int):
    ids = np.array([row["case_id"] for row in rows])
    volumes = np.array([float(row["lesion_mm3_1mm"]) for row in rows], dtype=float)
    labels, info = _volume_bins(volumes, num_folds)

    if labels is None:
        simple_splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        kfold_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        train_idx, val_idx = next(simple_splitter.split(ids))
        kfold_iter = kfold_splitter.split(ids)
    else:
        simple_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        kfold_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        train_idx, val_idx = next(simple_splitter.split(ids, labels))
        kfold_iter = kfold_splitter.split(ids, labels)

    simple = {"train": sorted(ids[train_idx].tolist()), "val": sorted(ids[val_idx].tolist())}
    kfold = [{"train": sorted(ids[tr].tolist()), "val": sorted(ids[va].tolist())} for tr, va in kfold_iter]
    single_run = _stratified_train_val_test_split(ids, labels, random_state)
    return {
        "kfold": {num_folds: kfold},
        "simple_train_val_split": {0.2: [simple]},
        "stratified_train_val_test_split": {0.8: [single_run]},
    }, info

def _summary(case_ids, volume_map):
    vols = [volume_map[cid] for cid in case_ids]
    if not vols:
        return {"n": 0}
    vols = sorted(float(v) for v in vols)
    mid = len(vols) // 2
    median = vols[mid] if len(vols) % 2 else (vols[mid - 1] + vols[mid]) / 2.0
    return {
        "n": len(vols),
        "volume_mean": sum(vols) / len(vols),
        "volume_median": median,
        "volume_min": vols[0],
        "volume_max": vols[-1],
    }

def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    manifest_path = task_dir / "manifest_atlas2_t1seg.tsv"
    splits_path = task_dir / "splits.pkl"
    summary_path = task_dir / "split_summary_atlas2_t1seg.json"

    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    if not splits_path.exists():
        raise SystemExit(f"Missing splits file: {splits_path}")

    with manifest_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit("Manifest has no rows")

    valid_case_ids = {row["case_id"] for row in rows}
    removed = []
    if args.delete_stale_zero_byte_npys:
        for npy_path in sorted(task_dir.glob("*.npy")):
            if npy_path.stat().st_size != 0:
                continue
            if npy_path.stem in valid_case_ids:
                continue
            npy_path.unlink()
            removed.append(npy_path.name)

    refreshed_splits, split_info = build_splits(rows, args.num_folds, args.random_state)
    try:
        with splits_path.open("rb") as f:
            splits = pickle.load(f)
        if not isinstance(splits, dict):
            raise ValueError("splits.pkl does not contain a dict")
    except (EOFError, pickle.UnpicklingError, ValueError):
        splits = {}
    splits["simple_train_val_split"] = refreshed_splits["simple_train_val_split"]
    splits["stratified_train_val_test_split"] = refreshed_splits["stratified_train_val_test_split"]
    with splits_path.open("wb") as f:
        pickle.dump(splits, f)

    volume_map = {row["case_id"]: float(row["lesion_mm3_1mm"]) for row in rows}
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    single_run = splits["stratified_train_val_test_split"][0.8][0]
    summary.update(
        {
            "n_cases": len(rows),
            "stratification": split_info.get("stratification"),
            "n_bins": split_info.get("n_bins"),
            "edges_mm3": split_info.get("edges_mm3"),
            "stratified_train_val_test_split": [
                {
                    "fold": 0,
                    "train": _summary(single_run["train"], volume_map),
                    "val": _summary(single_run["val"], volume_map),
                    "test": _summary(single_run["test"], volume_map),
                }
            ],
            "single_run_split_param": 0.8,
            "stale_zero_byte_npys_removed": removed,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        json.dumps(
            {
                "task_dir": str(task_dir),
                "n_cases": len(rows),
                "single_run_counts": {k: len(v) for k, v in single_run.items()},
                "removed_zero_byte_npys": removed,
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
