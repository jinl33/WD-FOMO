#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", type=Path, required=True)
    p.add_argument("--num_test_centers", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio_within_trainval", type=float, default=0.1)
    return p.parse_args()

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

def _volume_bins(volumes: np.ndarray):
    for n_bins in (5, 4, 3, 2):
        edges = np.unique(np.quantile(volumes, np.linspace(0, 1, n_bins + 1)[1:-1]))
        if len(edges) == 0:
            continue
        bins = np.digitize(volumes, edges, right=False)
        _, counts = np.unique(bins, return_counts=True)
        if counts.min() >= 2:
            return bins, {
                "stratification": "lesion_volume",
                "n_bins": int(len(np.unique(bins))),
                "edges_mm3": [float(x) for x in edges.tolist()],
            }
    return None, {"stratification": "random_fallback"}

def _extract_center(t1_source: str) -> str:
    match = re.search(r"/Training_Raw/(R\d{3})/", t1_source)
    if not match:
        raise ValueError(f"Could not parse center from {t1_source}")
    return match.group(1)

def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    manifest_path = task_dir / "manifest_atlas2_t1seg.tsv"
    splits_path = task_dir / "splits.pkl"
    summary_path = task_dir / "split_summary_atlas2_t1seg.json"

    rows = list(csv.DictReader(open(manifest_path), delimiter="\t"))
    if not rows:
        raise SystemExit("Manifest has no rows")

    center_to_ids: dict[str, list[str]] = defaultdict(list)
    case_to_volume: dict[str, float] = {}
    case_to_center: dict[str, str] = {}
    for row in rows:
        case_id = row["case_id"]
        center = _extract_center(row["t1_source"])
        center_to_ids[center].append(case_id)
        case_to_center[case_id] = center
        case_to_volume[case_id] = float(row["lesion_mm3_1mm"])

    centers = sorted(center_to_ids)
    if args.num_test_centers >= len(centers):
        raise SystemExit("num_test_centers must be smaller than the number of centers")

    rng = random.Random(args.seed)
    test_centers = sorted(rng.sample(centers, args.num_test_centers))
    trainval_centers = [c for c in centers if c not in test_centers]

    test_ids = sorted(cid for c in test_centers for cid in center_to_ids[c])
    trainval_ids = np.array(sorted(cid for c in trainval_centers for cid in center_to_ids[c]))

    trainval_volumes = np.array([case_to_volume[cid] for cid in trainval_ids], dtype=float)
    strat_labels, strat_info = _volume_bins(trainval_volumes)
    val_size = max(1, int(round(len(trainval_ids) * args.val_ratio_within_trainval)))
    if strat_labels is None:
        splitter = ShuffleSplit(n_splits=1, test_size=val_size, random_state=args.seed + 1)
        train_idx, val_idx = next(splitter.split(trainval_ids))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=args.seed + 1)
        train_idx, val_idx = next(splitter.split(trainval_ids, strat_labels))

    split = {
        "train": sorted(trainval_ids[train_idx].tolist()),
        "val": sorted(trainval_ids[val_idx].tolist()),
        "test": test_ids,
    }

    if set(split["train"]) & set(split["val"]):
        raise SystemExit("Train/val overlap detected")
    if set(split["train"]) & set(split["test"]):
        raise SystemExit("Train/test overlap detected")
    if set(split["val"]) & set(split["test"]):
        raise SystemExit("Val/test overlap detected")

    try:
        splits = pickle.load(open(splits_path, "rb"))
        if not isinstance(splits, dict):
            raise ValueError
    except Exception:
        splits = {}

    method = "paper_center_holdout_split"
    param = str(args.seed)
    splits.setdefault(method, {})
    splits[method][param] = [split]
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    summary["paper_center_holdout_split"] = {
        "seed": args.seed,
        "num_test_centers": args.num_test_centers,
        "val_ratio_within_trainval": args.val_ratio_within_trainval,
        "test_centers": test_centers,
        "trainval_centers": trainval_centers,
        "stratification": strat_info.get("stratification"),
        "n_bins": strat_info.get("n_bins"),
        "edges_mm3": strat_info.get("edges_mm3"),
        "counts": {k: len(v) for k, v in split.items()},
        "train": _summary(split["train"], case_to_volume),
        "val": _summary(split["val"], case_to_volume),
        "test": _summary(split["test"], case_to_volume),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        json.dumps(
            {
                "task_dir": str(task_dir),
                "split_method": method,
                "split_param": param,
                "counts": {k: len(v) for k, v in split.items()},
                "test_centers": test_centers,
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
