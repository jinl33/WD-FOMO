#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_dir",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-wilms1/FOMO/Data/fomo-fine-tuning-new/fomo-task2/Task002_FOMO2"
        ),
    )
    parser.add_argument("--source_method", default="kfold")
    parser.add_argument("--source_param", type=int, default=5)
    parser.add_argument("--target_method", default="nested_lesion_kfold")
    parser.add_argument("--inner_val_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument(
        "--stratified_outer",
        action="store_true",
        help="Build the outer test folds directly with lesion-volume stratification instead of nesting inside an existing split.",
    )
    parser.add_argument(
        "--outer_candidate_seeds",
        type=int,
        default=200,
        help="Number of shuffled stratified outer splits to screen for lesion-volume balance.",
    )
    return parser.parse_args()


def case_number(case_id: str) -> int:
    return int(case_id.rsplit("_", 1)[-1])


def lesion_volume(task_dir: Path, case_id: str) -> int:
    arr = np.load(task_dir / f"{case_id}.npy", allow_pickle=True)
    label = np.asarray(arr[-1])
    return int((label > 0).sum())


def quantile_bins(ids: Sequence[str], volumes: dict[str, int], max_bins: int = 3) -> np.ndarray:
    n_bins = min(max_bins, max(1, len(ids) // 2))
    ordered = sorted(ids, key=lambda x: (volumes[x], case_number(x)))
    bins = np.zeros(len(ids), dtype=int)
    lookup = {case_id: i for i, case_id in enumerate(ids)}
    for rank, case_id in enumerate(ordered):
        bins[lookup[case_id]] = min(n_bins - 1, int(rank * n_bins / len(ids)))
    return bins


def choose_inner_split(
    train_ids: Sequence[str],
    volumes: dict[str, int],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    train_ids = list(sorted(train_ids, key=case_number))
    val_n = max(2, int(math.ceil(len(train_ids) * val_fraction)))
    val_n = min(val_n, len(train_ids) - 2)

    y = quantile_bins(train_ids, volumes)
    splitter = StratifiedShuffleSplit(
        n_splits=100,
        test_size=val_n,
        random_state=seed,
    )

    best = None
    all_vols = np.array([volumes[i] for i in train_ids], dtype=float)
    all_log = np.log1p(all_vols)
    for split_train_idx, split_val_idx in splitter.split(np.zeros(len(train_ids)), y):
        inner_train = [train_ids[i] for i in split_train_idx]
        inner_val = [train_ids[i] for i in split_val_idx]
        val_log = np.log1p([volumes[i] for i in inner_val])
        score = (
            -abs(float(val_log.mean()) - float(all_log.mean())),
            -abs(float(np.median(val_log)) - float(np.median(all_log))),
            -abs(len(inner_val) - val_n),
        )
        candidate = (score, inner_train, inner_val)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    return sorted(best[1], key=case_number), sorted(best[2], key=case_number)


def choose_outer_splits(
    case_ids: Sequence[str],
    volumes: dict[str, int],
    n_splits: int,
    seed: int,
    n_candidates: int,
) -> list[tuple[list[str], list[str]]]:
    case_ids = list(sorted(case_ids, key=case_number))
    y = quantile_bins(case_ids, volumes)
    log_vols = np.log1p([volumes[i] for i in case_ids])
    global_mean = float(log_vols.mean())
    global_median = float(np.median(log_vols))

    best = None
    for offset in range(max(1, n_candidates)):
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + offset,
        )
        folds = []
        test_stats = []
        for train_idx, test_idx in splitter.split(np.zeros(len(case_ids)), y):
            train_ids = [case_ids[i] for i in train_idx]
            test_ids = [case_ids[i] for i in test_idx]
            test_log = np.log1p([volumes[i] for i in test_ids])
            test_stats.append(
                (
                    abs(float(test_log.mean()) - global_mean),
                    abs(float(np.median(test_log)) - global_median),
                    len(test_ids),
                )
            )
            folds.append(
                (
                    sorted(train_ids, key=case_number),
                    sorted(test_ids, key=case_number),
                )
            )

        mean_gaps = [s[0] for s in test_stats]
        median_gaps = [s[1] for s in test_stats]
        sizes = [s[2] for s in test_stats]
        score = (
            -max(mean_gaps),
            -float(np.mean(mean_gaps)),
            -max(median_gaps),
            -(max(sizes) - min(sizes)),
        )
        candidate = (score, folds)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    return best[1]


def summarize_ids(ids: Sequence[str], volumes: dict[str, int]) -> dict:
    vals = np.array([volumes[i] for i in ids], dtype=float)
    return {
        "n": len(ids),
        "ids": list(ids),
        "lesion_voxels_min": int(vals.min()) if len(vals) else None,
        "lesion_voxels_median": float(np.median(vals)) if len(vals) else None,
        "lesion_voxels_max": int(vals.max()) if len(vals) else None,
        "lesion_voxels_mean": float(vals.mean()) if len(vals) else None,
    }


def main() -> None:
    args = parse_args()
    splits_path = args.task_dir / "splits.pkl"
    splits = pickle.load(open(splits_path, "rb"))

    case_ids = sorted(
        [p.stem for p in args.task_dir.glob("FOMO2_sub_*.npy")],
        key=case_number,
    )
    volumes = {case_id: lesion_volume(args.task_dir, case_id) for case_id in case_ids}
    if args.stratified_outer:
        outer_splits = choose_outer_splits(
            case_ids,
            volumes,
            n_splits=args.source_param,
            seed=args.seed,
            n_candidates=args.outer_candidate_seeds,
        )
    else:
        if (
            args.source_method not in splits
            or args.source_param not in splits[args.source_method]
        ):
            raise SystemExit(f"Missing {args.source_method}[{args.source_param}] in {splits_path}")
        outer_splits = [
            (list(outer["train"]), list(outer["val"]))
            for outer in splits[args.source_method][args.source_param]
        ]

    nested_folds = []
    summary = {
        "task_dir": str(args.task_dir),
        "source_method": args.source_method,
        "source_param": args.source_param,
        "target_method": args.target_method,
        "inner_val_fraction": args.inner_val_fraction,
        "seed": args.seed,
        "stratified_outer": args.stratified_outer,
        "outer_candidate_seeds": args.outer_candidate_seeds,
        "global": summarize_ids(case_ids, volumes),
        "folds": [],
    }

    for fold_idx, (outer_train, outer_test) in enumerate(outer_splits):
        inner_train, inner_val = choose_inner_split(
            outer_train,
            volumes,
            val_fraction=args.inner_val_fraction,
            seed=args.seed + fold_idx,
        )

        if set(inner_train) & set(inner_val):
            raise SystemExit(f"Train/val overlap in fold {fold_idx}")
        if set(inner_train) & set(outer_test):
            raise SystemExit(f"Train/test overlap in fold {fold_idx}")
        if set(inner_val) & set(outer_test):
            raise SystemExit(f"Val/test overlap in fold {fold_idx}")

        nested_folds.append(
            {
                "train": inner_train,
                "val": inner_val,
                "test": sorted(outer_test, key=case_number),
            }
        )
        summary["folds"].append(
            {
                "fold_idx": fold_idx,
                "train": summarize_ids(inner_train, volumes),
                "val": summarize_ids(inner_val, volumes),
                "test": summarize_ids(sorted(outer_test, key=case_number), volumes),
            }
        )

    splits.setdefault(args.target_method, {})[args.source_param] = nested_folds
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    summary_path = args.task_dir / f"split_summary_{args.target_method}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
