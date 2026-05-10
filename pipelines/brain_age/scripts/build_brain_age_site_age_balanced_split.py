#!/usr/bin/env python3
"""
Generate a reproducible site- and age-balanced finetune/test split for FOMO300K brain age.

This keeps the same 700-subject universe and exact site counts as the current
disk-validated split, but samples the finetune set within each site using
coarse age strata so that the outer 200/500 split is better age-balanced.

It can also update one or more task roots' `splits.pkl` with:
  - the new 160/40 train/val fold
  - a new split key pointing at that predefined fold
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def dataset_from_id(subject_id: str) -> str:
    if subject_id.startswith("GSP_"):
        return "GSP"
    if subject_id.startswith("BrainLat_"):
        return "BrainLat"
    if subject_id.startswith("Task3_"):
        return "Task3"
    raise ValueError(f"Unknown dataset prefix for {subject_id}")


def load_ages(task_dir: Path) -> dict[str, float]:
    ages = {}
    for p in task_dir.glob("*.txt"):
        ages[p.stem] = float(p.read_text().strip())
    return ages


def quantile_strata(
    ids: list[str], ages: dict[str, float], max_bins: int = 5
) -> np.ndarray:
    values = np.array([ages[i] for i in ids], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(
            "All finetune/test candidates must have valid ages for site-age balancing."
        )

    for n_bins in range(max_bins, 1, -1):
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
        strata = np.digitize(values, edges[1:-1], right=True)
        counts = Counter(strata.tolist())
        if min(counts.values()) >= 2:
            return strata

    median = float(np.median(values))
    strata = (values > median).astype(int)
    counts = Counter(strata.tolist())
    if min(counts.values()) < 2:
        raise RuntimeError("Could not derive stable age strata for one site.")
    return strata


def sample_site_age_balanced(
    ids: list[str],
    ages: dict[str, float],
    n_finetune: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    ids = sorted(ids)
    strata = quantile_strata(ids, ages)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=n_finetune, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(np.array(ids), strata))
    finetune_ids = sorted([ids[i] for i in train_idx])
    test_ids = sorted([ids[i] for i in test_idx])
    return finetune_ids, test_ids


def make_train_val(
    ft_ids: list[str], ages: dict[str, float]
) -> tuple[list[str], list[str]]:
    """Sort by age, pick every 5th into val for broad age coverage."""
    with_age = sorted(
        [
            (fid, ages[fid])
            for fid in ft_ids
            if np.isfinite(ages[fid]) and ages[fid] > 0
        ],
        key=lambda x: x[1],
    )
    no_age = [fid for fid in ft_ids if not np.isfinite(ages[fid]) or ages[fid] <= 0]
    val_ids = [with_age[i][0] for i in range(0, len(with_age), 5)][:40]
    val_set = set(val_ids)
    train_ids = [fid for fid, _ in with_age if fid not in val_set] + no_age
    if len(train_ids) != 160 or len(val_ids) != 40:
        raise RuntimeError(
            f"Unexpected train/val sizes: {len(train_ids)} / {len(val_ids)}"
        )
    return sorted(train_ids), sorted(val_ids)


def summarize(ids: list[str], ages: dict[str, float]) -> dict[str, object]:
    vals = np.array([ages[i] for i in ids], dtype=float)
    return {
        "n": len(ids),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "bins_0_20_40_60_80_100": np.histogram(vals, bins=[0, 20, 40, 60, 80, 100])[
            0
        ].tolist(),
        "by_dataset": dict(Counter(dataset_from_id(i) for i in ids)),
    }


def update_splits_pkl(
    task_dir: Path, split_key: str, finetune_ids: list[str], ages: dict[str, float]
) -> None:
    splits_path = task_dir / "splits.pkl"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits.pkl: {splits_path}")

    with splits_path.open("rb") as f:
        splits = pickle.load(f)

    train_ids, val_ids = make_train_val(finetune_ids, ages)
    payload = {"predefined": [{"train": train_ids, "val": val_ids}]}
    splits[split_key] = payload

    if "predefined_ft_split" not in splits:
        splits["predefined_ft_split"] = payload

    with splits_path.open("wb") as f:
        pickle.dump(splits, f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-split-json",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k_unified/splits/"
            "fomo300k_v2_stratified_ft200_test500_gsp179_brainlat179_task3142_diskvalidated.json"
        ),
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-wilms1/FOMO/Data/"
            "fomo300k_unified_match_s23_p160_20260502_exactpretrain_full700_sub100headerfix/"
            "Task_FOMO300K_BrainAge_T1_v2"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k_unified/splits/"
            "fomo300k_v2_stratified_ft200_test500_gsp179_brainlat179_task3142_"
            "diskvalidated_siteagebalanced.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-key",
        type=str,
        default="match_s23_p160_20260502_siteagebalanced",
    )
    parser.add_argument(
        "--update-task-dir",
        action="append",
        default=[],
        help="Task dirs whose splits.pkl should be updated with the new split key.",
    )
    args = parser.parse_args()

    base = json.loads(args.base_split_json.read_text())
    ages = load_ages(args.task_dir)
    universe = sorted(set(base["finetune"]) | set(base["test"]))
    if len(universe) != 700:
        raise RuntimeError(f"Expected 700-subject universe, got {len(universe)}")

    target_ft_counts = Counter(dataset_from_id(i) for i in base["finetune"])
    target_test_counts = Counter(dataset_from_id(i) for i in base["test"])

    grouped = {
        ds: sorted([i for i in universe if dataset_from_id(i) == ds])
        for ds in ("GSP", "BrainLat", "Task3")
    }

    finetune_ids: list[str] = []
    test_ids: list[str] = []
    for ds in ("GSP", "BrainLat", "Task3"):
        ft, tst = sample_site_age_balanced(
            grouped[ds],
            ages,
            n_finetune=target_ft_counts[ds],
            seed=args.seed,
        )
        finetune_ids.extend(ft)
        test_ids.extend(tst)

    finetune_ids = sorted(finetune_ids)
    test_ids = sorted(test_ids)

    if len(finetune_ids) != 200 or len(test_ids) != 500:
        raise RuntimeError(f"Unexpected sizes: {len(finetune_ids)} / {len(test_ids)}")
    if set(finetune_ids) & set(test_ids):
        raise RuntimeError("Overlap detected in finetune/test split")
    if Counter(dataset_from_id(i) for i in finetune_ids) != target_ft_counts:
        raise RuntimeError("Finetune site counts do not match target")
    if Counter(dataset_from_id(i) for i in test_ids) != target_test_counts:
        raise RuntimeError("Test site counts do not match target")

    payload = {
        "finetune": finetune_ids,
        "test": test_ids,
        "source_split": str(args.base_split_json.name),
        "target_test_counts": dict(target_test_counts),
        "notes": "Outer split randomized within each site using site-specific age strata.",
        "meta": {
            "seed": args.seed,
            "task_dir_for_ages": str(args.task_dir),
            "split_key": args.split_key,
            "finetune_summary": summarize(finetune_ids, ages),
            "test_summary": summarize(test_ids, ages),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))

    for td in args.update_task_dir:
        update_splits_pkl(Path(td), args.split_key, finetune_ids, ages)

    print(f"Saved split JSON: {args.output_json}")
    print(
        "Finetune summary:", json.dumps(payload["meta"]["finetune_summary"], indent=2)
    )
    print("Test summary:", json.dumps(payload["meta"]["test_summary"], indent=2))


if __name__ == "__main__":
    main()
