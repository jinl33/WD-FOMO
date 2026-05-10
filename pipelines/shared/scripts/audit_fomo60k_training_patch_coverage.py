#!/usr/bin/env python3
"""Audit whether FOMO60k arrays fit inside the fixed training patch.

This script is intentionally header-first: it reads .npy headers for the full
count and only loads voxel data when explicitly asked for a foreground check.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npfmt


DEFAULT_DATA_DIR = Path("/nfs/turbo/umms-wilms1/FOMO/Data/preprocessed/FOMO60k")
DEFAULT_SPLIT_JSON = Path("/nfs/turbo/umms-wilms1/FOMO/fomo60k_split.json")
DEFAULT_TARGET = (192, 256, 192)
KNOWN_CASE = "sub_1106_ses_1_t1_4_skull_stripped.npy"


def read_npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as f:
        version = npfmt.read_magic(f)
        if version == (1, 0):
            shape, _, _ = npfmt.read_array_header_1_0(f)
        else:
            shape, _, _ = npfmt.read_array_header_2_0(f)
    return tuple(int(v) for v in shape)


def spatial_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) < 3:
        raise ValueError(f"Expected at least 3 dims, got {shape}")
    return tuple(int(v) for v in shape[-3:])


def crop_pairs(crop: Any) -> list[tuple[int, int]] | None:
    if crop is None:
        return None
    if len(crop) == 6:
        return [
            (int(crop[0]), int(crop[1])),
            (int(crop[2]), int(crop[3])),
            (int(crop[4]), int(crop[5])),
        ]
    if len(crop) == 3 and all(
        hasattr(item, "__len__") and len(item) == 2 for item in crop
    ):
        return [(int(lo), int(hi)) for lo, hi in crop]
    return None


def crop_touches_original(meta: dict[str, Any]) -> list[str]:
    pairs = crop_pairs(meta.get("crop_to_nonzero"))
    original_size = meta.get("original_size")
    if pairs is None or original_size is None:
        return []

    touches: list[str] = []
    for axis, ((lo, hi), size) in enumerate(zip(pairs, original_size)):
        if lo <= 0:
            touches.append(f"axis{axis}_low")
        if hi >= int(size):
            touches.append(f"axis{axis}_high")
    return touches


def center_slices(
    shape: tuple[int, int, int], target: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    slices: list[slice] = []
    for size, wanted in zip(shape, target):
        if size > wanted:
            start = (size - wanted) // 2
            slices.append(slice(start, start + wanted))
        else:
            slices.append(slice(0, size))
    return tuple(slices)  # type: ignore[return-value]


def foreground_loss(path: Path, target: tuple[int, int, int]) -> dict[str, Any]:
    arr = np.load(path)
    if arr.ndim > 3:
        arr = arr.reshape((-1,) + arr.shape[-3:])[0]
    mask = np.abs(arr) > 1e-6
    total = int(mask.sum())
    kept = int(mask[center_slices(spatial_shape(mask.shape), target)].sum())
    boundary_fractions: dict[str, float] = {}
    for axis in range(3):
        for name, index in (("low", 0), ("high", -1)):
            slab = np.take(mask, index, axis=axis)
            boundary_fractions[f"axis{axis}_{name}"] = float(slab.sum()) / float(
                slab.size
            )
    return {
        "foreground_voxels": total,
        "foreground_lost_by_center_patch": total - kept,
        "boundary_nonzero_fraction": boundary_fractions,
    }


def load_split_files(split_json: Path, subset: str) -> list[str]:
    with split_json.open() as f:
        split = json.load(f)
    if subset == "all":
        files = list(split.get("pretrain", [])) + list(split.get("distillation", []))
    else:
        files = list(split[subset])
    return files


def resolve_files(data_dir: Path, split_json: Path | None, subset: str) -> list[Path]:
    if split_json is None:
        return sorted(data_dir.glob("*.npy"))
    return [data_dir / Path(item).name for item in load_split_files(split_json, subset)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument(
        "--subset", choices=["pretrain", "distillation", "all"], default="pretrain"
    )
    parser.add_argument("--target-size", nargs=3, type=int, default=DEFAULT_TARGET)
    parser.add_argument(
        "--report", type=Path, default=Path("tmp_fomo60k_patch_coverage_report.json")
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="If >0, sample this many split entries.",
    )
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument(
        "--foreground-check-count",
        type=int,
        default=1,
        help="Number of files to load for foreground-loss checks; known case is prioritized.",
    )
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = tuple(int(v) for v in args.target_size)
    start = time.time()

    files = resolve_files(
        args.data_dir,
        args.split_json if args.split_json.exists() else None,
        args.subset,
    )
    total_entries = len(files)
    if args.sample_count > 0 and args.sample_count < len(files):
        rng = random.Random(args.seed)
        known = args.data_dir / KNOWN_CASE
        sampled = rng.sample(files, args.sample_count)
        if known in files and known not in sampled:
            sampled[0] = known
        files = sampled

    counts = {
        "entries": total_entries,
        "checked": 0,
        "within_patch": 0,
        "over_patch": 0,
        "missing_npy": 0,
        "bad_npy": 0,
        "missing_pkl": 0,
        "bad_pkl": 0,
        "crop_touches_original": 0,
    }
    over_by_axis = [0, 0, 0]
    touch_by_axis = {
        f"axis{axis}_{side}": 0 for axis in range(3) for side in ("low", "high")
    }
    min_shape = [10**9, 10**9, 10**9]
    max_shape = [0, 0, 0]
    over_examples: list[dict[str, Any]] = []
    touch_examples: list[dict[str, Any]] = []
    missing_examples: list[str] = []
    bad_examples: list[dict[str, str]] = []
    known_case: dict[str, Any] | None = None

    foreground_candidates: list[Path] = []
    known_path = args.data_dir / KNOWN_CASE
    if args.foreground_check_count > 0 and known_path in files:
        foreground_candidates.append(known_path)

    for idx, path in enumerate(files, 1):
        if not path.exists():
            counts["missing_npy"] += 1
            if len(missing_examples) < 10:
                missing_examples.append(str(path))
            continue

        try:
            shape = spatial_shape(read_npy_shape(path))
        except Exception as exc:  # noqa: BLE001 - report and continue auditing
            counts["bad_npy"] += 1
            if len(bad_examples) < 10:
                bad_examples.append({"path": str(path), "error": repr(exc)})
            continue

        counts["checked"] += 1
        for axis, size in enumerate(shape):
            min_shape[axis] = min(min_shape[axis], size)
            max_shape[axis] = max(max_shape[axis], size)

        over_axes = [
            axis
            for axis, (size, wanted) in enumerate(zip(shape, target))
            if size > wanted
        ]
        if over_axes:
            counts["over_patch"] += 1
            for axis in over_axes:
                over_by_axis[axis] += 1
            if len(over_examples) < 20:
                over_examples.append(
                    {"file": path.name, "shape": shape, "over_axes": over_axes}
                )
        else:
            counts["within_patch"] += 1

        touches: list[str] = []
        pkl_path = path.with_suffix(".pkl")
        if not pkl_path.exists():
            counts["missing_pkl"] += 1
        else:
            try:
                with pkl_path.open("rb") as f:
                    meta = pickle.load(f)
                touches = crop_touches_original(meta)
            except Exception:  # noqa: BLE001
                counts["bad_pkl"] += 1
                touches = []

        if touches:
            counts["crop_touches_original"] += 1
            for touch in touches:
                touch_by_axis[touch] += 1
            if len(touch_examples) < 20:
                touch_examples.append(
                    {"file": path.name, "shape": shape, "touches": touches}
                )

        if path.name == KNOWN_CASE:
            known_case = {
                "file": path.name,
                "shape": shape,
                "over_axes": over_axes,
                "touches": touches,
            }

        if (
            args.foreground_check_count > 0
            and len(foreground_candidates) < args.foreground_check_count
        ):
            if path not in foreground_candidates and (touches or over_axes):
                foreground_candidates.append(path)

        if args.progress_every > 0 and idx % args.progress_every == 0:
            elapsed = time.time() - start
            print(
                f"progress {idx}/{len(files)} checked={counts['checked']} "
                f"within={counts['within_patch']} over={counts['over_patch']} "
                f"touch={counts['crop_touches_original']} elapsed={elapsed:.1f}s",
                flush=True,
            )

    foreground_checks = []
    for path in foreground_candidates[: args.foreground_check_count]:
        if path.exists():
            item = {"file": path.name}
            item.update(foreground_loss(path, target))
            foreground_checks.append(item)

    report = {
        "data_dir": str(args.data_dir),
        "split_json": str(args.split_json) if args.split_json else None,
        "subset": args.subset,
        "target_size": target,
        "sample_count": args.sample_count,
        "counts": counts,
        "within_patch_fraction_checked": (
            counts["within_patch"] / counts["checked"] if counts["checked"] else 0.0
        ),
        "over_by_axis": over_by_axis,
        "touch_by_axis": touch_by_axis,
        "min_shape": None if counts["checked"] == 0 else tuple(min_shape),
        "max_shape": None if counts["checked"] == 0 else tuple(max_shape),
        "known_case": known_case,
        "foreground_checks": foreground_checks,
        "over_examples": over_examples,
        "touch_examples": touch_examples,
        "missing_examples": missing_examples,
        "bad_examples": bad_examples,
        "elapsed_seconds": time.time() - start,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as f:
        json.dump(report, f, indent=2)

    print("\nRESULT")
    print(f"entries={counts['entries']}")
    print(f"checked={counts['checked']}")
    print(f"within_patch={counts['within_patch']}")
    print(f"over_patch={counts['over_patch']}")
    print(f"crop_touches_original={counts['crop_touches_original']}")
    print(
        f"within_patch_pct_checked={report['within_patch_fraction_checked'] * 100:.4f}"
    )
    print(f"min_shape={report['min_shape']}")
    print(f"max_shape={report['max_shape']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
