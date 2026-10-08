#!/usr/bin/env python3
"""Build a BrainLat T1 + SynthSeg anatomical segmentation warm-up task.

This creates a non-FOMO60K-overlap dense supervision task from BrainLat T1
volumes and their SynthSeg pseudo-labels. Labels are remapped to contiguous
class IDs so the downstream segmentation head can be trained stably.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from nibabel.processing import resample_from_to
from sklearn.model_selection import KFold, ShuffleSplit
from yucca.functional.preprocessing import preprocess_case_for_training_with_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np


REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_SRC = REPO_ROOT / "src" / "downstream"
if str(DOWNSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_SRC))

from data.preprocessing_defaults import build_pretrain_style_preprocess_config


TASK_NAME = "Task011_BrainLat_SynthSeg_T1_1mm"
PREFIX = "brainlatsseg"
PREPROCESS_CONFIG = build_pretrain_style_preprocess_config(num_modalities=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=Path, required=True)
    p.add_argument("--label_dir", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--task_name", type=str, default=TASK_NAME)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def label_key(seg_path: Path) -> str:
    return seg_path.name.replace("_synthseg.nii.gz", "")


def image_key(img_path: Path) -> str:
    return img_path.name.replace("_skull_stripped.nii.gz", "")


def collect_cases(image_dir: Path, label_dir: Path) -> List[Tuple[str, Path, Path]]:
    images = {image_key(p): p for p in sorted(image_dir.glob("*_skull_stripped.nii.gz"))}
    labels = {label_key(p): p for p in sorted(label_dir.glob("*_synthseg.nii.gz"))}
    shared = sorted(set(images) & set(labels))
    return [(key, images[key], labels[key]) for key in shared]


def collect_global_labels(cases: List[Tuple[str, Path, Path]]) -> Dict[int, int]:
    labels: set[int] = set()
    for _, _, seg_path in cases:
        arr = np.asarray(nib.load(str(seg_path)).dataobj)
        labels.update(int(x) for x in np.unique(arr).tolist())
    ordered = sorted(labels)
    return {old: new for new, old in enumerate(ordered)}


def remap_labels(arr: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    for old, new in mapping.items():
        out[arr == old] = new
    return out


def process_case(
    task_dir_str: str,
    case_id: str,
    img_path_str: str,
    seg_path_str: str,
    label_mapping: Dict[int, int],
    overwrite: bool,
) -> dict:
    task_dir = Path(task_dir_str)
    npy_out = task_dir / f"{case_id}.npy"
    pkl_out = task_dir / f"{case_id}.pkl"
    if npy_out.exists() and pkl_out.exists() and not overwrite:
        return {"case_id": case_id, "status": "exists"}

    img_path = Path(img_path_str)
    seg_path = Path(seg_path_str)
    img = nib.load(str(img_path))
    seg = nib.load(str(seg_path))
    seg_in_img = resample_from_to(seg, img, order=0)
    seg_arr = np.asarray(seg_in_img.dataobj)
    seg_arr = remap_labels(seg_arr, label_mapping).astype(np.uint8)
    seg_in_img = nib.Nifti1Image(seg_arr, seg_in_img.affine, seg_in_img.header)

    preprocessed_data, preprocessed_label, props = preprocess_case_for_training_with_label(
        images=[read_file_to_nifti_or_np(str(img_path))],
        label=seg_in_img,
        **PREPROCESS_CONFIG,
    )

    stacked = list(preprocessed_data) + [preprocessed_label]
    np.save(npy_out, np.array(stacked, dtype=object))
    save_pickle(props, str(pkl_out))

    uniq = np.unique(preprocessed_label)
    return {
        "case_id": case_id,
        "status": "ok",
        "image_source": str(img_path),
        "label_source": str(seg_path),
        "present_classes": [int(x) for x in uniq.tolist()],
        "n_present_classes": int(len(uniq)),
        "shape": list(np.asarray(stacked, dtype=object).shape),
        "original_size": list(img.shape),
        "aligned_label_size": list(seg_in_img.shape),
    }


def build_splits(case_ids: List[str], num_folds: int, random_state: int) -> dict:
    ids = np.array(case_ids)
    kfold = []
    splitter = KFold(n_splits=min(num_folds, len(ids)), shuffle=True, random_state=random_state)
    for train_idx, val_idx in splitter.split(ids):
        kfold.append({"train": ids[train_idx].tolist(), "val": ids[val_idx].tolist()})

    ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(ss.split(ids))
    simple = {"train": ids[train_idx].tolist(), "val": ids[val_idx].tolist()}
    return {
        "kfold": {num_folds: kfold},
        "simple_train_val_split": {0.2: [simple]},
    }


def main() -> None:
    args = parse_args()
    task_dir = args.output_root.resolve() / args.task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(args.image_dir.resolve(), args.label_dir.resolve())
    if not cases:
        raise SystemExit("No matching BrainLat T1 / SynthSeg pairs found")

    label_mapping = collect_global_labels(cases)
    rows: List[dict] = []
    failures: List[dict] = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {
            ex.submit(
                process_case,
                str(task_dir),
                f"{PREFIX}_{case_key}",
                str(img_path),
                str(seg_path),
                label_mapping,
                args.overwrite,
            ): (case_key, img_path, seg_path)
            for case_key, img_path, seg_path in cases
        }
        for idx, fut in enumerate(as_completed(futs), 1):
            case_key, img_path, seg_path = futs[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"[{idx}/{len(cases)}] OK   {case_key}", flush=True)
            except Exception as exc:
                failures.append(
                    {
                        "case_key": case_key,
                        "image_source": str(img_path),
                        "label_source": str(seg_path),
                        "reason": str(exc),
                    }
                )
                print(f"[{idx}/{len(cases)}] FAIL {case_key}: {exc}", flush=True)

    ok_rows = [r for r in rows if r.get("status") in {"ok", "exists"}]
    if not ok_rows:
        raise SystemExit("No BrainLat SynthSeg cases were prepared")

    case_ids = sorted(row["case_id"] for row in ok_rows)
    splits = build_splits(case_ids, args.num_folds, args.random_state)
    save_pickle(splits, str(task_dir / "splits.pkl"))

    manifest_path = task_dir / "manifest_brainlat_synthseg.tsv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = [
            "case_id",
            "status",
            "image_source",
            "label_source",
            "n_present_classes",
            "present_classes",
            "shape",
            "original_size",
            "aligned_label_size",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in sorted(ok_rows, key=lambda x: x["case_id"]):
            out = row.copy()
            out["present_classes"] = json.dumps(out.get("present_classes", []))
            out["shape"] = json.dumps(out.get("shape", []))
            out["original_size"] = json.dumps(out.get("original_size", []))
            out["aligned_label_size"] = json.dumps(out.get("aligned_label_size", []))
            writer.writerow(out)

    with (task_dir / "label_mapping.json").open("w") as f:
        json.dump(
            {
                "task_name": args.task_name,
                "num_classes": len(label_mapping),
                "old_to_new": {str(k): int(v) for k, v in label_mapping.items()},
            },
            f,
            indent=2,
        )
    with (task_dir / "failed_cases_brainlat_synthseg.json").open("w") as f:
        json.dump(failures, f, indent=2)

    summary = {
        "n_cases": len(ok_rows),
        "n_failures": len(failures),
        "num_classes": len(label_mapping),
        "task_dir": str(task_dir),
        "manifest_tsv": str(manifest_path),
    }
    with (task_dir / "split_summary_brainlat_synthseg.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
