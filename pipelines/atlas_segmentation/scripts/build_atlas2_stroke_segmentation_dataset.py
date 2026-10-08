#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from nibabel.processing import resample_from_to
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from yucca.functional.preprocessing import preprocess_case_for_training_with_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np

REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_SRC = REPO_ROOT / "src" / "downstream"
if str(DOWNSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_SRC))

from data.preprocessing_defaults import build_pretrain_style_preprocess_config

TASK_NAME = "Task012_ATLAS2_StrokeLesion_T1_1mm"
PREFIX = "atlas2"
PREPROCESS_CONFIG = build_pretrain_style_preprocess_config(num_modalities=1)

@dataclass(frozen=True)
class AtlasCase:
    case_id: str
    t1_path: str
    label_path: str
    raw_lesion_mm3: float

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--task_name", type=str, default=TASK_NAME)
    p.add_argument("--subset_size", type=int, default=0)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--min_lesion_mm3", type=float, default=0.0)
    p.add_argument("--max_lesion_mm3", type=float, default=0.0)
    p.add_argument("--include_mni", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem

def _safe_id(text: str) -> str:
    text = re.sub(r"_T1w$", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return f"{PREFIX}_{text}"

def _candidate_label_paths(t1_path: Path) -> List[Path]:
    stem = _strip_nii_suffix(t1_path)
    base = stem
    for suffix in ("_desc-brain_T1w", "_T1w"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    candidates = []
    for label_suffix in (
        "_label-lesion_desc-T1lesion_mask",
        "_label-L_desc-T1lesion_mask",
    ):
        candidates.extend(
            [
                t1_path.with_name(base + label_suffix + ".nii.gz"),
                t1_path.with_name(base + label_suffix + ".nii"),
            ]
        )
    candidates.extend(sorted(t1_path.parent.glob(f"{base}*label*desc-T1lesion_mask.nii*")))

    deduped = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped

def lesion_mm3(label_path: Path) -> float:
    label = nib.load(str(label_path))
    data = np.asarray(label.dataobj)
    voxels = int(np.sum(data > 0))
    voxel_volume = float(np.prod(label.header.get_zooms()[:3]))
    return float(voxels * voxel_volume)

def discover_cases(
    source_dir: Path,
    include_mni: bool,
    min_lesion_mm3: float,
    max_lesion_mm3: float,
) -> List[AtlasCase]:
    cases: List[AtlasCase] = []
    seen = set()
    for t1_path in sorted(source_dir.rglob("*_T1w.nii*")):
        name = t1_path.name
        if "label-L_desc-T1lesion_mask" in name:
            continue
        if not include_mni and "MNI152" in str(t1_path):
            continue
        if "Testing" in t1_path.parts:
            continue
        label_path = next((p for p in _candidate_label_paths(t1_path) if p.exists()), None)
        if label_path is None:
            continue
        vol = lesion_mm3(label_path)
        if vol <= 0:
            continue
        if vol < min_lesion_mm3:
            continue
        if max_lesion_mm3 > 0 and vol > max_lesion_mm3:
            continue
        case_id = _safe_id(_strip_nii_suffix(t1_path))
        if case_id in seen:
            rel = "_".join(t1_path.relative_to(source_dir).parts)
            case_id = _safe_id(_strip_nii_suffix(Path(rel)))
        seen.add(case_id)
        cases.append(
            AtlasCase(
                case_id=case_id,
                t1_path=str(t1_path),
                label_path=str(label_path),
                raw_lesion_mm3=vol,
            )
        )
    return cases

def select_subset(
    cases: List[AtlasCase], subset_size: int, random_state: int
) -> Tuple[List[AtlasCase], dict]:
    if subset_size <= 0 or subset_size >= len(cases):
        return cases, {"subset_strategy": "all_available"}

    rng = random.Random(random_state)
    volumes = np.array([c.raw_lesion_mm3 for c in cases], dtype=float)
    edges = np.unique(np.quantile(volumes, [0.2, 0.4, 0.6, 0.8]))
    bins = np.digitize(volumes, edges, right=False)
    selected: List[AtlasCase] = []
    per_bin = max(1, subset_size // len(np.unique(bins)))
    leftovers: List[AtlasCase] = []
    for bin_id in sorted(np.unique(bins)):
        bucket = [case for case, b in zip(cases, bins) if b == bin_id]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_bin])
        leftovers.extend(bucket[per_bin:])
    if len(selected) < subset_size:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: subset_size - len(selected)])
    selected = selected[:subset_size]
    selected_ids = {case.case_id for case in selected}
    return (
        [case for case in cases if case.case_id in selected_ids],
        {
            "subset_strategy": "lesion_volume_quintile_stratified",
            "subset_size": int(subset_size),
            "volume_edges_mm3": [float(x) for x in edges.tolist()],
        },
    )

def _props_subset(props: dict) -> dict:
    keep = {}
    for key in ("original_spacing", "new_spacing", "crop_to_nonzero", "original_size", "new_size"):
        if key not in props:
            continue
        value = props[key]
        if isinstance(value, np.ndarray):
            keep[key] = value.tolist()
        elif isinstance(value, tuple):
            keep[key] = list(value)
        else:
            keep[key] = value
    return keep

def _save_nifti(img: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))

def _align_label_to_t1(
    t1_img: nib.Nifti1Image, label_img: nib.Nifti1Image
) -> Tuple[nib.Nifti1Image, bool]:
    same_shape = t1_img.shape[:3] == label_img.shape[:3]
    same_affine = np.allclose(t1_img.affine, label_img.affine, atol=1e-4)
    if same_shape and same_affine:
        data = (np.asarray(label_img.dataobj) > 0).astype(np.uint8)
        return nib.Nifti1Image(data, label_img.affine, label_img.header), False
    aligned = resample_from_to(label_img, t1_img, order=0)
    data = (np.asarray(aligned.dataobj) > 0.5).astype(np.uint8)
    return nib.Nifti1Image(data, aligned.affine, aligned.header), True

def process_case(task_dir_str: str, tmp_dir_str: str, overwrite: bool, case: AtlasCase) -> dict:
    task_dir = Path(task_dir_str)
    tmp_dir = Path(tmp_dir_str)
    npy_out = task_dir / f"{case.case_id}.npy"
    pkl_out = task_dir / f"{case.case_id}.pkl"
    if npy_out.exists() and pkl_out.exists() and not overwrite:
        return {
            "case_id": case.case_id,
            "status": "exists",
            "t1_source": case.t1_path,
            "label_source": case.label_path,
            "raw_lesion_mm3": case.raw_lesion_mm3,
        }

    t1_img = nib.load(case.t1_path)
    label_img = nib.load(case.label_path)
    label_aligned, label_resampled = _align_label_to_t1(t1_img, label_img)

    aligned_dir = tmp_dir / "aligned"
    t1_tmp = aligned_dir / f"{case.case_id}_T1w.nii.gz"
    label_tmp = aligned_dir / f"{case.case_id}_label.nii.gz"
    _save_nifti(t1_img, t1_tmp)
    _save_nifti(label_aligned, label_tmp)

    data, label, props = preprocess_case_for_training_with_label(
        images=[read_file_to_nifti_or_np(str(t1_tmp))],
        label=read_file_to_nifti_or_np(str(label_tmp)),
        **PREPROCESS_CONFIG,
    )
    data_with_label = list(data) + [label]
    np.save(npy_out, np.array(data_with_label, dtype=object))
    save_pickle(props, str(pkl_out))

    lesion_voxels = int(np.sum(np.asarray(label) > 0))
    return {
        "case_id": case.case_id,
        "status": "ok",
        "t1_source": case.t1_path,
        "label_source": case.label_path,
        "t1_used": str(t1_tmp),
        "label_used": str(label_tmp),
        "label_resampled_to_t1": bool(label_resampled),
        "raw_lesion_mm3": case.raw_lesion_mm3,
        "lesion_voxels_1mm": lesion_voxels,
        "lesion_mm3_1mm": float(lesion_voxels),
        **_props_subset(props),
    }

def _volume_bins(volumes: np.ndarray, num_folds: int) -> Tuple[Optional[np.ndarray], dict]:
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

def _summary(case_ids: List[str], volume_map: Dict[str, float]) -> dict:
    vols = np.array([volume_map[cid] for cid in case_ids], dtype=float)
    if len(vols) == 0:
        return {"n": 0}
    return {
        "n": int(len(case_ids)),
        "volume_mean": float(vols.mean()),
        "volume_median": float(np.median(vols)),
        "volume_min": float(vols.min()),
        "volume_max": float(vols.max()),
    }

def _stratified_train_val_test_split(
    ids: np.ndarray,
    labels: Optional[np.ndarray],
    random_state: int,
) -> dict:
    total = int(len(ids))
    train_target = int(round(total * 0.8))
    val_target = int(round(total * 0.1))
    test_target = max(1, total - train_target - val_target)

    if labels is None:
        first_splitter = ShuffleSplit(n_splits=1, test_size=test_target, random_state=random_state)
        trainval_idx, test_idx = next(first_splitter.split(ids))
        trainval_ids = ids[trainval_idx]
        test_ids = ids[test_idx]

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
        test_ids = ids[test_idx]

        second_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_target,
            random_state=random_state + 1,
        )
        train_idx, val_idx = next(second_splitter.split(trainval_ids, trainval_labels))

    return {
        "train": sorted(trainval_ids[train_idx].tolist()),
        "val": sorted(trainval_ids[val_idx].tolist()),
        "test": sorted(test_ids.tolist()),
    }

def build_splits(rows: List[dict], num_folds: int, random_state: int) -> Tuple[dict, dict]:
    ids = np.array([row["case_id"] for row in rows])
    volumes = np.array([row["lesion_mm3_1mm"] for row in rows], dtype=float)
    labels, info = _volume_bins(volumes, num_folds)

    def make_splitter():
        if labels is None:
            return KFold(n_splits=num_folds, shuffle=True, random_state=random_state), None
        return StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state), labels

    splitter, split_labels = make_splitter()
    kfold = []
    split_iter = splitter.split(ids, split_labels) if split_labels is not None else splitter.split(ids)
    for train_idx, val_idx in split_iter:
        kfold.append({"train": sorted(ids[train_idx].tolist()), "val": sorted(ids[val_idx].tolist())})

    if labels is None:
        simple_splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, val_idx = next(simple_splitter.split(ids))
    else:
        simple_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, val_idx = next(simple_splitter.split(ids, labels))
    simple = {"train": sorted(ids[train_idx].tolist()), "val": sorted(ids[val_idx].tolist())}

    single_run = _stratified_train_val_test_split(ids, labels, random_state)

    outer_splitter, outer_labels = make_splitter()
    outer_iter = (
        outer_splitter.split(ids, outer_labels)
        if outer_labels is not None
        else outer_splitter.split(ids)
    )
    nested = []
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_iter):
        outer_train_ids = ids[outer_train_idx]
        outer_test_ids = ids[outer_test_idx]
        outer_vols = volumes[outer_train_idx]
        inner_labels, _ = _volume_bins(outer_vols, num_folds)
        if inner_labels is None:
            inner_splitter = KFold(
                n_splits=num_folds,
                shuffle=True,
                random_state=random_state + 100 + fold_idx,
            )
            inner_splits = list(inner_splitter.split(outer_train_ids))
        else:
            inner_splitter = StratifiedKFold(
                n_splits=num_folds,
                shuffle=True,
                random_state=random_state + 100 + fold_idx,
            )
            inner_splits = list(inner_splitter.split(outer_train_ids, inner_labels))
        inner_train_idx, inner_val_idx = inner_splits[fold_idx % num_folds]
        nested.append(
            {
                "train": sorted(outer_train_ids[inner_train_idx].tolist()),
                "val": sorted(outer_train_ids[inner_val_idx].tolist()),
                "test": sorted(outer_test_ids.tolist()),
            }
        )

    return {
        "kfold": {num_folds: kfold},
        "simple_train_val_split": {0.2: [simple]},
        "stratified_train_val_test_split": {0.8: [single_run]},
        "nested_volume_kfold": {num_folds: nested},
    }, info

def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    task_dir = args.output_root.resolve() / args.task_name
    tmp_dir = task_dir / "_tmp"
    task_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_cases(
        source_dir,
        include_mni=args.include_mni,
        min_lesion_mm3=args.min_lesion_mm3,
        max_lesion_mm3=args.max_lesion_mm3,
    )
    if not discovered:
        raise SystemExit(f"No ATLAS T1w + lesion-mask pairs found under {source_dir}")
    cases, subset_info = select_subset(discovered, args.subset_size, args.random_state)

    results: List[dict] = []
    failures: List[dict] = []
    if args.num_workers <= 1:
        for idx, case in enumerate(cases, 1):
            try:
                result = process_case(str(task_dir), str(tmp_dir), args.overwrite, case)
                results.append(result)
                print(f"[{idx}/{len(cases)}] OK   {case.case_id}", flush=True)
            except Exception as exc:
                failures.append({"case_id": case.case_id, "reason": str(exc)})
                print(f"[{idx}/{len(cases)}] FAIL {case.case_id}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(process_case, str(task_dir), str(tmp_dir), args.overwrite, case): case
                for case in cases
            }
            for idx, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{idx}/{len(cases)}] OK   {case.case_id}", flush=True)
                except Exception as exc:
                    failures.append({"case_id": case.case_id, "reason": str(exc)})
                    print(f"[{idx}/{len(cases)}] FAIL {case.case_id}: {exc}", flush=True)

    ok_rows = [row for row in results if row.get("status") in {"ok", "exists"}]
    if not ok_rows:
        raise SystemExit("No ATLAS cases were prepared successfully")

    fieldnames = [
        "case_id",
        "status",
        "t1_source",
        "label_source",
        "t1_used",
        "label_used",
        "label_resampled_to_t1",
        "raw_lesion_mm3",
        "lesion_voxels_1mm",
        "lesion_mm3_1mm",
        "original_spacing",
        "new_spacing",
        "crop_to_nonzero",
        "original_size",
        "new_size",
    ]
    manifest_path = task_dir / "manifest_atlas2_t1seg.tsv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in sorted(ok_rows, key=lambda x: x["case_id"]):
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    failures_path = task_dir / "failed_cases_atlas2_t1seg.json"
    failures_path.write_text(json.dumps(sorted(failures, key=lambda x: x["case_id"]), indent=2))

    splits, split_info = build_splits(ok_rows, args.num_folds, args.random_state)
    save_pickle(splits, str(task_dir / "splits.pkl"))

    volume_map = {row["case_id"]: float(row["lesion_mm3_1mm"]) for row in ok_rows}
    split_summary = {
        "task_name": args.task_name,
        "source_dir": str(source_dir),
        "task_dir": str(task_dir),
        "n_discovered": len(discovered),
        "n_cases": len(ok_rows),
        "n_failures": len(failures),
        **subset_info,
        **split_info,
        "nested_volume_kfold": [
            {
                "fold": idx,
                "train": _summary(fold["train"], volume_map),
                "val": _summary(fold["val"], volume_map),
                "test": _summary(fold["test"], volume_map),
            }
            for idx, fold in enumerate(splits["nested_volume_kfold"][args.num_folds])
        ],
    }
    (task_dir / "split_summary_atlas2_t1seg.json").write_text(
        json.dumps(split_summary, indent=2)
    )

    print("", flush=True)
    print(f"Discovered cases: {len(discovered)}", flush=True)
    print(f"Prepared cases: {len(ok_rows)}", flush=True)
    print(f"Failures: {len(failures)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Splits: {task_dir / 'splits.pkl'}", flush=True)

if __name__ == "__main__":
    main()
