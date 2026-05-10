#!/usr/bin/env python3
"""Prepare OpenNeuro ds004199 as corrected Task 10 (FCD lesion segmentation).

This builds a dense segmentation benchmark from the ds004199 FCD-positive
subjects that include lesion ROI masks. The benchmark is intentionally kept
outside FOMO60K overlap: ds004199 is part of the PT030 OpenNeuro slice of
FOMO300K and is not one of the OpenNeuro subsets used for FOMO45K/60K.

Processing policy:
- use only ROI-positive FCD subjects
- SynthStrip only the T1 volume
- resample FLAIR and ROI into T1 space before preprocessing
- apply the T1 brain mask to the aligned FLAIR
- reorient to RAS
- resample to 1 mm isotropic
- crop to nonzero
- volume-wise z-normalize modalities

Outputs:
- <output_root>/<task_name>/<case_id>.npy   object array: [T1, FLAIR, seg]
- <output_root>/<task_name>/<case_id>.pkl
- manifest / failures / split summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from nibabel.processing import resample_from_to
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from yucca.functional.preprocessing import preprocess_case_for_training_with_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np


REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_SRC = REPO_ROOT / "src" / "downstream"
if str(DOWNSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_SRC))

from data.preprocessing_defaults import build_pretrain_style_preprocess_config


TASK_NAME = "Task010_OpenNeuro_ds004199_FCDSeg_T1FLAIR_1mm"
PREFIX = "ds004199seg"
PREPROCESS_CONFIG = build_pretrain_style_preprocess_config(num_modalities=2)


@dataclass(frozen=True)
class SegCase:
    case_id: str
    participant_id: str
    official_split: str
    t1_path: str
    flair_path: str
    flair_roi_path: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--task_name", type=str, default=TASK_NAME)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_skull_strip", action="store_true")
    p.add_argument("--allow_unstripped_fallback", action="store_true")
    p.add_argument("--limit_subjects", type=int, default=0)
    return p.parse_args()


def load_participants(participants_tsv: Path) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(participants_tsv.open(), delimiter="\t"))
    cleaned = []
    for row in rows:
        cleaned.append({(k or "").lstrip("\ufeff"): (v or "") for k, v in row.items()})
    return cleaned


def is_fcd(group_value: str) -> bool:
    g = group_value.strip().lower()
    return g in {"fcd", "focal cortical dysplasia", "patient", "patients"}


def find_subject_modalities(sub_dir: Path) -> Optional[Dict[str, str]]:
    anat_dir = sub_dir / "anat"
    if not anat_dir.exists():
        anat_dir = sub_dir / "ses-01" / "anat"
    if not anat_dir.exists():
        return None

    files = sorted(p for p in anat_dir.glob("*.nii.gz") if p.is_file())
    t1 = None
    flair = None
    flair_roi = None
    for path in files:
        name = path.name.lower()
        if name.endswith("_roi.nii.gz"):
            if "flair" in name and flair_roi is None:
                flair_roi = path
            continue
        if t1 is None and "t1w" in name:
            t1 = path
        if flair is None and "flair" in name:
            flair = path

    if t1 is None or flair is None or flair_roi is None:
        return None

    return {
        "t1_path": str(t1),
        "flair_path": str(flair),
        "flair_roi_path": str(flair_roi),
    }


def build_cases(source_dir: Path, limit_subjects: int = 0) -> List[SegCase]:
    participants = load_participants(source_dir / "participants.tsv")
    sub_dirs = {p.name: p for p in sorted(source_dir.glob("sub-*")) if p.is_dir()}
    cases: List[SegCase] = []
    for row in participants:
        participant_id = row.get("participant_id", "").strip()
        if not participant_id or not is_fcd(row.get("group", "")):
            continue
        sub_dir = sub_dirs.get(participant_id)
        if sub_dir is None:
            continue
        mods = find_subject_modalities(sub_dir)
        if mods is None:
            continue
        case_id = f"{PREFIX}_{participant_id}"
        cases.append(
            SegCase(
                case_id=case_id,
                participant_id=participant_id,
                official_split=row.get("split", "").strip(),
                t1_path=mods["t1_path"],
                flair_path=mods["flair_path"],
                flair_roi_path=mods["flair_roi_path"],
            )
        )
    if limit_subjects > 0:
        cases = cases[:limit_subjects]
    return cases


def run_synthstrip(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    cmd = shutil.which("mri_synthstrip")
    if cmd is None:
        return False, "mri_synthstrip not found in PATH"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [cmd, "-i", str(input_path), "-o", str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "unknown synthstrip error").strip()
    return True, "ok"


def _props_subset(props: dict) -> dict:
    keep = {}
    for key in (
        "original_spacing",
        "new_spacing",
        "crop_to_nonzero",
        "original_size",
        "new_size",
    ):
        if key in props:
            value = props[key]
            if isinstance(value, np.ndarray):
                keep[key] = value.tolist()
            elif isinstance(value, tuple):
                keep[key] = list(value)
            else:
                keep[key] = value
    return keep


def _save_aligned_nifti(img: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def _align_flair_and_roi(
    t1_ref: nib.Nifti1Image,
    flair_img: nib.Nifti1Image,
    roi_img: nib.Nifti1Image,
    t1_mask: Optional[np.ndarray],
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    flair_aligned = resample_from_to(flair_img, t1_ref, order=1)
    roi_aligned = resample_from_to(roi_img, t1_ref, order=0)

    if t1_mask is not None:
        flair_data = np.asarray(flair_aligned.dataobj).astype(np.float32)
        flair_data *= t1_mask.astype(np.float32)
        flair_aligned = nib.Nifti1Image(
            flair_data, flair_aligned.affine, flair_aligned.header
        )

    roi_data = (np.asarray(roi_aligned.dataobj) > 0.5).astype(np.uint8)
    roi_aligned = nib.Nifti1Image(roi_data, roi_aligned.affine, roi_aligned.header)
    return flair_aligned, roi_aligned


def process_case(
    task_dir_str: str,
    tmp_dir_str: str,
    no_skull_strip: bool,
    allow_unstripped_fallback: bool,
    overwrite: bool,
    case: SegCase,
) -> dict:
    task_dir = Path(task_dir_str)
    tmp_dir = Path(tmp_dir_str)
    npy_out = task_dir / f"{case.case_id}.npy"
    pkl_out = task_dir / f"{case.case_id}.pkl"

    if npy_out.exists() and pkl_out.exists() and not overwrite:
        return {
            "case_id": case.case_id,
            "participant_id": case.participant_id,
            "official_split": case.official_split,
            "status": "exists",
            "t1_source": case.t1_path,
            "flair_source": case.flair_path,
            "flair_roi_source": case.flair_roi_path,
        }

    raw_t1 = Path(case.t1_path)
    raw_flair = Path(case.flair_path)
    raw_roi = Path(case.flair_roi_path)
    strip_notes: List[str] = []

    t1_used = raw_t1
    t1_mask = None
    if not no_skull_strip:
        t1_strip = tmp_dir / "t1_skullstrip" / f"{case.case_id}_T1w.nii.gz"
        ok_t1, msg_t1 = run_synthstrip(raw_t1, t1_strip)
        if ok_t1:
            t1_used = t1_strip
            t1_strip_img = nib.load(str(t1_strip))
            t1_mask = np.asarray(t1_strip_img.dataobj) != 0
        else:
            strip_notes.append(f"T1:{msg_t1}")
            if not allow_unstripped_fallback:
                raise RuntimeError("skull_strip_failed: " + " | ".join(strip_notes))

    t1_img = nib.load(str(t1_used))
    flair_img = nib.load(str(raw_flair))
    roi_img = nib.load(str(raw_roi))

    flair_aligned, roi_aligned = _align_flair_and_roi(
        t1_img, flair_img, roi_img, t1_mask
    )

    aligned_dir = tmp_dir / "aligned"
    t1_tmp = aligned_dir / f"{case.case_id}_T1w.nii.gz"
    flair_tmp = aligned_dir / f"{case.case_id}_FLAIR_inT1.nii.gz"
    roi_tmp = aligned_dir / f"{case.case_id}_ROI_inT1.nii.gz"
    _save_aligned_nifti(t1_img, t1_tmp)
    _save_aligned_nifti(flair_aligned, flair_tmp)
    _save_aligned_nifti(roi_aligned, roi_tmp)

    (
        preprocessed_data,
        preprocessed_label,
        props,
    ) = preprocess_case_for_training_with_label(
        images=[
            read_file_to_nifti_or_np(str(t1_tmp)),
            read_file_to_nifti_or_np(str(flair_tmp)),
        ],
        label=read_file_to_nifti_or_np(str(roi_tmp)),
        **PREPROCESS_CONFIG,
    )

    data_with_label = list(preprocessed_data) + [preprocessed_label]
    np.save(npy_out, np.array(data_with_label, dtype=object))
    save_pickle(props, str(pkl_out))

    lesion_voxels = int(np.sum(np.asarray(roi_aligned.dataobj) > 0.5))
    return {
        "case_id": case.case_id,
        "participant_id": case.participant_id,
        "official_split": case.official_split,
        "status": "ok",
        "t1_source": case.t1_path,
        "flair_source": case.flair_path,
        "flair_roi_source": case.flair_roi_path,
        "t1_used": str(t1_used),
        "flair_used": str(flair_tmp),
        "roi_used": str(roi_tmp),
        "shape": list(np.asarray(data_with_label, dtype=object).shape),
        "lesion_voxels_1mm": lesion_voxels,
        "lesion_mm3_1mm": float(lesion_voxels),
        "skull_strip_notes": " | ".join(strip_notes),
        **_props_subset(props),
    }


def _volume_bins(
    volumes: np.ndarray, num_folds: int
) -> Tuple[Optional[np.ndarray], dict]:
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
    return None, {"method": "random_kfold_fallback"}


def build_splits(
    case_rows: List[dict], num_folds: int, random_state: int
) -> Tuple[dict, dict]:
    ids = np.array([row["case_id"] for row in case_rows])
    volumes = np.array([row["lesion_mm3_1mm"] for row in case_rows], dtype=float)

    volume_labels, volume_info = _volume_bins(volumes, num_folds)
    kfold = []
    if volume_labels is not None:
        splitter = StratifiedKFold(
            n_splits=min(num_folds, len(ids)), shuffle=True, random_state=random_state
        )
        for train_idx, val_idx in splitter.split(ids, volume_labels):
            kfold.append(
                {"train": ids[train_idx].tolist(), "val": ids[val_idx].tolist()}
            )
    else:
        splitter = KFold(
            n_splits=min(num_folds, len(ids)), shuffle=True, random_state=random_state
        )
        for train_idx, val_idx in splitter.split(ids):
            kfold.append(
                {"train": ids[train_idx].tolist(), "val": ids[val_idx].tolist()}
            )

    if volume_labels is not None:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        train_idx, val_idx = next(sss.split(ids, volume_labels))
    else:
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, val_idx = next(ss.split(ids))
    simple = {"train": ids[train_idx].tolist(), "val": ids[val_idx].tolist()}

    split_meta = {
        "n_subjects": int(len(ids)),
        "volume_mm3_mean": float(np.mean(volumes)),
        "volume_mm3_std": float(np.std(volumes)),
        "volume_mm3_median": float(np.median(volumes)),
        **volume_info,
    }
    return {
        "kfold": {num_folds: kfold},
        "simple_train_val_split": {0.2: [simple]},
    }, split_meta


def split_summary(ids: List[str], volume_map: Dict[str, float]) -> Dict[str, float]:
    vols = (
        np.array([volume_map[i] for i in ids], dtype=float)
        if ids
        else np.array([], dtype=float)
    )
    if vols.size == 0:
        return {
            "n": 0,
            "volume_mm3_mean": 0.0,
            "volume_mm3_std": 0.0,
            "volume_mm3_median": 0.0,
        }
    return {
        "n": int(len(ids)),
        "volume_mm3_mean": float(np.mean(vols)),
        "volume_mm3_std": float(np.std(vols)),
        "volume_mm3_median": float(np.median(vols)),
    }


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    task_dir = output_root / args.task_name
    tmp_dir = task_dir / "_tmp"
    task_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(source_dir, args.limit_subjects)
    if not cases:
        raise SystemExit("No valid ds004199 T1+FLAIR+ROI cases found")

    results: List[dict] = []
    failures: List[dict] = []

    if args.num_workers <= 1:
        iterator = enumerate(cases, 1)
        for idx, case in iterator:
            try:
                result = process_case(
                    str(task_dir),
                    str(tmp_dir),
                    args.no_skull_strip,
                    args.allow_unstripped_fallback,
                    args.overwrite,
                    case,
                )
                results.append(result)
                print(
                    f"[{idx}/{len(cases)}] OK   {case.case_id} -> lesion_mm3={result.get('lesion_mm3_1mm', 'exists')}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "case_id": case.case_id,
                        "participant_id": case.participant_id,
                        "official_split": case.official_split,
                        "t1_source": case.t1_path,
                        "flair_source": case.flair_path,
                        "flair_roi_source": case.flair_roi_path,
                        "reason": str(exc),
                    }
                )
                print(f"[{idx}/{len(cases)}] FAIL {case.case_id}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(
                    process_case,
                    str(task_dir),
                    str(tmp_dir),
                    args.no_skull_strip,
                    args.allow_unstripped_fallback,
                    args.overwrite,
                    case,
                ): case
                for case in cases
            }
            for idx, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"[{idx}/{len(cases)}] OK   {case.case_id} -> lesion_mm3={result.get('lesion_mm3_1mm', 'exists')}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "case_id": case.case_id,
                            "participant_id": case.participant_id,
                            "official_split": case.official_split,
                            "t1_source": case.t1_path,
                            "flair_source": case.flair_path,
                            "flair_roi_source": case.flair_roi_path,
                            "reason": str(exc),
                        }
                    )
                    print(
                        f"[{idx}/{len(cases)}] FAIL {case.case_id}: {exc}", flush=True
                    )

    ok_rows = [r for r in results if r.get("status") in {"ok", "exists"}]
    if len(ok_rows) == 0:
        raise SystemExit("No valid cases were prepared")

    manifest_path = task_dir / "manifest_ds004199_t1flairseg.tsv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = [
            "case_id",
            "participant_id",
            "official_split",
            "status",
            "t1_source",
            "flair_source",
            "flair_roi_source",
            "t1_used",
            "flair_used",
            "roi_used",
            "shape",
            "lesion_voxels_1mm",
            "lesion_mm3_1mm",
            "original_spacing",
            "new_spacing",
            "crop_to_nonzero",
            "original_size",
            "new_size",
            "skull_strip_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in sorted(ok_rows, key=lambda r: r["case_id"]):
            writer.writerow(row)

    failures_path = task_dir / "failed_cases_ds004199_t1flairseg.json"
    failures_path.write_text(
        json.dumps(sorted(failures, key=lambda r: r["case_id"]), indent=2)
    )

    splits, split_meta = build_splits(
        ok_rows, num_folds=args.num_folds, random_state=args.random_state
    )
    save_pickle(splits, str(task_dir / "splits.pkl"))

    volume_map = {row["case_id"]: float(row["lesion_mm3_1mm"]) for row in ok_rows}
    split_report = {
        "task_name": args.task_name,
        "source_dir": str(source_dir),
        "task_dir": str(task_dir),
        "n_cases": len(ok_rows),
        "n_failures": len(failures),
        "official_split_counts": {
            split: sum(1 for row in ok_rows if row["official_split"] == split)
            for split in sorted({row["official_split"] for row in ok_rows})
        },
        **split_meta,
        "kfold": [
            {
                "fold": fold_idx,
                "train": split_summary(split["train"], volume_map),
                "val": split_summary(split["val"], volume_map),
            }
            for fold_idx, split in enumerate(splits["kfold"][args.num_folds])
        ],
        "simple_train_val_split_0.2": {
            "train": split_summary(
                splits["simple_train_val_split"][0.2][0]["train"], volume_map
            ),
            "val": split_summary(
                splits["simple_train_val_split"][0.2][0]["val"], volume_map
            ),
        },
    }
    (task_dir / "split_summary_ds004199_t1flairseg.json").write_text(
        json.dumps(split_report, indent=2)
    )

    print("", flush=True)
    print(f"Prepared cases: {len(ok_rows)}", flush=True)
    print(f"Failures: {len(failures)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Splits: {task_dir / 'splits.pkl'}", flush=True)


if __name__ == "__main__":
    main()
