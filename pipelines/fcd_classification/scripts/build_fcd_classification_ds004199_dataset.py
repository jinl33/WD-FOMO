#!/usr/bin/env python3
"""Prepare OpenNeuro ds004199 as corrected Task 9 (binary FCD classification).

This replaces the invalid legacy Task 8 export, which duplicated channels by
aliasing FLAIR into a fake T2 slot. The corrected task uses the actual
modalities present in ds004199: T1 + FLAIR.

Processing policy:
- SynthStrip only the T1 volume
- resample FLAIR into T1 space before preprocessing
- apply the T1 brain mask to the aligned FLAIR
- reorient to RAS
- resample to 1 mm isotropic
- crop to nonzero foreground
- per-modality volume-wise z-normalization

Outputs:
- <output_root>/<task_name>/<case_id>.npy
- <output_root>/<task_name>/<case_id>.txt
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
from typing import Dict, Iterable, List, Optional

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from nibabel.processing import resample_from_to
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np


REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_SRC = REPO_ROOT / "src" / "downstream"
if str(DOWNSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_SRC))

from data.preprocessing_defaults import build_pretrain_style_preprocess_config


TASK_NAME = "Task009_OpenNeuro_ds004199_FCD_T1FLAIR_1mm"
PREFIX = "ds004199"
PREPROCESS_CONFIG = build_pretrain_style_preprocess_config(num_modalities=2)


@dataclass(frozen=True)
class SubjectCase:
    case_id: str
    participant_id: str
    label: int
    group: str
    official_split: str
    t1_path: str
    flair_path: str
    flair_roi_path: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source_dir", type=Path, required=True, help="Path to ds004199 root"
    )
    p.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root dir that will contain the task folder",
    )
    p.add_argument("--task_name", type=str, default=TASK_NAME)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_skull_strip", action="store_true")
    p.add_argument(
        "--allow_unstripped_fallback",
        action="store_true",
        help="If skull stripping fails for a case, continue with the unstripped images.",
    )
    p.add_argument(
        "--limit_subjects",
        type=int,
        default=0,
        help="For debugging only: process at most this many subjects.",
    )
    return p.parse_args()


def map_group_to_label(group_value: str) -> Optional[int]:
    g = group_value.strip().lower()
    if g in {"hc", "control", "healthy", "cn"}:
        return 0
    if g in {"fcd", "focal cortical dysplasia", "patient", "patients"}:
        return 1
    return None


def load_participants(participants_tsv: Path) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(participants_tsv.open(), delimiter="\t"))
    cleaned = []
    for row in rows:
        cleaned.append({(k or "").lstrip("\ufeff"): (v or "") for k, v in row.items()})
    return cleaned


def pick_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


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

    if t1 is None or flair is None:
        return None

    return {
        "t1_path": str(t1),
        "flair_path": str(flair),
        "flair_roi_path": str(flair_roi) if flair_roi is not None else "",
    }


def build_cases(source_dir: Path, limit_subjects: int = 0) -> List[SubjectCase]:
    participants = load_participants(source_dir / "participants.tsv")
    sub_dirs = {p.name: p for p in sorted(source_dir.glob("sub-*")) if p.is_dir()}
    cases: List[SubjectCase] = []
    for row in participants:
        participant_id = row["participant_id"].strip()
        label = map_group_to_label(row.get("group", ""))
        if not participant_id or label is None:
            continue
        sub_dir = sub_dirs.get(participant_id)
        if sub_dir is None:
            continue
        mods = find_subject_modalities(sub_dir)
        if mods is None:
            continue
        case_id = f"{PREFIX}_{participant_id}"
        cases.append(
            SubjectCase(
                case_id=case_id,
                participant_id=participant_id,
                label=label,
                group=row.get("group", "").strip(),
                official_split=row.get("split", "").strip(),
                t1_path=mods["t1_path"],
                flair_path=mods["flair_path"],
                flair_roi_path=mods["flair_roi_path"],
            )
        )
    if limit_subjects > 0:
        cases = cases[:limit_subjects]
    return cases


def run_synthstrip(input_path: Path, output_path: Path) -> tuple[bool, str]:
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


def _align_flair_to_t1(
    t1_ref: nib.Nifti1Image,
    flair_img: nib.Nifti1Image,
    t1_mask: Optional[np.ndarray],
) -> nib.Nifti1Image:
    flair_aligned = resample_from_to(flair_img, t1_ref, order=1)
    if t1_mask is None:
        return flair_aligned

    flair_data = np.asarray(flair_aligned.dataobj).astype(np.float32)
    flair_data *= t1_mask.astype(np.float32)
    return nib.Nifti1Image(flair_data, flair_aligned.affine, flair_aligned.header)


def process_case(
    task_dir_str: str,
    skull_dir_str: str,
    no_skull_strip: bool,
    allow_unstripped_fallback: bool,
    overwrite: bool,
    case: SubjectCase,
) -> dict:
    task_dir = Path(task_dir_str)
    skull_dir = Path(skull_dir_str)
    npy_out = task_dir / f"{case.case_id}.npy"
    txt_out = task_dir / f"{case.case_id}.txt"
    pkl_out = task_dir / f"{case.case_id}.pkl"

    if npy_out.exists() and txt_out.exists() and pkl_out.exists() and not overwrite:
        return {
            "case_id": case.case_id,
            "participant_id": case.participant_id,
            "label": case.label,
            "group": case.group,
            "official_split": case.official_split,
            "status": "exists",
            "t1_source": case.t1_path,
            "flair_source": case.flair_path,
            "flair_roi_source": case.flair_roi_path,
        }

    raw_t1 = Path(case.t1_path)
    raw_flair = Path(case.flair_path)
    strip_notes: List[str] = []

    t1_used = raw_t1
    t1_mask = None
    if not no_skull_strip:
        t1_strip = skull_dir / f"{case.case_id}_T1w.nii.gz"
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
    flair_aligned = _align_flair_to_t1(t1_img, flair_img, t1_mask)

    aligned_dir = skull_dir / "aligned"
    t1_tmp = aligned_dir / f"{case.case_id}_T1w.nii.gz"
    flair_tmp = aligned_dir / f"{case.case_id}_FLAIR_inT1.nii.gz"
    _save_aligned_nifti(t1_img, t1_tmp)
    _save_aligned_nifti(flair_aligned, flair_tmp)

    preprocessed, props = preprocess_case_for_training_without_label(
        images=[
            read_file_to_nifti_or_np(str(t1_tmp)),
            read_file_to_nifti_or_np(str(flair_tmp)),
        ],
        **PREPROCESS_CONFIG,
    )
    arr = np.asarray(preprocessed, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != 2:
        raise RuntimeError(
            f"unexpected preprocessed shape for {case.case_id}: {arr.shape}"
        )

    np.save(npy_out, arr)
    txt_out.write_text(f"{case.label}\n")
    save_pickle(props, str(pkl_out))

    return {
        "case_id": case.case_id,
        "participant_id": case.participant_id,
        "label": case.label,
        "group": case.group,
        "official_split": case.official_split,
        "status": "ok",
        "t1_source": case.t1_path,
        "flair_source": case.flair_path,
        "flair_roi_source": case.flair_roi_path,
        "t1_used": str(t1_used),
        "flair_used": str(flair_tmp),
        "shape": list(arr.shape),
        "skull_strip_notes": " | ".join(strip_notes),
        **_props_subset(props),
    }


def build_splits(cases: List[SubjectCase], num_folds: int, random_state: int) -> dict:
    ids = np.array([case.case_id for case in cases])
    y = np.array([case.label for case in cases])
    unique, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if len(counts) > 0 else 0
    effective_folds = min(num_folds, len(ids), min_class) if min_class > 0 else 0

    kfold = []
    if effective_folds >= 2:
        skf = StratifiedKFold(
            n_splits=effective_folds, shuffle=True, random_state=random_state
        )
        for train_idx, val_idx in skf.split(ids, y):
            kfold.append(
                {"train": ids[train_idx].tolist(), "val": ids[val_idx].tolist()}
            )

    if len(ids) >= 2 and len(unique) >= 2:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        train_idx, val_idx = next(sss.split(ids, y))
        simple = {"train": ids[train_idx].tolist(), "val": ids[val_idx].tolist()}
    else:
        simple = {"train": ids.tolist(), "val": []}

    return {
        "kfold": {effective_folds if effective_folds >= 2 else num_folds: kfold},
        "simple_train_val_split": {0.2: [simple]},
    }


def split_summary(ids: List[str], label_map: Dict[str, int]) -> Dict[str, int]:
    labels = [label_map[i] for i in ids]
    return {
        "n": len(ids),
        "control": int(sum(l == 0 for l in labels)),
        "fcd": int(sum(l == 1 for l in labels)),
    }


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    task_dir = output_root / args.task_name
    skull_dir = task_dir / "_tmp_skullstrip"
    task_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(source_dir, args.limit_subjects)
    if not cases:
        raise SystemExit("No valid ds004199 T1+FLAIR cases found")

    results: List[dict] = []
    failures: List[dict] = []

    if args.num_workers <= 1:
        iterator = enumerate(cases, 1)
        for idx, case in iterator:
            try:
                result = process_case(
                    str(task_dir),
                    str(skull_dir),
                    args.no_skull_strip,
                    args.allow_unstripped_fallback,
                    args.overwrite,
                    case,
                )
                results.append(result)
                print(
                    f"[{idx}/{len(cases)}] OK   {case.case_id} -> {result.get('shape', 'exists')}"
                )
            except Exception as exc:
                failures.append(
                    {
                        "case_id": case.case_id,
                        "participant_id": case.participant_id,
                        "label": case.label,
                        "group": case.group,
                        "official_split": case.official_split,
                        "t1_source": case.t1_path,
                        "flair_source": case.flair_path,
                        "flair_roi_source": case.flair_roi_path,
                        "reason": str(exc),
                    }
                )
                print(f"[{idx}/{len(cases)}] FAIL {case.case_id}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(
                    process_case,
                    str(task_dir),
                    str(skull_dir),
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
                        f"[{idx}/{len(cases)}] OK   {case.case_id} -> {result.get('shape', 'exists')}"
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "case_id": case.case_id,
                            "participant_id": case.participant_id,
                            "label": case.label,
                            "group": case.group,
                            "official_split": case.official_split,
                            "t1_source": case.t1_path,
                            "flair_source": case.flair_path,
                            "flair_roi_source": case.flair_roi_path,
                            "reason": str(exc),
                        }
                    )
                    print(f"[{idx}/{len(cases)}] FAIL {case.case_id}: {exc}")

    ok_rows = [r for r in results if r.get("status") in {"ok", "exists"}]
    ok_case_ids = {r["case_id"] for r in ok_rows}
    ok_cases = [case for case in cases if case.case_id in ok_case_ids]
    if len(ok_cases) == 0:
        raise SystemExit("No valid cases were prepared")

    manifest_path = task_dir / "manifest_ds004199_t1flair.tsv"
    with manifest_path.open("w", newline="") as f:
        fieldnames = [
            "case_id",
            "participant_id",
            "label",
            "group",
            "official_split",
            "status",
            "t1_source",
            "flair_source",
            "flair_roi_source",
            "t1_used",
            "flair_used",
            "shape",
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

    failures_path = task_dir / "failed_cases_ds004199_t1flair.json"
    failures_path.write_text(
        json.dumps(sorted(failures, key=lambda r: r["case_id"]), indent=2)
    )

    splits = build_splits(
        ok_cases, num_folds=args.num_folds, random_state=args.random_state
    )
    save_pickle(splits, str(task_dir / "splits.pkl"))

    label_map = {case.case_id: case.label for case in ok_cases}
    kfold_key = next(iter(splits["kfold"].keys()))
    split_report = {
        "task_name": args.task_name,
        "source_dir": str(source_dir),
        "task_dir": str(task_dir),
        "n_cases": len(ok_cases),
        "n_failures": len(failures),
        "class_balance": split_summary([case.case_id for case in ok_cases], label_map),
        "official_split_counts": {
            split: sum(1 for case in ok_cases if case.official_split == split)
            for split in sorted({case.official_split for case in ok_cases})
        },
        "kfold_n_splits": kfold_key,
        "kfold": [
            {
                "fold": fold_idx,
                "train": split_summary(split["train"], label_map),
                "val": split_summary(split["val"], label_map),
            }
            for fold_idx, split in enumerate(splits["kfold"][kfold_key])
        ],
        "simple_train_val_split_0.2": {
            "train": split_summary(
                splits["simple_train_val_split"][0.2][0]["train"], label_map
            ),
            "val": split_summary(
                splits["simple_train_val_split"][0.2][0]["val"], label_map
            ),
        },
    }
    (task_dir / "split_summary_ds004199_t1flair.json").write_text(
        json.dumps(split_report, indent=2)
    )

    print("")
    print(f"Prepared cases: {len(ok_cases)}")
    print(f"Failures: {len(failures)}")
    print(f"Manifest: {manifest_path}")
    print(f"Splits: {task_dir / 'splits.pkl'}")


if __name__ == "__main__":
    main()
