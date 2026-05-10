#!/usr/bin/env python3
"""Build a fully rebuilt brain-age task root using the same array preprocess family as pretraining.

This rebuild keeps the corrected 700-case benchmark composition and split artifacts, but regenerates
all 700 downstream arrays from source skull-stripped NIfTIs via the same preprocess config used for
the pretraining `.npy/.pkl` export:

    - normalization_operation = ["volume_wise_znorm"]
    - crop_to_nonzero = True
    - target_orientation = "RAS"
    - target_spacing = [1.0, 1.0, 1.0]
    - keep_aspect_ratio_when_using_target_size = False
    - transpose = [0, 1, 2]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np


REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_SRC = REPO_ROOT / "src" / "downstream"
if str(DOWNSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_SRC))

from data.preprocessing_defaults import build_pretrain_style_preprocess_config


CURRENT_ROOT = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/"
    "fomo300k_unified_match_s23_p160_20260428_gspmissing11fixed"
)
CURRENT_TASK_DIR = CURRENT_ROOT / "Task_FOMO300K_BrainAge_T1_v2"
NEW_ROOT = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/"
    "fomo300k_unified_match_s23_p160_20260430_exactpretrain_full700"
)
NEW_TASK_DIR = NEW_ROOT / "Task_FOMO300K_BrainAge_T1_v2"

GSP_STRIPPED_ROOT = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k_gsp/raw_nifti_stripped/PT016_GSP"
)
BRAINLAT_STRIPPED_ROOT = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k/preprocessed_stripped/PT010_BrainLat"
)
TASK3_SKULL_ROOT = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/fomo-fine-tuning-new/fomo-task3/skull_stripped"
)

PREPROCESS_CONFIG = build_pretrain_style_preprocess_config(num_modalities=1)


def _source_for_stem(stem: str) -> tuple[str, Path]:
    if stem.startswith("GSP_"):
        base = stem[len("GSP_") :]
        subject = base.replace("_ses-01_T1w", "")
        src = GSP_STRIPPED_ROOT / subject / "ses-01" / "anat" / f"{base}.nii.gz"
        cohort = "GSP"
    elif stem.startswith("BrainLat_"):
        base = stem[len("BrainLat_") :]
        subject = base.replace("_T1w", "")
        src = BRAINLAT_STRIPPED_ROOT / f"{subject}_ses-01_T1w_skull_stripped.nii.gz"
        cohort = "BrainLat"
    elif stem.startswith("Task3_"):
        base = stem[len("Task3_") :]
        subject = base.replace("_T1w", "")
        src = TASK3_SKULL_ROOT / subject / "ses_1" / "ss_t1.nii.gz"
        cohort = "Task3"
    else:
        raise ValueError(f"Unsupported case stem: {stem}")
    return cohort, src


def _process_case(task: tuple[str, str, str, str]) -> str:
    stem, src_nii, out_npy, out_pkl = task
    src_nii = Path(src_nii)
    out_npy = Path(out_npy)
    out_pkl = Path(out_pkl)

    if out_npy.exists() and out_pkl.exists():
        return f"SKIP {stem}"

    images, image_props = preprocess_case_for_training_without_label(
        images=[read_file_to_nifti_or_np(str(src_nii))],
        **PREPROCESS_CONFIG,
    )
    image = images[0]
    np.save(out_npy, image)
    save_pickle(image_props, str(out_pkl))
    return f"OK {stem} -> {tuple(int(x) for x in image.shape)}"


def build_root(num_workers: int) -> None:
    if not CURRENT_TASK_DIR.exists():
        raise FileNotFoundError(
            f"Missing current corrected task dir: {CURRENT_TASK_DIR}"
        )

    NEW_TASK_DIR.mkdir(parents=True, exist_ok=True)

    splits_src = CURRENT_TASK_DIR / "splits.pkl"
    splits_dst = NEW_TASK_DIR / "splits.pkl"
    if not splits_src.exists():
        raise FileNotFoundError(f"Missing split file: {splits_src}")
    if not splits_dst.exists():
        shutil.copy2(splits_src, splits_dst)

    tasks: list[tuple[str, str, str, str]] = []
    cohort_counts: Counter[str] = Counter()

    label_paths = sorted(CURRENT_TASK_DIR.glob("*.txt"))
    if len(label_paths) != 700:
        raise RuntimeError(
            f"Expected 700 labels in current root, got {len(label_paths)}"
        )

    for label_path in label_paths:
        stem = label_path.stem
        cohort, src = _source_for_stem(stem)
        if not src.exists():
            raise FileNotFoundError(f"Missing source for {stem}: {src}")

        cohort_counts[cohort] += 1

        dst_txt = NEW_TASK_DIR / label_path.name
        if not dst_txt.exists():
            os.symlink(label_path, dst_txt)

        tasks.append(
            (
                stem,
                str(src),
                str(NEW_TASK_DIR / f"{stem}.npy"),
                str(NEW_TASK_DIR / f"{stem}.pkl"),
            )
        )

    expected = Counter({"GSP": 250, "BrainLat": 250, "Task3": 200})
    if cohort_counts != expected:
        raise RuntimeError(f"Unexpected cohort counts: {cohort_counts} != {expected}")

    print(f"Queued cases by cohort: {dict(cohort_counts)}", flush=True)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(_process_case, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if i % 25 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {msg}", flush=True)

    n_npy = len(list(NEW_TASK_DIR.glob("*.npy")))
    n_txt = len(list(NEW_TASK_DIR.glob("*.txt")))
    n_pkl = len(list(NEW_TASK_DIR.glob("*.pkl")))
    has_split = splits_dst.exists()
    print(f"Final counts: npy={n_npy} txt={n_txt} pkl={n_pkl} splits={has_split}")
    if n_npy != 700 or n_txt != 700 or n_pkl != 701 or not has_split:
        raise RuntimeError("Exact-pretrain full700 root validation failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    build_root(args.num_workers)


if __name__ == "__main__":
    main()
