#!/usr/bin/env python3
"""Clone the exact-pretrain full700 brain-age root, but rebuild BrainLat_sub-100
with a corrected x-spacing header before preprocessing.

Rationale:
- BrainLat_sub-100 is the only BrainLat skull-stripped source with header zooms
  (2.0, 1.0, 1.0), which yields an implausible ~274 mm LR foreground extent.
- The case is test-only, so a targeted root repair allows fair inference-only
  reevaluation without changing the trained models.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np


REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_SRC = REPO_ROOT / "src" / "downstream"
if str(DOWNSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_SRC))

from data.preprocessing_defaults import build_pretrain_style_preprocess_config


SRC_ROOT = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/"
    "fomo300k_unified_match_s23_p160_20260430_exactpretrain_full700"
)
SRC_TASK_DIR = SRC_ROOT / "Task_FOMO300K_BrainAge_T1_v2"

NEW_ROOT = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/"
    "fomo300k_unified_match_s23_p160_20260502_exactpretrain_full700_sub100headerfix"
)
NEW_TASK_DIR = NEW_ROOT / "Task_FOMO300K_BrainAge_T1_v2"

SUB100_STEM = "BrainLat_sub-100_T1w"
SUB100_SOURCE = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k/preprocessed_stripped/"
    "PT010_BrainLat/sub-100_ses-01_T1w_skull_stripped.nii.gz"
)
SUB100_FIXED_NIFTI = Path(
    "/nfs/turbo/umms-wilms1/FOMO/qc/brainlat_sub100/"
    "BrainLat_sub-100_source_skull_stripped_HEADERFIX_RAS_1x1x1.nii.gz"
)

PREPROCESS_CONFIG = build_pretrain_style_preprocess_config(num_modalities=1)


def ensure_fixed_header_nifti() -> Path:
    if SUB100_FIXED_NIFTI.exists():
        return SUB100_FIXED_NIFTI

    img = nib.load(str(SUB100_SOURCE))
    aff = img.affine.copy()
    col = aff[:3, 0]
    scale = np.linalg.norm(col)
    if scale == 0:
        raise RuntimeError("Invalid affine: zero-length first axis vector")
    aff[:3, 0] = col / scale * 1.0

    fixed = nib.Nifti1Image(img.get_fdata().astype(np.float32), aff, img.header.copy())
    fixed.header.set_zooms((1.0, 1.0, 1.0))
    SUB100_FIXED_NIFTI.parent.mkdir(parents=True, exist_ok=True)
    nib.save(fixed, str(SUB100_FIXED_NIFTI))
    return SUB100_FIXED_NIFTI


def clone_root_with_symlinks() -> None:
    NEW_TASK_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(SRC_TASK_DIR.iterdir()):
        if path.stem == SUB100_STEM and path.suffix in {".npy", ".pkl"}:
            continue
        dst = NEW_TASK_DIR / path.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(path, dst)


def rebuild_sub100() -> None:
    fixed_nifti = ensure_fixed_header_nifti()
    with tempfile.TemporaryDirectory() as td:
        tmp_fp = Path(td) / "sub100_fixed.nii.gz"
        shutil.copy2(fixed_nifti, tmp_fp)
        images, props = preprocess_case_for_training_without_label(
            images=[read_file_to_nifti_or_np(str(tmp_fp))],
            **PREPROCESS_CONFIG,
        )
    arr = images[0]
    np.save(NEW_TASK_DIR / f"{SUB100_STEM}.npy", arr)
    save_pickle(props, str(NEW_TASK_DIR / f"{SUB100_STEM}.pkl"))
    print(f"Rebuilt {SUB100_STEM}: shape={tuple(int(x) for x in arr.shape)}")
    print(f"  original_spacing={props.get('original_spacing')}")
    print(f"  new_size={props.get('new_size')}")


def validate() -> None:
    n_npy = len(list(NEW_TASK_DIR.glob("*.npy")))
    n_txt = len(list(NEW_TASK_DIR.glob("*.txt")))
    n_pkl = len(list(NEW_TASK_DIR.glob("*.pkl")))
    if n_npy != 700 or n_txt != 700 or n_pkl != 701:
        raise RuntimeError(f"Unexpected counts: npy={n_npy} txt={n_txt} pkl={n_pkl}")


def main() -> None:
    if not SRC_TASK_DIR.exists():
        raise FileNotFoundError(SRC_TASK_DIR)
    if not SUB100_SOURCE.exists():
        raise FileNotFoundError(SUB100_SOURCE)
    clone_root_with_symlinks()
    rebuild_sub100()
    validate()
    print(f"Done. New root: {NEW_TASK_DIR}")


if __name__ == "__main__":
    main()
