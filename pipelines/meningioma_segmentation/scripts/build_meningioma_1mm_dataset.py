#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from yucca.functional.preprocessing import preprocess_case_for_training_with_label

from data.preprocessing_defaults import build_pretrain_style_preprocess_config
from data.task_configs import task2_config
from utils.utils import parallel_process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("/nfs/turbo/umms-wilms1/FOMO/Data/fomo-fine-tuning-new/fomo-task2"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/nfs/turbo/umms-wilms1/FOMO/Data/fomo-task2_meningioma_1mm_20260513"),
    )
    parser.add_argument(
        "--image_source",
        choices=["skull_stripped", "preprocessed"],
        default="skull_stripped",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def subject_number(subject: str) -> int:
    return int(subject.rsplit("_", 1)[-1])


def find_modality(session_dir: Path, patterns: tuple[str, ...]) -> Path:
    matches = []
    for pattern in patterns:
        matches.extend(session_dir.glob(pattern))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"Missing modality in {session_dir}: {patterns}")
    return matches[0]


def modality_paths(source_root: Path, image_source: str, subject: str) -> list[Path]:
    session_dir = source_root / image_source / subject / "ses_1"
    if image_source == "skull_stripped":
        return [
            find_modality(session_dir, ("ss_dwi_b1000.nii.gz", "*dwi*.nii.gz")),
            find_modality(session_dir, ("ss_flair.nii.gz", "*flair*.nii.gz")),
            find_modality(session_dir, ("ss_swi.nii.gz", "ss_t2s.nii.gz", "*swi*.nii.gz", "*t2s*.nii.gz")),
        ]
    return [
        find_modality(session_dir, ("dwi_b1000.nii.gz", "*dwi*.nii.gz")),
        find_modality(session_dir, ("flair.nii.gz", "*flair*.nii.gz")),
        find_modality(session_dir, ("swi.nii.gz", "t2s.nii.gz", "*swi*.nii.gz", "*t2s*.nii.gz")),
    ]


def process_subject(task: dict) -> dict:
    source_root = Path(task["source_root"])
    task_dir = Path(task["task_dir"])
    subject = task["subject"]
    image_source = task["image_source"]

    label_path = source_root / "labels" / subject / "ses_1" / "seg.nii.gz"
    image_paths = modality_paths(source_root, image_source, subject)
    images = [nib.load(str(path)) for path in image_paths]
    label = nib.load(str(label_path))

    preprocess_config = build_pretrain_style_preprocess_config(
        num_modalities=len(task2_config["modalities"]),
        norm_op=task2_config["norm_op"],
    )
    data, seg, properties = preprocess_case_for_training_with_label(
        images=images,
        label=label,
        **preprocess_config,
    )

    case_id = f"FOMO2_{subject}"
    out_base = task_dir / case_id
    np.save(str(out_base) + ".npy", np.array(data + [seg], dtype=object))
    properties["source_subject"] = subject
    properties["source_image_paths"] = [str(path) for path in image_paths]
    properties["source_label_path"] = str(label_path)
    properties["image_source"] = image_source
    save_pickle(properties, str(out_base) + ".pkl")

    return {
        "case_id": case_id,
        "subject": subject,
        "image_source": image_source,
        "new_spacing": properties.get("new_spacing"),
        "new_size": properties.get("new_size"),
        "foreground_voxels": int((seg > 0).sum()),
        "source_spacing": [float(x) for x in label.header.get_zooms()[:3]],
        "source_shape": list(label.shape),
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    task_dir = args.output_root / task2_config["task_name"]
    task_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(
        [p.name for p in (args.source_root / "labels").iterdir() if p.is_dir()],
        key=subject_number,
    )
    tasks = [
        {
            "source_root": str(args.source_root),
            "task_dir": str(task_dir),
            "subject": subject,
            "image_source": args.image_source,
        }
        for subject in subjects
    ]

    results = parallel_process(
        process_subject,
        tasks,
        num_workers=args.num_workers,
        desc="Building Task2 1mm dataset",
    )
    manifest = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "task_dir": str(task_dir),
        "image_source": args.image_source,
        "preprocessing": build_pretrain_style_preprocess_config(
            num_modalities=len(task2_config["modalities"]),
            norm_op=task2_config["norm_op"],
        ),
        "subjects": results,
    }
    manifest_path = task_dir / "manifest_meningioma_1mm.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"task_dir": str(task_dir), "n_subjects": len(results)}, indent=2))


if __name__ == "__main__":
    main()
