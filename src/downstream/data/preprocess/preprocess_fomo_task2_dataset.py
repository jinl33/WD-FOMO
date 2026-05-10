import os
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
    save_pickle,
)
from yucca.functional.preprocessing import preprocess_case_for_training_with_label
from data.preprocessing_defaults import build_pretrain_style_preprocess_config
from data.task_configs import task2_config
from utils.utils import parallel_process


def process_subject(task_info):
    """
    Process a single subject for Task 2.

    Args:
        task_info: A tuple containing (subject, source_path, images_dir, labels_dir, pp_config, target_preprocessed, prefix)

    Returns:
        Success message or error message
    """
    (
        subject,
        source_path,
        images_dir,
        labels_dir,
        pp_config,
        target_preprocessed,
        prefix,
    ) = task_info
    modalities = pp_config["modalities"]

    try:
        session_path = join(images_dir, subject, "ses_1")
        label_path = join(labels_dir, subject, "ses_1", "seg.nii.gz")

        if not os.path.exists(session_path) or not os.path.exists(label_path):
            return f"Error: Missing data for {subject}"

        image_files = []
        modality_mapping = {}

        for file in os.listdir(session_path):
            if not file.endswith(".nii.gz"):
                continue

            if "dwi" in file.lower():
                modality_index = 0  # DWI
            elif "flair" in file.lower():
                modality_index = 1  # T2FLAIR
            elif "swi" in file.lower() or "t2s" in file.lower():
                modality_index = 2  # SWI_OR_T2STAR
            else:
                return f"Warning: Skipping file {file}"

            source_img = join(session_path, file)
            image_files.append(source_img)
            modality_mapping[modality_index] = source_img

        if len(image_files) < len(modalities):
            return f"Error: Not all modalities found for {subject}"

        images = [
            nib.load(modality_mapping[i])
            for i in range(len(modalities))
            if i in modality_mapping
        ]

        label = nib.load(label_path)

        preprocess_config = build_pretrain_style_preprocess_config(
            num_modalities=len(pp_config["modalities"]),
            norm_op=pp_config["norm_op"],
        )

        (
            preprocessed_data,
            preprocessed_label,
            properties,
        ) = preprocess_case_for_training_with_label(
            images=images,
            label=label,
            **preprocess_config,
        )

        data_with_label = preprocessed_data + [preprocessed_label]

        save_path = join(target_preprocessed, f"{prefix}_{subject}")
        np.save(save_path + ".npy", np.array(data_with_label, dtype=object))

        save_pickle(properties, save_path + ".pkl")

        return f"Processed {subject}"

    except Exception as e:
        return f"Error processing {subject}: {str(e)}"


def convert_and_preprocess_task2(
    source_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess all subjects for Task 2 in parallel.

    Args:
        source_path: Path to the source data directory
        output_path: Path where preprocessed data will be saved (optional)
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    pp_config = task2_config
    task_name = pp_config["task_name"]
    prefix = "FOMO2"

    labels_dir = join(source_path, "labels")
    images_dir = join(source_path, "preprocessed")

    target_preprocessed = join(output_path, task_name)

    ensure_dir_exists(target_preprocessed)

    subjects = sorted(os.listdir(images_dir))

    tasks = [
        (
            subject,
            source_path,
            images_dir,
            labels_dir,
            pp_config,
            target_preprocessed,
            prefix,
        )
        for subject in subjects
    ]

    parallel_process(
        process_subject, tasks, num_workers, desc="Processing subjects for Task 2"
    )

    print(f"Task 2 preprocessing completed. Data saved to {target_preprocessed}")
