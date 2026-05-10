import os
import shutil
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from data.preprocessing_defaults import build_pretrain_style_preprocess_config
from data.task_configs import task1_config
from utils.utils import parallel_process


def process_subject(task_info):
    """
    Process a single subject for Task 1.

    Args:
        task_info: A tuple containing (folder_name, source_path, labels_dir, pp_config, target_preprocessed, prefix)

    Returns:
        Success message or error message
    """
    (
        folder_name,
        source_path,
        labels_dir,
        pp_config,
        target_preprocessed,
        prefix,
    ) = task_info
    modalities = pp_config["modalities"]

    try:
        images_dir = join(source_path, "preprocessed")
        session_path = join(images_dir, folder_name, "ses_1")

        if not os.path.isdir(session_path):
            return f"Error: {folder_name} is not a valid directory"

        label_file = join(labels_dir, folder_name, "ses_1", "label.txt")
        if not os.path.exists(label_file):
            return f"Error: No label file found for {folder_name}"

        subject_id = folder_name.replace(".", "_")

        image_files = []
        modality_mapping = {}

        for file in os.listdir(session_path):
            if not file.endswith(".nii.gz"):
                continue

            if "dwi" in file:
                modality_index = 0  # DWI
            elif "flair" in file:
                modality_index = 1  # T2FLAIR
            elif "adc" in file:
                modality_index = 2  # ADC
            elif "swi" in file or "t2s" in file:
                modality_index = 3  # SWI_OR_T2STAR
            else:
                continue

            source_img = join(session_path, file)
            image_files.append(source_img)
            modality_mapping[modality_index] = source_img

        if len(image_files) < len(modalities):
            return f"Error: Not all modalities found for {folder_name}"

        images = [
            nib.load(modality_mapping[i])
            for i in range(len(modalities))
            if i in modality_mapping
        ]

        preprocess_config = build_pretrain_style_preprocess_config(
            num_modalities=len(pp_config["modalities"]),
            norm_op=pp_config["norm_op"],
        )
        preprocessed_images, _ = preprocess_case_for_training_without_label(
            images=images, **preprocess_config
        )

        save_path = join(target_preprocessed, f"{prefix}_{subject_id}")
        np.save(save_path + ".npy", preprocessed_images)
        shutil.copy(label_file, join(target_preprocessed, f"{prefix}_{subject_id}.txt"))

        return f"Processed {folder_name}"

    except Exception as e:
        return f"Error processing {folder_name}: {str(e)}"


def convert_and_preprocess_task1(
    source_path: str,
    output_path: str,
    num_workers=None,
):
    """
    Preprocess all subjects for Task 1 in parallel.

    Args:
        source_path: Path to the source data directory
        output_path: Path where preprocessed data will be saved
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    pp_config = task1_config
    task_name = pp_config["task_name"]
    prefix = "FOMO1"

    labels_dir = join(source_path, "labels")
    images_dir = join(source_path, "preprocessed")

    target_preprocessed = join(output_path, task_name)

    ensure_dir_exists(target_preprocessed)

    folder_names = [
        f for f in os.listdir(images_dir) if os.path.isdir(join(images_dir, f, "ses_1"))
    ]

    assert len(folder_names) > 0, "Did not collect any subjects to preprocess."

    tasks = [
        (folder_name, source_path, labels_dir, pp_config, target_preprocessed, prefix)
        for folder_name in folder_names
    ]

    parallel_process(
        process_subject, tasks, num_workers, desc="Processing subjects for Task 1"
    )

    print(f"Task 1 preprocessing completed. Data saved to {target_preprocessed}")
