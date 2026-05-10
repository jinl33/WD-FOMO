from data.preprocessing_defaults import OFFLINE_EXPORT_NORM_OP


task1_config = {
    "task_name": "Task001_FOMO1",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("DWI", "T2FLAIR", "ADC", "SWI_OR_T2STAR"),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 2,
    "keep_aspect_ratio": False,
    "task_type": "classification",
    "label_extension": ".txt",
    "labels": {0: "Negative", 1: "Positive"},
}

task2_config = {
    "task_name": "Task002_FOMO2",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("DWI", "T2FLAIR", "SWI_OR_T2STAR"),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 2,
    "keep_aspect_ratio": False,
    "task_type": "segmentation",
    "label_extension": ".txt",
    "labels": {0: "background", 1: "meningioma"},
}

task3_config = {
    "task_name": "Task003_FOMO3",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1", "T2"),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 1,
    "keep_aspect_ratio": False,
    "task_type": "regression",
    "label_extension": ".txt",
    "labels": {"regression": "Age"},
}

task5_config = {
    "task_name": "Task_FOMO300K_BrainAge_T1",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1",),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 1,
    "keep_aspect_ratio": False,
    "task_type": "regression",
    "label_extension": ".txt",
    "labels": {"regression": "Age"},
}

task6_config = {
    "task_name": "Task_FOMO300K_BrainAge_T1_FT200",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1",),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 1,
    "keep_aspect_ratio": False,
    "task_type": "regression",
    "label_extension": ".txt",
    "labels": {"regression": "Age"},
}

task7_config = {
    "task_name": "Task_FOMO300K_BrainAge_T1_v2",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1",),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 1,
    "keep_aspect_ratio": False,
    "task_type": "regression",
    "label_extension": ".txt",
    "labels": {"regression": "Age"},
}

task8_config = {
    "task_name": "Task008_OpenNeuro_ds004199_FCD_T2FLAIR",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T2", "T2FLAIR"),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 2,
    "keep_aspect_ratio": False,
    "task_type": "classification",
    "label_extension": ".txt",
    "labels": {0: "Control", 1: "Focal cortical dysplasia"},
}

task9_config = {
    "task_name": "Task009_OpenNeuro_ds004199_FCD_T1FLAIR_1mm",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1", "FLAIR"),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 2,
    "keep_aspect_ratio": False,
    "task_type": "classification",
    "label_extension": ".txt",
    "labels": {0: "Control", 1: "Focal cortical dysplasia"},
}

task10_config = {
    "task_name": "Task010_OpenNeuro_ds004199_FCDSeg_T1FLAIR_1mm",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1", "FLAIR"),
    "norm_op": OFFLINE_EXPORT_NORM_OP,
    "num_classes": 2,
    "keep_aspect_ratio": False,
    "task_type": "segmentation",
    "label_extension": ".txt",
    "labels": {0: "background", 1: "FCD lesion"},
}
