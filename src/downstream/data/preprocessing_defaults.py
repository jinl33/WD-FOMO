"""Shared preprocessing constants used by pretraining-style dataset exports."""


OFFLINE_EXPORT_NORM_OP = "volume_wise_znorm"
RUNTIME_PERCENTILE_SCALE = "percentile_1_99_to_neg1_pos1"


PRETRAIN_STYLE_PREPROCESS_CONFIG = {
    "crop_to_nonzero": True,
    "target_orientation": "RAS",
    "target_spacing": [1.0, 1.0, 1.0],
    "keep_aspect_ratio_when_using_target_size": False,
    "transpose": [0, 1, 2],
}


def repeated_normalization(norm_op, num_modalities):
    return [norm_op] * int(num_modalities)


def build_pretrain_style_preprocess_config(
    num_modalities,
    norm_op=OFFLINE_EXPORT_NORM_OP,
    allow_missing_modalities=False,
    **overrides,
):
    config = dict(PRETRAIN_STYLE_PREPROCESS_CONFIG)
    config["normalization_operation"] = repeated_normalization(norm_op, num_modalities)
    config["allow_missing_modalities"] = allow_missing_modalities
    config.update(overrides)
    return config
