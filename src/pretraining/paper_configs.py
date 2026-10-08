import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet

VOLUME_SHAPE = (192, 256, 192)
WAVELET_SHAPE = (48, 64, 48)
WAVELET_CHANNELS = 64
WAVELET_NORMALIZE = 8.0

NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 2e-2

SCALE_CHANNELS = {
    "paper_232m": [128, 128, 256, 256, 512],
    "s23": [40, 40, 80, 80, 160],
    "tiny": [16, 16, 32, 32, 64],
}

DISTILL_TAPS = ("middle", "up0", "up1")
SEG_SKIP_STAGES = (0, 1, 2, 3)

def unet_config(scale: str) -> dict:
    channels = list(SCALE_CHANNELS[scale])
    return {
        "spatial_dims": 3,
        "in_channels": WAVELET_CHANNELS,
        "out_channels": WAVELET_CHANNELS,
        "num_channels": channels,
        "norm_num_groups": _group_count_all(channels),
        "attention_levels": [False, False, False, True, True],
        "num_head_channels": [0, 0, 0, 32, 32],
        "num_res_blocks": 2,
        "use_flash_attention": False,
        "with_conditioning": False,
    }

def _group_count_all(channels) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if all(c % g == 0 for c in channels):
            return g
    return 1

def build_unet(scale: str) -> nn.Module:
    return DiffusionModelUNet(**unet_config(scale))
