"""
CleanDIFT: Wavelet-Domain Diffusion Feature Extraction for Downstream Tasks.

Architecture:
  1. Per-sample robust percentile normalization to [-1, 1] (matches pretraining)
  2. GPU Haar wavelet transform (2-level, /8.0 normalization): [B,1,D,H,W] -> [B,64,D/4,H/4,W/4]
  3. Concatenate all modality wavelet features: [B, M*64, d, h, w]
  4. Learnable modality fusion: Conv3d(M*64, 64, 1) → [B, 64, d, h, w]
  5. DiffusionModelUNet backbone (MONAI generative):
     - in_channels=64, out_channels=64  ← matches pretrained exactly (440/440 weights)
     - num_channels=[128, 128, 256, 256, 512]
     - Feature hooks on middle_block + up_blocks[0..3]
  6. Task-specific head:
     - Scalar (cls/reg): bottleneck → AdaptiveAvgPool3d → Linear(512, num_classes)
     - Segmentation: multi-scale decoder with skip connections

Pretrained weights: distilled student checkpoint from wavelet-domain diffusion pretraining.
Weight loading: pretrained backbone weights + randomly initialized modality fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.networks.nets import DiffusionModelUNet
from yucca.modules.networks.networks.YuccaNet import YuccaNet

try:
    from gpu_wavelet import HaarDWT3D_2Level

    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False


class CleanDIFTBase(YuccaNet):
    """
    Base class for all CleanDIFT downstream models.

    Wavelet-transforms each modality independently, concatenates along channels,
    then uses a learnable 1x1 fusion conv to project M*64 → 64 channels.
    The backbone uses in_channels=64, matching the pretrained checkpoint EXACTLY,
    so ALL 440 pretrained weights load without any shape mismatch.
    Only the small fusion layer + task head are randomly initialized.

    Args:
        num_modalities:   Number of input imaging modalities.
        normalize_factor: Wavelet normalization divisor (default 8.0).
        backbone_num_channels: Channel progression for DiffusionModelUNet
            (default matches 232 M teacher).
        backbone_num_res_blocks: Residual blocks per level (default 2).
        backbone_attention_levels: Attention flags per level.
        backbone_num_head_channels: Head channels per level.
    """

    _DEFAULT_NUM_CHANNELS = [128, 128, 256, 256, 512]
    _DEFAULT_NUM_RES_BLOCKS = 2
    _DEFAULT_ATTN_LEVELS = [False, False, False, True, True]
    _DEFAULT_HEAD_CHANNELS = [0, 0, 0, 32, 32]

    def __init__(
        self,
        num_modalities,
        normalize_factor=8.0,
        backbone_num_channels=None,
        backbone_num_res_blocks=None,
        backbone_attention_levels=None,
        backbone_num_head_channels=None,
        backbone_norm_num_groups=None,
    ):
        super().__init__()

        if not WAVELET_AVAILABLE:
            raise ImportError(
                "gpu_wavelet module required. CleanDIFT was trained with GPU Haar wavelets "
                "which have a specific subband ordering distinct from pywt."
            )

        self.num_modalities = num_modalities
        self.wavelet = HaarDWT3D_2Level(normalize_factor=normalize_factor)

        if num_modalities > 1:
            self.modality_fusion = nn.Conv3d(num_modalities * 64, 64, kernel_size=1)
        else:
            self.modality_fusion = nn.Identity()
        self.fusion_adapter = None

        _nc = (
            backbone_num_channels
            if backbone_num_channels is not None
            else self._DEFAULT_NUM_CHANNELS
        )
        _nr = (
            backbone_num_res_blocks
            if backbone_num_res_blocks is not None
            else self._DEFAULT_NUM_RES_BLOCKS
        )
        _al = (
            backbone_attention_levels
            if backbone_attention_levels is not None
            else self._DEFAULT_ATTN_LEVELS
        )
        _hc = (
            backbone_num_head_channels
            if backbone_num_head_channels is not None
            else self._DEFAULT_HEAD_CHANNELS
        )

        n_levels = len(_nc)
        if len(_al) != n_levels:
            if n_levels <= 3:
                _al = [False] * (n_levels - 1) + [True]
            else:
                _al = ([False] * (n_levels - 2) + [True, True])[:n_levels]
        if len(_hc) != n_levels:
            if n_levels <= 3:
                _hc = [0] * (n_levels - 1) + [32]
            else:
                _hc = ([0] * (n_levels - 2) + [32, 32])[:n_levels]

        if backbone_norm_num_groups is not None:
            _nng = backbone_norm_num_groups
        else:
            _nng = 32
            for g in (32, 16, 8, 4):
                if all(ch % g == 0 for ch in _nc):
                    _nng = g
                    break

        self.backbone = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=64,
            out_channels=64,
            num_channels=_nc,
            attention_levels=_al,
            num_head_channels=_hc,
            num_res_blocks=_nr,
            norm_num_groups=_nng,
            with_conditioning=False,
        )

        self.backbone_num_channels = list(_nc)
        self.bottleneck_channels = _nc[-1]

        self.features = {}
        self.backbone.middle_block.register_forward_hook(
            lambda m, i, o: self.features.update({"mid": o})
        )
        for idx in range(min(4, len(self.backbone.up_blocks))):

            def make_hook(name):
                return lambda m, i, o: self.features.update({name: o})

            self.backbone.up_blocks[idx].register_forward_hook(make_hook(f"up{idx}"))

    def _normalize_to_neg1_pos1(self, x):
        """
        Per-sample robust percentile scaling to [-1, 1].

        This mirrors the pretraining path in `src/lightweight_fomo_all.py`,
        which uses the 1st and 99th percentiles rather than simple min-max
        scaling before the wavelet transform.
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_flat = x.flatten(1)
        p1 = torch.quantile(x_flat, 0.01, dim=1, keepdim=True).reshape(-1, 1, 1, 1, 1)
        p99 = torch.quantile(x_flat, 0.99, dim=1, keepdim=True).reshape(-1, 1, 1, 1, 1)
        denom = p99 - p1
        scaled = torch.where(
            denom > 1e-6,
            (x - p1) / (denom + 1e-10),
            torch.zeros_like(x),
        )
        scaled = torch.clamp(scaled, 0.0, 1.0)
        return scaled * 2.0 - 1.0

    def extract_features(self, x_all_modalities):
        """
        Wavelet-transform each modality, fuse to 64 channels, run through backbone.

        Args:
            x_all_modalities: [B, M, D, H, W] where M = num_modalities
        Returns:
            dict of feature tensors keyed by 'mid', 'up0', 'up1', 'up2', 'up3'
        """
        B, M, D, H, W = x_all_modalities.shape

        align = 64
        pad_d = (align - D % align) % align
        pad_h = (align - H % align) % align
        pad_w = (align - W % align) % align
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_all_modalities = torch.nn.functional.pad(
                x_all_modalities, (0, pad_w, 0, pad_h, 0, pad_d)
            )
            _, _, D, H, W = x_all_modalities.shape

        x_flat = x_all_modalities.reshape(B * M, 1, D, H, W)
        x_flat = self._normalize_to_neg1_pos1(x_flat)
        x_wave = self.wavelet(x_flat)  # [B*M, 64, d, h, w]

        _, C, d, h, w = x_wave.shape
        x_concat = x_wave.reshape(B, M * C, d, h, w)  # [B, M*64, d, h, w]

        x_fused = self.modality_fusion(x_concat)  # [B, 64, d, h, w]
        if self.fusion_adapter is not None:
            x_fused = x_fused + self.fusion_adapter(x_fused)

        t = torch.zeros(B, device=x_fused.device).long()
        _ = self.backbone(x_fused, t)

        feats = {k: v for k, v in self.features.items()}
        self.features.clear()
        return feats


class CleanDIFTScalarWrapper(CleanDIFTBase):
    """
    For classification (Task 1) and regression (Task 3).
    Uses bottleneck features → global avg pool → Linear.
    Matches the proven v17 architecture.
    """

    def __init__(self, input_channels, num_classes, **kwargs):
        backbone_cfg = kwargs  # passes backbone_num_channels etc. if provided
        super().__init__(num_modalities=input_channels, **backbone_cfg)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(self.bottleneck_channels, num_classes)

    def forward(self, x):
        feats = self.extract_features(x)
        feat_mid = feats["mid"]  # [B, 512, d, h, w]
        pooled = self.global_pool(feat_mid).flatten(1)  # [B, 512]
        return self.head(pooled)


class CleanDIFTScalarMultiTapWrapper(CleanDIFTBase):
    """
    Scalar probe that taps multiple decoder depths instead of only the bottleneck.
    This keeps the pretrained backbone untouched while giving the regression head
    access to both global and slightly finer-scale context.
    """

    def __init__(self, input_channels, num_classes, **kwargs):
        backbone_cfg = kwargs
        super().__init__(num_modalities=input_channels, **backbone_cfg)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.tap_names = ("mid", "up0", "up1")
        self.tap_dims = [
            self.backbone_num_channels[-1],
            self.backbone_num_channels[-1],
            self.backbone_num_channels[-2],
        ]
        fused_dim = sum(self.tap_dims)
        hidden_dim = max(self.bottleneck_channels, fused_dim // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        feats = self.extract_features(x)
        missing = [name for name in self.tap_names if name not in feats]
        if missing:
            raise RuntimeError(f"Missing CleanDIFT feature taps: {missing}")
        pooled = [self.global_pool(feats[name]).flatten(1) for name in self.tap_names]
        fused = torch.cat(pooled, dim=1)
        return self.head(fused)


class ResidualAdapter3D(nn.Module):
    """
    Lightweight bottleneck adapter used for PEFT-style CleanDIFT finetuning.
    Starts as an exact identity because the up-projection is zero-initialized.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.down = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.up = nn.Conv3d(hidden, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class CleanDIFTScalarPEFTTapWrapper(CleanDIFTBase):
    """
    PEFT-style scalar wrapper:
      - freezes the pretrained diffusion backbone during finetuning
      - adapts with small residual adapters on fused wavelet features and
        selected backbone feature taps
      - uses a lightweight gated multi-tap fusion instead of a large MLP head
    """

    def __init__(self, input_channels, num_classes, **kwargs):
        backbone_cfg = kwargs
        super().__init__(num_modalities=input_channels, **backbone_cfg)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.tap_names = ("mid", "up0", "up1")
        self.tap_dims = {
            "mid": self.backbone_num_channels[-1],
            "up0": self.backbone_num_channels[-1],
            "up1": self.backbone_num_channels[-2],
        }

        self.fusion_adapter = ResidualAdapter3D(64, reduction=16)
        self.tap_adapters = nn.ModuleDict(
            {
                name: ResidualAdapter3D(ch, reduction=8)
                for name, ch in self.tap_dims.items()
            }
        )
        self.tap_projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(ch),
                    nn.Linear(ch, self.bottleneck_channels),
                    nn.GELU(),
                )
                for name, ch in self.tap_dims.items()
            }
        )
        self.tap_logits = nn.Parameter(torch.zeros(len(self.tap_names)))
        self.fused_norm = nn.LayerNorm(self.bottleneck_channels)
        self.head = nn.Linear(self.bottleneck_channels, num_classes)

    def forward(self, x):
        feats = self.extract_features(x)
        pooled = []
        for name in self.tap_names:
            feat = feats[name] + self.tap_adapters[name](feats[name])
            pooled.append(self.tap_projections[name](self.global_pool(feat).flatten(1)))

        weights = torch.softmax(self.tap_logits, dim=0)
        fused = torch.stack(pooled, dim=0)  # [T, B, C]
        fused = (weights[:, None, None] * fused).sum(dim=0)
        fused = self.fused_norm(fused)
        return self.head(fused)


class DecoderBlock(nn.Module):
    """Trainable upsampling block with skip connection fusion."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return self.conv(x)


class CleanDIFTSegWrapper(CleanDIFTBase):
    """
    Multi-scale decoder probe on backbone features for segmentation.
    Backbone uses in_channels=64 (matching pretrained), with modality fusion before it.
    Decoder uses multi-scale skip connections from backbone feature hooks.

    Channel dims are derived from backbone_num_channels so this works correctly
    for both the 232 M teacher ([128,128,256,256,512]) and the medium student
    ([32,64,128,192,256]).  Requires a 5-level backbone (cleandift_tiny unsupported).
    """

    def __init__(self, input_channels, num_classes, **kwargs):
        backbone_kw = {k: v for k, v in kwargs.items() if k.startswith("backbone_")}
        super().__init__(num_modalities=input_channels, **backbone_kw)
        self.num_classes = num_classes

        _nc = backbone_kw.get("backbone_num_channels", self._DEFAULT_NUM_CHANNELS)
        assert len(_nc) >= 5, (
            f"CleanDIFTSegWrapper requires ≥5 backbone levels, got {len(_nc)}. "
            "Use CleanDIFT-Tiny only for Tasks 1/3 (scalar head)."
        )
        dim_mid = _nc[-1]  # 256 (medium) / 512 (teacher)
        dim_up0 = _nc[
            -1
        ]  # 256 (medium) / 512 (teacher)  — same as mid (coarsest up stage)
        dim_up1 = _nc[-2]  # 192 (medium) / 256 (teacher)
        dim_up2 = _nc[-3]  # 128 (medium) / 256 (teacher)
        dim_up3 = _nc[-4]  #  64 (medium) / 128 (teacher)

        bottleneck_out = min(256, dim_mid)
        gn_groups = min(16, bottleneck_out // 8)  # ≥8 ch per group
        self.decoder = nn.ModuleDict(
            {
                "bottleneck": nn.Sequential(
                    nn.Conv3d(dim_mid, bottleneck_out, kernel_size=3, padding=1),
                    nn.GroupNorm(gn_groups, bottleneck_out),
                    nn.ReLU(),
                ),
                "up_block1": DecoderBlock(bottleneck_out, dim_up0, 128),
                "up_block2": DecoderBlock(128, dim_up1, 64),
                "up_block3": DecoderBlock(64, dim_up2, 32),
                "up_block4": DecoderBlock(32, dim_up3, 16),
                "final": nn.Conv3d(16, num_classes, kernel_size=1),
            }
        )

    def forward(self, x):
        B, M, D_orig, H_orig, W_orig = x.shape
        feats = self.extract_features(x)

        dec = self.decoder["bottleneck"](feats["mid"])
        dec = self.decoder["up_block1"](dec, feats["up0"])
        dec = self.decoder["up_block2"](dec, feats["up1"])
        dec = self.decoder["up_block3"](dec, feats["up2"])
        dec = self.decoder["up_block4"](dec, feats["up3"])
        logits = self.decoder["final"](dec)

        return F.interpolate(
            logits, size=(D_orig, H_orig, W_orig), mode="trilinear", align_corners=False
        )


class CleanDIFTRegWrapper(CleanDIFTScalarWrapper):
    """Regression wrapper — identical architecture to CleanDIFTScalarWrapper."""

    pass


class CleanDIFTRegMultiTapWrapper(CleanDIFTScalarMultiTapWrapper):
    """Regression wrapper with a multi-tap scalar head."""

    pass


class CleanDIFTRegPEFTTapWrapper(CleanDIFTScalarPEFTTapWrapper):
    """Regression wrapper with lightweight PEFT adapters and gated tap fusion."""

    pass


def _cleandift_factory(
    mode,
    input_channels,
    num_classes,
    backbone_num_channels=None,
    backbone_num_res_blocks=None,
):
    """Internal helper shared by all CleanDIFT factory variants."""
    bkw = {}
    if backbone_num_channels is not None:
        bkw["backbone_num_channels"] = backbone_num_channels
    if backbone_num_res_blocks is not None:
        bkw["backbone_num_res_blocks"] = backbone_num_res_blocks

    if mode == "classification":
        return CleanDIFTScalarWrapper(
            input_channels=input_channels, num_classes=num_classes, **bkw
        )
    elif mode == "segmentation":
        return CleanDIFTSegWrapper(
            input_channels=input_channels, num_classes=num_classes, **bkw
        )
    elif mode == "regression":
        return CleanDIFTRegWrapper(
            input_channels=input_channels, num_classes=num_classes, **bkw
        )
    else:
        raise ValueError(f"Unsupported mode for CleanDIFT: {mode}")


def _cleandift_multitap_factory(
    mode,
    input_channels,
    num_classes,
    backbone_num_channels=None,
    backbone_num_res_blocks=None,
):
    """Internal helper for multitap scalar variants."""
    bkw = {}
    if backbone_num_channels is not None:
        bkw["backbone_num_channels"] = backbone_num_channels
    if backbone_num_res_blocks is not None:
        bkw["backbone_num_res_blocks"] = backbone_num_res_blocks

    if mode == "classification":
        return CleanDIFTScalarMultiTapWrapper(
            input_channels=input_channels,
            num_classes=num_classes,
            **bkw,
        )
    elif mode == "regression":
        return CleanDIFTRegMultiTapWrapper(
            input_channels=input_channels,
            num_classes=num_classes,
            **bkw,
        )
    else:
        raise ValueError(f"Unsupported mode for CleanDIFT multitap: {mode}")


def cleandift(mode, input_channels, output_channels=None, num_classes=None, **kwargs):
    """
    Factory function that creates the appropriate CleanDIFT wrapper based on mode.
    Called by BaseSupervisedModel.load_model() via getattr(networks, model_name).

    Original 232 M teacher architecture: num_channels=[128,128,256,256,512].

    Args:
        mode: 'classification', 'segmentation', or 'regression'
        input_channels: Number of input modalities
        output_channels: Number of output channels (alias for num_classes)
        num_classes: Number of output classes
    """
    nc = num_classes or output_channels or 1
    return _cleandift_factory(mode, input_channels, nc)


def cleandift_medium(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    """
    CleanDIFT-Medium — distilled student matching FOMO UNet-XL parameter count (~56 M).

    Backbone: num_channels=[32,64,128,192,256], num_res_blocks=2  → 57.85 M params.
    Verified with DiffusionModelUNet(spatial_dims=3, in_channels=64, out_channels=64).
    Previous (wrong) config [64,128,256,512,512] r2 produced 279 M — larger than teacher.
    Pretrained via knowledge distillation from the 232 M teacher using the
    held-out 5 000-file distillation split (fomo60k_split.json).
    """
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[32, 64, 128, 192, 256],
        backbone_num_res_blocks=2,
    )


def cleandift_medium_backbone(num_modalities=3):
    """
    Return a bare CleanDIFTBase (medium config) for use as a frozen feature extractor.

    Used by eval_task2.py autoresearch harness — the backbone is kept frozen and a
    separate Task2Head (in task2_head.py) is trained on top of its hook outputs.

    The caller is responsible for:
      1. Loading pretrained weights into backbone.backbone (the DiffusionModelUNet)
         via ``backbone.load_state_dict({"backbone." + k: v ...}, strict=False)``.
      2. Calling ``backbone.eval()`` and freezing all parameters.

    Returns:
        CleanDIFTBase with num_channels=[32,64,128,192,256], num_res_blocks=2.
        Hook outputs: {"mid": (B,256,...), "up0": (B,256,...), "up1": (B,192,...),
                       "up2": (B,128,...), "up3": (B,64,...)}
    """
    return CleanDIFTBase(
        num_modalities=num_modalities,
        backbone_num_channels=[32, 64, 128, 192, 256],
        backbone_num_res_blocks=2,
    )


def cleandift_s15(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    """CleanDIFT-S15 — scratch-pretrained on FOMO60k, 12.7M backbone params."""
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[32, 32, 64, 64, 128],
        backbone_num_res_blocks=2,
    )


def cleandift_s23(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    """CleanDIFT-S23 — scratch-pretrained on FOMO60k, 19.8M backbone params."""
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[40, 40, 80, 80, 160],
        backbone_num_res_blocks=2,
    )


def cleandift_s23_multitap(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    """CleanDIFT-S23 with a multi-tap scalar head for Tasks 1/3."""
    nc = num_classes or output_channels or 1
    return _cleandift_multitap_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[40, 40, 80, 80, 160],
        backbone_num_res_blocks=2,
    )


def cleandift_s23_pefttap(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    """CleanDIFT-S23 with lightweight PEFT adapters and gated tap fusion."""
    nc = num_classes or output_channels or 1
    if mode == "classification":
        return CleanDIFTScalarPEFTTapWrapper(
            input_channels=input_channels,
            num_classes=nc,
            backbone_num_channels=[40, 40, 80, 80, 160],
            backbone_num_res_blocks=2,
        )
    elif mode == "regression":
        return CleanDIFTRegPEFTTapWrapper(
            input_channels=input_channels,
            num_classes=nc,
            backbone_num_channels=[40, 40, 80, 80, 160],
            backbone_num_res_blocks=2,
        )
    raise ValueError(f"Unsupported mode for CleanDIFT PEFTTap: {mode}")


def cleandift_s33(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    """CleanDIFT-S33 — scratch-pretrained on FOMO60k, 28.5M backbone params."""
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[48, 48, 96, 96, 192],
        backbone_num_res_blocks=2,
    )


def cleandift_s33_pefttap(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    """CleanDIFT-S33 with lightweight PEFT adapters and gated tap fusion."""
    nc = num_classes or output_channels or 1
    if mode == "classification":
        return CleanDIFTScalarPEFTTapWrapper(
            input_channels=input_channels,
            num_classes=nc,
            backbone_num_channels=[48, 48, 96, 96, 192],
            backbone_num_res_blocks=2,
        )
    elif mode == "regression":
        return CleanDIFTRegPEFTTapWrapper(
            input_channels=input_channels,
            num_classes=nc,
            backbone_num_channels=[48, 48, 96, 96, 192],
            backbone_num_res_blocks=2,
        )
    raise ValueError(f"Unsupported mode for CleanDIFT PEFTTap: {mode}")


def cleandift_tiny(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    """
    CleanDIFT-Tiny — distilled student matching FOMO UNet-B parameter count (~14 M).

    Backbone: num_channels=[32,64,128], num_res_blocks=2  → 10.53 M params (3-level UNet).
    Verified with DiffusionModelUNet(spatial_dims=3, in_channels=64, out_channels=64).
    Previous (wrong) config [32,64,128,256,512] r1 produced 115 M — 8x the target.
    NOTE: 3-level architecture only populates feats['mid','up0','up1'].
          CleanDIFTSegWrapper (which needs up2/up3) is NOT supported for Tiny.
          Tasks 1 & 3 (scalar head via CleanDIFTScalarWrapper) are fully supported.
    Pretrained via knowledge distillation from the 232 M teacher using the
    held-out 5 000-file distillation split (fomo60k_split.json).
    """
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[32, 64, 128],
        backbone_num_res_blocks=2,
    )
