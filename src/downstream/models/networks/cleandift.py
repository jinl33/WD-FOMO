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

try:
    from gpu_wavelet import HaarIDWT3D_2Level
except ImportError:
    HaarIDWT3D_2Level = None

def _group_count(channels: int, max_groups: int = 16) -> int:
    for groups in (max_groups, 8, 4, 2, 1):
        if groups <= channels and channels % groups == 0:
            return groups
    return 1

class CleanDIFTBase(YuccaNet):

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

    def _normalize_to_neg1_pos1(self, x):
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
        x_wave = self._wavelet_modalities(x_all_modalities)
        B, M, C, d, h, w = x_wave.shape
        x_concat = x_wave.reshape(B, M * C, d, h, w)
        x_fused = self.modality_fusion(x_concat)
        if self.fusion_adapter is not None:
            x_fused = x_fused + self.fusion_adapter(x_fused)
        return self._run_backbone(x_fused)

    def extract_features_per_modality(self, x_all_modalities):
        x_wave = self._wavelet_modalities(x_all_modalities)
        return [self._run_backbone(x_wave[:, idx]) for idx in range(x_wave.shape[1])]

    def _wavelet_modalities(self, x_all_modalities):
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
        x_wave = self.wavelet(x_flat)
        _, C, d, h, w = x_wave.shape
        return x_wave.reshape(B, M, C, d, h, w)

    def _run_backbone(self, x_features):
        B = x_features.shape[0]
        t = torch.zeros(B, device=x_features.device).long()
        local_feats = {}
        handles = [
            self.backbone.middle_block.register_forward_hook(
                lambda m, i, o: local_feats.update({"mid": o})
            )
        ]

        def make_hook(name):
            return lambda m, i, o: local_feats.update({name: o})

        for idx in range(min(4, len(self.backbone.up_blocks))):
            handles.append(
                self.backbone.up_blocks[idx].register_forward_hook(make_hook(f"up{idx}"))
            )
        if getattr(self, "pretrained_decoder", False):
            handles.append(
                self.backbone.up_blocks[-1].register_forward_hook(make_hook("up_last"))
            )
        try:
            _ = self.backbone(x_features, t)
        finally:
            for handle in handles:
                handle.remove()
        return local_feats

class CleanDIFTScalarWrapper(CleanDIFTBase):

    def __init__(self, input_channels, num_classes, **kwargs):
        backbone_cfg = kwargs
        super().__init__(num_modalities=input_channels, **backbone_cfg)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(self.bottleneck_channels, num_classes)

    def forward(self, x):
        feats = self.extract_features(x)
        feat_mid = feats["mid"]
        pooled = self.global_pool(feat_mid).flatten(1)
        return self.head(pooled)

class CleanDIFTScalarMultiTapWrapper(CleanDIFTBase):

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
        fused = torch.stack(pooled, dim=0)
        fused = (weights[:, None, None] * fused).sum(dim=0)
        fused = self.fused_norm(fused)
        return self.head(fused)

class DecoderBlock(nn.Module):

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

class SEBlock3d(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        mid = max(1, channels // ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3, 4))
        s = self.fc(s).view(*s.shape, 1, 1, 1)
        return x * s

class MultiScaleRefineHead3d(nn.Module):

    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        fine_channels: int,
        coarse_channels: int,
        hidden_channels: int = 24,
    ):
        super().__init__()
        proj_channels = max(8, hidden_channels // 2)
        self.fine_proj = nn.Sequential(
            nn.Conv3d(fine_channels, proj_channels, kernel_size=1),
            nn.GroupNorm(_group_count(proj_channels), proj_channels),
            nn.ReLU(inplace=True),
        )
        self.coarse_proj = nn.Sequential(
            nn.Conv3d(coarse_channels, proj_channels, kernel_size=1),
            nn.GroupNorm(_group_count(proj_channels), proj_channels),
            nn.ReLU(inplace=True),
        )
        refine_in = num_classes + input_channels + proj_channels + proj_channels
        self.refine = nn.Sequential(
            nn.Conv3d(refine_in, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.delta = nn.Conv3d(hidden_channels, num_classes, kernel_size=1)
        self.boundary = nn.Conv3d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.delta.weight)
        if self.delta.bias is not None:
            nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.boundary.weight)
        if self.boundary.bias is not None:
            nn.init.zeros_(self.boundary.bias)

    def forward(
        self,
        logits: torch.Tensor,
        image: torch.Tensor,
        fine_feat: torch.Tensor,
        coarse_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_size = logits.shape[2:]
        target_size = fine_feat.shape[2:]
        fine = fine_feat
        coarse = coarse_feat
        logits_small = logits
        image_small = image
        if logits_small.shape[2:] != target_size:
            logits_small = F.interpolate(
                logits_small, size=target_size, mode="trilinear", align_corners=False
            )
        if image_small.shape[2:] != target_size:
            image_small = F.interpolate(
                image_small, size=target_size, mode="trilinear", align_corners=False
            )
        if fine.shape[2:] != target_size:
            fine = F.interpolate(
                fine, size=target_size, mode="trilinear", align_corners=False
            )
        if coarse.shape[2:] != target_size:
            coarse = F.interpolate(
                coarse, size=target_size, mode="trilinear", align_corners=False
            )
        hidden = self.refine(
            torch.cat(
                [
                    logits_small,
                    image_small,
                    self.fine_proj(fine),
                    self.coarse_proj(coarse),
                ],
                dim=1,
            )
        )
        delta = self.delta(hidden)
        boundary = self.boundary(hidden)
        if delta.shape[2:] != full_size:
            delta = F.interpolate(
                delta, size=full_size, mode="trilinear", align_corners=False
            )
        if boundary.shape[2:] != full_size:
            boundary = F.interpolate(
                boundary, size=full_size, mode="trilinear", align_corners=False
            )
        return logits + delta, boundary, hidden

class CleanDIFTSegWrapper(CleanDIFTBase):

    _PTDEC_SUFFIX = "_ptdec"
    _PTDEC_BASE_VARIANTS = ("earlyfusion", "image_refine", "waveletavg", "waveletavg_refine")

    def __init__(
        self,
        input_channels,
        num_classes,
        seg_attention: str = "none",
        seg_head_variant: str = "earlyfusion",
        seg_output_fg_prior=None,
        **kwargs,
    ):
        backbone_kw = {k: v for k, v in kwargs.items() if k.startswith("backbone_")}
        super().__init__(num_modalities=input_channels, **backbone_kw)
        full_seg_head_variant = seg_head_variant
        self.pretrained_decoder = seg_head_variant.endswith(self._PTDEC_SUFFIX)
        if self.pretrained_decoder:
            seg_head_variant = seg_head_variant[: -len(self._PTDEC_SUFFIX)]
            if seg_head_variant not in self._PTDEC_BASE_VARIANTS:
                raise ValueError(
                    f"{full_seg_head_variant}: '_ptdec' supports base variants "
                    f"{self._PTDEC_BASE_VARIANTS}, got '{seg_head_variant}'"
                )
            if seg_attention != "none":
                raise ValueError(
                    "'_ptdec' variants have no tap-fed decoder for SE gates to act on; "
                    f"use seg_attention='none' (got '{seg_attention}')"
                )
            if seg_output_fg_prior is not None:
                raise ValueError("seg_output_fg_prior is not supported with '_ptdec' variants")
            if HaarIDWT3D_2Level is None:
                raise ImportError("'_ptdec' variants need gpu_wavelet.HaarIDWT3D_2Level")
        self.num_classes = num_classes
        self.seg_attention = seg_attention
        self.seg_head_variant = seg_head_variant
        self._last_aux_outputs = {}

        _nc = backbone_kw.get("backbone_num_channels", self._DEFAULT_NUM_CHANNELS)
        assert len(_nc) >= 5, (
            f"CleanDIFTSegWrapper requires ≥5 backbone levels, got {len(_nc)}. "
            "Use CleanDIFT-Tiny only for Tasks 1/3 (scalar head)."
        )
        dim_mid = _nc[-1]
        dim_up0 = _nc[-1]
        dim_up1 = _nc[-2]
        dim_up2 = _nc[-3]
        dim_up3 = _nc[-4]

        bottleneck_out = min(256, dim_mid)
        if self.pretrained_decoder:
            self.decoder = None
        else:
            self.decoder = nn.ModuleDict(
                {
                    "bottleneck": nn.Sequential(
                        nn.Conv3d(dim_mid, bottleneck_out, kernel_size=3, padding=1),
                        nn.GroupNorm(_group_count(bottleneck_out), bottleneck_out),
                        nn.ReLU(),
                    ),
                    "up_block1": DecoderBlock(bottleneck_out, dim_up0, 128),
                    "up_block2": DecoderBlock(128, dim_up1, 64),
                    "up_block3": DecoderBlock(64, dim_up2, 32),
                    "up_block4": DecoderBlock(32, dim_up3, 16),
                    "final": nn.Conv3d(16, num_classes, kernel_size=1),
                }
            )
        if self.seg_attention == "se":
            self.seg_gates = nn.ModuleDict(
                {
                    "mid": SEBlock3d(dim_mid),
                    "up0": SEBlock3d(dim_up0),
                    "up1": SEBlock3d(dim_up1),
                    "up2": SEBlock3d(dim_up2),
                    "up3": SEBlock3d(dim_up3),
                }
            )
        else:
            self.seg_gates = None

        if self.seg_head_variant == "latefusion_logits":
            self.logit_fusion = nn.Conv3d(
                self.num_modalities * num_classes, num_classes, kernel_size=1
            )
            self._init_logit_fusion()
        else:
            self.logit_fusion = None

        if self.seg_head_variant == "flairskip_latefusion":
            self.mid_fuse = self._make_fusion_block(dim_mid)
            self.up0_fuse = self._make_fusion_block(dim_up0)
            self.flair_mid_residual = nn.Conv3d(dim_mid, dim_mid, kernel_size=1)
            nn.init.zeros_(self.flair_mid_residual.weight)
            if self.flair_mid_residual.bias is not None:
                nn.init.zeros_(self.flair_mid_residual.bias)
        else:
            self.mid_fuse = None
            self.up0_fuse = None
            self.flair_mid_residual = None

        if self.seg_head_variant in {
            "image_refine",
            "image_refine_v1",
            "waveletavg_refine",
            "waveletweighted_refine",
        }:
            refine_channels = 16
            self.image_refine = nn.Sequential(
                nn.Conv3d(num_classes + input_channels, refine_channels, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(refine_channels), refine_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(refine_channels, refine_channels, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(refine_channels), refine_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(refine_channels, num_classes, kernel_size=1),
            )
            nn.init.zeros_(self.image_refine[-1].weight)
            if self.image_refine[-1].bias is not None:
                nn.init.zeros_(self.image_refine[-1].bias)
        else:
            self.image_refine = None

        if self.seg_head_variant in {"waveletweighted", "waveletweighted_refine"}:
            weight_hidden = 16
            self.wavelet_modal_weight = nn.Sequential(
                nn.Conv3d(64, weight_hidden, kernel_size=1),
                nn.GroupNorm(_group_count(weight_hidden), weight_hidden),
                nn.ReLU(inplace=True),
                nn.Conv3d(weight_hidden, 1, kernel_size=1),
            )
            nn.init.zeros_(self.wavelet_modal_weight[-1].weight)
            if self.wavelet_modal_weight[-1].bias is not None:
                nn.init.zeros_(self.wavelet_modal_weight[-1].bias)
        else:
            self.wavelet_modal_weight = None

        if self.seg_head_variant in {"image_refine_ms", "image_refine_ms_aux"}:
            self.image_refine_ms = MultiScaleRefineHead3d(
                num_classes=num_classes,
                input_channels=input_channels,
                fine_channels=dim_up2,
                coarse_channels=dim_up1,
                hidden_channels=16,
            )
        else:
            self.image_refine_ms = None

        if self.seg_head_variant == "cascade_refine":
            cascade_proj_ch = min(32, dim_up1, dim_up0)
            self.cascade_proj_up0 = nn.Conv3d(dim_up0, cascade_proj_ch, kernel_size=1)
            self.cascade_proj_up1 = nn.Conv3d(dim_up1, cascade_proj_ch, kernel_size=1)

            cascade_in_channels = num_classes + input_channels + cascade_proj_ch + cascade_proj_ch
            cascade_channels = 24
            self.cascade_refine = nn.Sequential(
                nn.Conv3d(cascade_in_channels, cascade_channels, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(cascade_channels), cascade_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(cascade_channels, cascade_channels, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(cascade_channels), cascade_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(cascade_channels, num_classes, kernel_size=1),
            )
            nn.init.zeros_(self.cascade_refine[-1].weight)
            if self.cascade_refine[-1].bias is not None:
                nn.init.zeros_(self.cascade_refine[-1].bias)
        else:
            self.cascade_refine = None

        if seg_output_fg_prior is not None:
            self._init_output_bias(float(seg_output_fg_prior))

        if self.pretrained_decoder:
            out_norm, out_act = self.backbone.out[0], self.backbone.out[1]
            if not (isinstance(out_norm, nn.GroupNorm) and isinstance(out_act, nn.SiLU)):
                raise RuntimeError(
                    "'_ptdec' expects backbone.out = (GroupNorm, SiLU, conv); got "
                    f"{type(out_norm).__name__}, {type(out_act).__name__}"
                )
            self.ptdec_out = nn.Conv3d(
                out_norm.num_channels, num_classes * 64, kernel_size=3, padding=1
            )
            nn.init.zeros_(self.ptdec_out.weight)
            nn.init.zeros_(self.ptdec_out.bias)
            self.ptdec_idwt = HaarIDWT3D_2Level(
                denormalize_factor=self.wavelet.normalize_factor
            )
        else:
            self.ptdec_out = None
            self.ptdec_idwt = None
        self._base_seg_head_variant = self.seg_head_variant
        self.seg_head_variant = full_seg_head_variant

    def _make_fusion_block(self, channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )

    def _init_logit_fusion(self):
        nn.init.zeros_(self.logit_fusion.weight)
        nn.init.zeros_(self.logit_fusion.bias)
        with torch.no_grad():
            for cls_idx in range(self.num_classes):
                for mod_idx in range(self.num_modalities):
                    src = mod_idx * self.num_classes + cls_idx
                    self.logit_fusion.weight[cls_idx, src, 0, 0, 0] = (
                        1.0 / self.num_modalities
                    )

    def _init_output_bias(self, fg_prior: float):
        if self.num_classes != 2:
            raise ValueError("seg_output_fg_prior currently supports binary segmentation")
        if not 0.0 < fg_prior < 1.0:
            raise ValueError("seg_output_fg_prior must be in (0, 1)")
        with torch.no_grad():
            final = self.decoder["final"]
            final.bias.zero_()
            final.bias[1] = torch.logit(torch.tensor(fg_prior)).item()

    def _apply_seg_gates(self, feats):
        if self.seg_gates is None:
            return feats
        return {
            name: self.seg_gates[name](feat) if name in self.seg_gates else feat
            for name, feat in feats.items()
        }

    def _decode_logits(self, mid, up0, up1, up2, up3):
        dec = self.decoder["bottleneck"](mid)
        dec = self.decoder["up_block1"](dec, up0)
        dec = self.decoder["up_block2"](dec, up1)
        dec = self.decoder["up_block3"](dec, up2)
        dec = self.decoder["up_block4"](dec, up3)
        return self.decoder["final"](dec)

    def _decode_logits_and_feature(self, mid, up0, up1, up2, up3):
        dec = self.decoder["bottleneck"](mid)
        dec = self.decoder["up_block1"](dec, up0)
        dec = self.decoder["up_block2"](dec, up1)
        dec = self.decoder["up_block3"](dec, up2)
        dec = self.decoder["up_block4"](dec, up3)
        return self.decoder["final"](dec), dec

    def _forward_earlyfusion(self, x):
        feats = self._apply_seg_gates(self.extract_features(x))
        return self._decode_logits(
            feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
        )

    def _forward_earlyfusion_with_feats(self, x):
        feats = self._apply_seg_gates(self.extract_features(x))
        logits = self._decode_logits(
            feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
        )
        return logits, feats

    def _forward_earlyfusion_with_decoder_feature(self, x):
        feats = self._apply_seg_gates(self.extract_features(x))
        return self._decode_logits_and_feature(
            feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
        )

    def _forward_waveletavg(self, x):
        x_wave = self._wavelet_modalities(x)
        x_fused = x_wave[:, 0] if x_wave.shape[1] == 1 else x_wave.mean(dim=1)
        feats = self._apply_seg_gates(self._run_backbone(x_fused))
        return self._decode_logits(
            feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
        )

    def _forward_waveletavg_with_feats(self, x):
        x_wave = self._wavelet_modalities(x)
        x_fused = x_wave[:, 0] if x_wave.shape[1] == 1 else x_wave.mean(dim=1)
        feats = self._apply_seg_gates(self._run_backbone(x_fused))
        logits = self._decode_logits(
            feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
        )
        return logits, feats

    def _forward_waveletweighted(self, x):
        x_wave = self._wavelet_modalities(x)
        if x_wave.shape[1] == 1:
            x_fused = x_wave[:, 0]
            weights = torch.ones(
                (x_wave.shape[0], 1, 1, 1, 1, 1),
                device=x_wave.device,
                dtype=x_wave.dtype,
            )
        else:
            pooled = x_wave.mean(dim=(3, 4, 5), keepdim=True)
            flat = pooled.reshape(-1, pooled.shape[2], 1, 1, 1)
            logits = self.wavelet_modal_weight(flat)
            logits = logits.reshape(x_wave.shape[0], x_wave.shape[1], 1, 1, 1, 1)
            weights = torch.softmax(logits, dim=1)
            x_fused = (x_wave * weights).sum(dim=1)
        self._last_aux_outputs["wavelet_modality_weights"] = weights.detach()
        feats = self._apply_seg_gates(self._run_backbone(x_fused))
        return self._decode_logits(
            feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
        )

    def _forward_cascade_refine(self, x):
        feats = self._apply_seg_gates(self.extract_features(x))
        logits = self._decode_logits(
            feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
        )

        target_size = feats["up0"].shape[2:]

        logits_small = F.interpolate(
            logits, size=target_size, mode="trilinear", align_corners=False
        )
        up0_small = feats["up0"]
        up1_small = feats["up1"]
        if up1_small.shape[2:] != target_size:
            up1_small = F.interpolate(up1_small, size=target_size, mode="trilinear", align_corners=False)

        x_small = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)

        if hasattr(self, "cascade_proj_up0"):
            up0_small = self.cascade_proj_up0(up0_small)
        if hasattr(self, "cascade_proj_up1"):
            up1_small = self.cascade_proj_up1(up1_small)

        refine_in = torch.cat([logits_small, x_small, up0_small, up1_small], dim=1)
        refined_small = self.cascade_refine(refine_in)

        refined_full = F.interpolate(
            refined_small, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        logits_full = F.interpolate(
            logits, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        return logits_full + refined_full

    def _forward_latefusion_logits(self, x):
        modality_feats = [
            self._apply_seg_gates(feats)
            for feats in self.extract_features_per_modality(x)
        ]
        per_modality_logits = [
            self._decode_logits(
                feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
            )
            for feats in modality_feats
        ]
        if len(per_modality_logits) == 1:
            return per_modality_logits[0]
        return self.logit_fusion(torch.cat(per_modality_logits, dim=1))

    def _forward_featureavg(self, x):
        modality_feats = [
            self._apply_seg_gates(feats)
            for feats in self.extract_features_per_modality(x)
        ]
        if len(modality_feats) == 1:
            feats = modality_feats[0]
        else:
            feats = {
                name: torch.stack(
                    [modality[name] for modality in modality_feats], dim=0
                ).mean(dim=0)
                for name in ("mid", "up0", "up1", "up2", "up3")
            }
        return self._decode_logits(
            feats["mid"], feats["up0"], feats["up1"], feats["up2"], feats["up3"]
        )

    def _forward_flairskip_latefusion(self, x):
        modality_feats = [
            self._apply_seg_gates(feats)
            for feats in self.extract_features_per_modality(x)
        ]
        if len(modality_feats) < 2:
            return self._decode_logits(
                modality_feats[0]["mid"],
                modality_feats[0]["up0"],
                modality_feats[0]["up1"],
                modality_feats[0]["up2"],
                modality_feats[0]["up3"],
            )

        flair_idx = 1 if len(modality_feats) > 1 else 0
        flair_feats = modality_feats[flair_idx]
        context_feats = [
            feats for idx, feats in enumerate(modality_feats) if idx != flair_idx
        ]

        def mean_context(name):
            return torch.stack([feats[name] for feats in context_feats], dim=0).mean(dim=0)

        mid = self.mid_fuse(torch.cat([mean_context("mid"), flair_feats["mid"]], dim=1))
        mid = mid + self.flair_mid_residual(flair_feats["mid"])
        up0 = self.up0_fuse(torch.cat([mean_context("up0"), flair_feats["up0"]], dim=1))
        return self._decode_logits(
            mid, up0, flair_feats["up1"], flair_feats["up2"], flair_feats["up3"]
        )

    def _forward_pretrained_decoder(self, x):
        _, _, d_orig, h_orig, w_orig = x.shape
        self._last_aux_outputs = {}
        if self._base_seg_head_variant in {"waveletavg", "waveletavg_refine"}:
            x_wave = self._wavelet_modalities(x)
            x_fused = x_wave[:, 0] if x_wave.shape[1] == 1 else x_wave.mean(dim=1)
            feats = self._run_backbone(x_fused)
        else:
            feats = self.extract_features(x)
        hidden = self.backbone.out[1](self.backbone.out[0](feats["up_last"]))
        coeffs = self.ptdec_out(hidden)
        b, _, d, h, w = coeffs.shape
        coeffs = coeffs.reshape(b * self.num_classes, 64, d, h, w)
        with torch.autocast(device_type=coeffs.device.type, enabled=False):
            logits = self.ptdec_idwt(coeffs.float())
        logits = logits.reshape(b, self.num_classes, *logits.shape[2:])
        logits = logits[:, :, :d_orig, :h_orig, :w_orig]
        if self.image_refine is not None:
            logits = logits + self.image_refine(torch.cat([logits, x], dim=1))
        return logits

    def forward(self, x):
        if self.pretrained_decoder:
            return self._forward_pretrained_decoder(x)
        _, _, d_orig, h_orig, w_orig = x.shape
        self._last_aux_outputs = {}
        if self.seg_head_variant == "latefusion_logits":
            logits = self._forward_latefusion_logits(x)
        elif self.seg_head_variant in {"waveletavg", "waveletavg_refine"}:
            logits = self._forward_waveletavg(x)
        elif self.seg_head_variant in {"waveletweighted", "waveletweighted_refine"}:
            logits = self._forward_waveletweighted(x)
        elif self.seg_head_variant in {"image_refine_ms", "image_refine_ms_aux"}:
            logits, feats = self._forward_waveletavg_with_feats(x)
        elif self.seg_head_variant == "cascade_refine":
            logits = self._forward_cascade_refine(x)
        elif self.seg_head_variant == "featureavg":
            logits = self._forward_featureavg(x)
        elif self.seg_head_variant == "flairskip_latefusion":
            logits = self._forward_flairskip_latefusion(x)
        else:
            logits = self._forward_earlyfusion(x)

        logits = F.interpolate(
            logits,
            size=(d_orig, h_orig, w_orig),
            mode="trilinear",
            align_corners=False,
        )
        if self.image_refine_ms is not None:
            logits, boundary_logits, _ = self.image_refine_ms(
                logits, x, feats["up2"], feats["up1"]
            )
            self._last_aux_outputs["boundary_logits"] = boundary_logits
        if self.image_refine is not None:
            logits = logits + self.image_refine(torch.cat([logits, x], dim=1))
        return logits

class CleanDIFTModalityExpertSegWrapper(YuccaNet):

    _LOGIT_FUSIONS = {
        "modality_expert",
        "modality_expert_softmax",
        "modality_expert_sharedbackbone",
    }
    def __init__(
        self,
        input_channels,
        num_classes,
        seg_attention: str = "none",
        seg_head_variant: str = "modality_expert",
        seg_output_fg_prior=None,
        backbone_num_channels=None,
        backbone_num_res_blocks=None,
        **kwargs,
    ):
        super().__init__()
        if input_channels < 1:
            raise ValueError("CleanDIFT modality expert requires at least one modality")
        self.num_modalities = input_channels
        self.num_classes = num_classes
        self.seg_attention = seg_attention
        self.seg_head_variant = seg_head_variant
        self._last_aux_outputs = {}

        branch_variant = (
            "image_refine_v1"
            if seg_head_variant in self._LOGIT_FUSIONS or seg_head_variant == "modality_expert_avg"
            else "earlyfusion"
        )
        branch_kwargs = {}
        if backbone_num_channels is not None:
            branch_kwargs["backbone_num_channels"] = backbone_num_channels
        if backbone_num_res_blocks is not None:
            branch_kwargs["backbone_num_res_blocks"] = backbone_num_res_blocks
        branch_kwargs.update(
            {
                k: v
                for k, v in kwargs.items()
                if k.startswith("backbone_")
            }
        )

        self.modality_experts = nn.ModuleList(
            [
                CleanDIFTSegWrapper(
                    input_channels=1,
                    num_classes=num_classes,
                    seg_attention=seg_attention,
                    seg_head_variant=branch_variant,
                    seg_output_fg_prior=seg_output_fg_prior,
                    **branch_kwargs,
                )
                for _ in range(input_channels)
            ]
        )

        if seg_head_variant == "modality_expert_sharedbackbone":
            shared_backbone = self.modality_experts[0].backbone
            for expert in self.modality_experts[1:]:
                expert.backbone = shared_backbone

        if seg_head_variant in self._LOGIT_FUSIONS:
            self.modality_logit_weights = nn.Parameter(torch.zeros(input_channels))
            self.modality_logit_fusion = None
            self.modality_feature_fusion = None
        elif seg_head_variant == "modality_expert_avg":
            self.modality_logit_weights = None
            self.modality_logit_fusion = None
            self.modality_feature_fusion = None
        elif seg_head_variant == "modality_expert_logitconv":
            self.modality_logit_weights = None
            self.modality_logit_fusion = nn.Conv3d(
                input_channels * num_classes, num_classes, kernel_size=3, padding=1
            )
            self.modality_feature_fusion = None
            self._init_logit_conv_average()
        elif seg_head_variant == "modality_expert_featureconcat":
            self.modality_logit_weights = None
            self.modality_logit_fusion = None
            self.modality_feature_fusion = nn.Sequential(
                nn.Conv3d(input_channels * 16, 32, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(32), 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, num_classes, kernel_size=1),
            )
            nn.init.zeros_(self.modality_feature_fusion[-1].weight)
            if self.modality_feature_fusion[-1].bias is not None:
                nn.init.zeros_(self.modality_feature_fusion[-1].bias)
        else:
            raise ValueError(f"Unsupported modality-expert fusion: {seg_head_variant}")

    def _init_logit_conv_average(self):
        nn.init.zeros_(self.modality_logit_fusion.weight)
        nn.init.zeros_(self.modality_logit_fusion.bias)
        center = self.modality_logit_fusion.kernel_size[0] // 2
        with torch.no_grad():
            for cls_idx in range(self.num_classes):
                for mod_idx in range(self.num_modalities):
                    src = mod_idx * self.num_classes + cls_idx
                    self.modality_logit_fusion.weight[
                        cls_idx, src, center, center, center
                    ] = 1.0 / self.num_modalities

    def expand_pretrained_state_dict(self, state_dict: dict, prefix: str = "") -> dict:
        expanded = {}
        for key, value in state_dict.items():
            bare_key = key
            if bare_key.startswith(prefix):
                bare_key = bare_key[len(prefix):]
            if bare_key.startswith("model."):
                bare_key = bare_key[len("model."):]
            if bare_key.startswith("backbone."):
                bare_key = bare_key[len("backbone."):]
            if bare_key.startswith("teacher.") or bare_key.startswith("projection_heads."):
                continue
            for idx in range(self.num_modalities):
                expanded[f"{prefix}modality_experts.{idx}.backbone.{bare_key}"] = value
        return expanded

    def _masked_weights(self, modality_mask, logits):
        if modality_mask is None:
            modality_mask = torch.ones(
                logits.shape[0],
                self.num_modalities,
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            modality_mask = modality_mask.to(device=logits.device, dtype=logits.dtype)
            if modality_mask.ndim == 1:
                modality_mask = modality_mask.unsqueeze(0).expand(logits.shape[0], -1)
        if modality_mask.shape != (logits.shape[0], self.num_modalities):
            raise ValueError(
                f"Expected modality_mask shape {(logits.shape[0], self.num_modalities)}, "
                f"got {tuple(modality_mask.shape)}"
            )
        if (modality_mask.sum(dim=1) <= 0).any():
            raise ValueError("Each sample must keep at least one modality")

        if self.modality_logit_weights is None:
            weights = modality_mask / modality_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            raw = self.modality_logit_weights.to(logits.dtype).unsqueeze(0).expand_as(modality_mask)
            raw = raw.masked_fill(modality_mask <= 0, torch.finfo(logits.dtype).min)
            weights = torch.softmax(raw, dim=1)
        return weights

    def _expert_logits(self, x):
        return [
            expert(x[:, idx : idx + 1])
            for idx, expert in enumerate(self.modality_experts)
        ]

    def _expert_decoder_features(self, x):
        logits = []
        features = []
        for idx, expert in enumerate(self.modality_experts):
            logit, feat = expert._forward_earlyfusion_with_decoder_feature(x[:, idx : idx + 1])
            logits.append(logit)
            features.append(feat)
        return logits, features

    def forward(self, x, modality_mask=None):
        _, channels, d_orig, h_orig, w_orig = x.shape
        if channels != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {channels}. "
                "Use FOMO_INPUT_MODALITY_INDICES at data loading time or pass a "
                "mask with the full modality tensor."
            )
        self._last_aux_outputs = {}

        if self.seg_head_variant == "modality_expert_featureconcat":
            _, features = self._expert_decoder_features(x)
            resized = [
                F.interpolate(feat, size=(d_orig, h_orig, w_orig), mode="trilinear", align_corners=False)
                for feat in features
            ]
            if modality_mask is not None:
                mask = modality_mask.to(device=x.device, dtype=x.dtype)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0).expand(x.shape[0], -1)
                resized = [
                    feat * mask[:, idx].view(-1, 1, 1, 1, 1)
                    for idx, feat in enumerate(resized)
                ]
            logits = self.modality_feature_fusion(torch.cat(resized, dim=1))
        else:
            logits_by_modality = [
                F.interpolate(logit, size=(d_orig, h_orig, w_orig), mode="trilinear", align_corners=False)
                for logit in self._expert_logits(x)
            ]
            stacked = torch.stack(logits_by_modality, dim=1)
            if self.seg_head_variant == "modality_expert_logitconv":
                if modality_mask is not None:
                    mask = modality_mask.to(device=x.device, dtype=stacked.dtype)
                    if mask.ndim == 1:
                        mask = mask.unsqueeze(0).expand(x.shape[0], -1)
                    stacked = stacked * mask.view(x.shape[0], self.num_modalities, 1, 1, 1, 1)
                logits = self.modality_logit_fusion(stacked.flatten(1, 2))
            else:
                weights = self._masked_weights(modality_mask, stacked)
                logits = (stacked * weights.view(x.shape[0], self.num_modalities, 1, 1, 1, 1)).sum(dim=1)
                self._last_aux_outputs["modality_weights"] = weights.detach()
        return logits

class CleanDIFTRegWrapper(CleanDIFTScalarWrapper):

    pass

class CleanDIFTRegMultiTapWrapper(CleanDIFTScalarMultiTapWrapper):

    pass

class CleanDIFTRegPEFTTapWrapper(CleanDIFTScalarPEFTTapWrapper):

    pass

def _cleandift_factory(
    mode,
    input_channels,
    num_classes,
    backbone_num_channels=None,
    backbone_num_res_blocks=None,
    seg_attention: str = "none",
    seg_head_variant: str = "earlyfusion",
    seg_output_fg_prior=None,
):
    modality_expert_variants = {
        "modality_expert",
        "modality_expert_softmax",
        "modality_expert_avg",
        "modality_expert_logitconv",
        "modality_expert_featureconcat",
        "modality_expert_sharedbackbone",
    }
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
        if seg_head_variant in modality_expert_variants:
            return CleanDIFTModalityExpertSegWrapper(
                input_channels=input_channels,
                num_classes=num_classes,
                seg_attention=seg_attention,
                seg_head_variant=seg_head_variant,
                seg_output_fg_prior=seg_output_fg_prior,
                **bkw,
            )
        return CleanDIFTSegWrapper(
            input_channels=input_channels,
            num_classes=num_classes,
            seg_attention=seg_attention,
            seg_head_variant=seg_head_variant,
            seg_output_fg_prior=seg_output_fg_prior,
            **bkw,
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
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        seg_attention=kwargs.get("seg_attention", "none"),
        seg_head_variant=kwargs.get("seg_head_variant", "earlyfusion"),
        seg_output_fg_prior=kwargs.get("seg_output_fg_prior", None),
    )

def cleandift_medium(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[32, 64, 128, 192, 256],
        backbone_num_res_blocks=2,
        seg_attention=kwargs.get("seg_attention", "none"),
        seg_head_variant=kwargs.get("seg_head_variant", "earlyfusion"),
        seg_output_fg_prior=kwargs.get("seg_output_fg_prior", None),
    )

def cleandift_medium_backbone(num_modalities=3):
    return CleanDIFTBase(
        num_modalities=num_modalities,
        backbone_num_channels=[32, 64, 128, 192, 256],
        backbone_num_res_blocks=2,
    )

def cleandift_s15(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[32, 32, 64, 64, 128],
        backbone_num_res_blocks=2,
        seg_attention=kwargs.get("seg_attention", "none"),
        seg_head_variant=kwargs.get("seg_head_variant", "earlyfusion"),
        seg_output_fg_prior=kwargs.get("seg_output_fg_prior", None),
    )

def cleandift_s23(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[40, 40, 80, 80, 160],
        backbone_num_res_blocks=2,
        seg_attention=kwargs.get("seg_attention", "none"),
        seg_head_variant=kwargs.get("seg_head_variant", "earlyfusion"),
        seg_output_fg_prior=kwargs.get("seg_output_fg_prior", None),
    )

def cleandift_s23_hc16(mode, input_channels, output_channels=None, num_classes=None, **kwargs):
    nc = num_classes or output_channels or 1
    bkw = dict(backbone_num_channels=[40, 40, 80, 80, 160], backbone_num_res_blocks=2,
               backbone_num_head_channels=[0, 0, 0, 16, 16])
    if mode == "classification":
        return CleanDIFTScalarWrapper(input_channels=input_channels, num_classes=nc, **bkw)
    if mode == "regression":
        return CleanDIFTRegWrapper(input_channels=input_channels, num_classes=nc, **bkw)
    if mode == "segmentation":
        return CleanDIFTSegWrapper(
            input_channels=input_channels, num_classes=nc,
            seg_attention=kwargs.get("seg_attention", "none"),
            seg_head_variant=kwargs.get("seg_head_variant", "earlyfusion"),
            seg_output_fg_prior=kwargs.get("seg_output_fg_prior", None), **bkw)
    raise ValueError(f"Unsupported mode for CleanDIFT: {mode}")

def cleandift_s23_multitap(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
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
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[48, 48, 96, 96, 192],
        backbone_num_res_blocks=2,
        seg_attention=kwargs.get("seg_attention", "none"),
        seg_head_variant=kwargs.get("seg_head_variant", "earlyfusion"),
        seg_output_fg_prior=kwargs.get("seg_output_fg_prior", None),
    )

def cleandift_s33_pefttap(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
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
    nc = num_classes or output_channels or 1
    return _cleandift_factory(
        mode,
        input_channels,
        nc,
        backbone_num_channels=[32, 64, 128],
        backbone_num_res_blocks=2,
        seg_attention=kwargs.get("seg_attention", "none"),
        seg_head_variant=kwargs.get("seg_head_variant", "earlyfusion"),
        seg_output_fg_prior=kwargs.get("seg_output_fg_prior", None),
    )
