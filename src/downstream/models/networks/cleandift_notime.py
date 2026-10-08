from __future__ import annotations

import types

import torch
import torch.nn as nn

from .cleandift import (
    CleanDIFTRegWrapper,
    CleanDIFTScalarWrapper,
    CleanDIFTSegWrapper,
)

_PRETRAIN_HEAD_CHANNELS = [0, 0, 0, 16, 16]
_S23_CHANNELS = [40, 40, 80, 80, 160]

class PassthroughTimeEmbed(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def _time_independent_resnet_forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
    h = x
    h = self.norm1(h)
    h = self.nonlinearity(h)

    if self.upsample is not None:
        if h.shape[0] >= 64:
            x = x.contiguous()
            h = h.contiguous()
        x = self.upsample(x)
        h = self.upsample(h)
    elif self.downsample is not None:
        x = self.downsample(x)
        h = self.downsample(h)

    h = self.conv1(h)

    h = self.norm2(h)
    h = self.nonlinearity(h)
    h = self.conv2(h)

    return self.skip_connection(x) + h

def strip_time_conditioning(backbone: nn.Module) -> int:
    removed = 0

    if hasattr(backbone, "time_embed"):
        removed += sum(p.numel() for p in backbone.time_embed.parameters())
        backbone.time_embed = PassthroughTimeEmbed()

    n_blocks = 0
    for module in backbone.modules():
        if hasattr(module, "time_emb_proj") and isinstance(module.time_emb_proj, nn.Linear):
            removed += sum(p.numel() for p in module.time_emb_proj.parameters())
            del module.time_emb_proj
            module.forward = types.MethodType(_time_independent_resnet_forward, module)
            n_blocks += 1

    if n_blocks == 0:
        raise RuntimeError(
            "no ResNet blocks with a time_emb_proj were found; the MONAI layout may have changed, and "
            "silently leaving time conditioning in place would invalidate the comparison"
        )

    return removed

def _strip_wrapper(model: nn.Module) -> nn.Module:
    stripped_any = False

    if hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        strip_time_conditioning(model.backbone)
        stripped_any = True

    for attr in ("branches", "experts"):
        container = getattr(model, attr, None)
        if isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            for branch in container:
                target = branch if isinstance(branch, nn.Module) else None
                if target is not None and hasattr(target, "backbone"):
                    strip_time_conditioning(target.backbone)
                    stripped_any = True

    if not stripped_any:
        raise RuntimeError(f"no backbone found to strip on {type(model).__name__}")

    return model

def cleandift_s23_hc16_notime(
    mode, input_channels, output_channels=None, num_classes=None, **kwargs
):
    nc = num_classes or output_channels or 1
    bkw = dict(
        backbone_num_channels=_S23_CHANNELS,
        backbone_num_res_blocks=2,
        backbone_num_head_channels=_PRETRAIN_HEAD_CHANNELS,
    )

    if mode == "classification":
        model = CleanDIFTScalarWrapper(input_channels=input_channels, num_classes=nc, **bkw)
    elif mode == "regression":
        model = CleanDIFTRegWrapper(input_channels=input_channels, num_classes=nc, **bkw)
    elif mode == "segmentation":
        model = CleanDIFTSegWrapper(
            input_channels=input_channels,
            num_classes=nc,
            seg_attention=kwargs.get("seg_attention", "none"),
            seg_head_variant=kwargs.get("seg_head_variant", "earlyfusion"),
            seg_output_fg_prior=kwargs.get("seg_output_fg_prior", None),
            **bkw,
        )
    else:
        raise ValueError(f"Unsupported mode for CleanDIFT: {mode}")

    return _strip_wrapper(model)

def cleandift_s23_notime(mode, input_channels, output_channels=None, num_classes=None, **kwargs):
    return cleandift_s23_hc16_notime(
        mode, input_channels, output_channels=output_channels, num_classes=num_classes, **kwargs
    )
