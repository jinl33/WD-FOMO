"""
MMUNetVAE downstream network adapted from the released FOMO25 baseline.

Self-contained: includes all conv-block and encoder/decoder dependencies.

Key architecture details (must match pretrained checkpoint keys):
  - Encoder: 2-layer MultiLayerConvDropoutNormNonlin blocks (conv1+conv2 per level)
  - Reconstruction decoder: 1-layer blocks (conv1 only per level)
  - starting_filters=32
  - Pretrained keys: model.encoder.*, model.decoder.*, model.conv_mu/logvar_*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from yucca.modules.networks.utils.get_steps_for_sliding_window import (
    get_steps_for_sliding_window,
)


# ── Conv building blocks ───────────────────────────────────────────────────────

class ConvDropoutNormNonlin(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv3d,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "bias": True},
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        self.dropout = (
            dropout_op(**dropout_op_kwargs)
            if dropout_op is not None and dropout_op_kwargs.get("p", 0) > 0
            else None
        )
        self.norm = norm_op(output_channels, **norm_op_kwargs)
        self.activation = nonlin(**nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.activation(self.norm(x))


class MultiLayerConvDropoutNormNonlin(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers=2,
        conv_op=nn.Conv3d,
        conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "bias": True},
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        kwargs = dict(conv_op=conv_op, conv_kwargs=conv_kwargs, norm_op=norm_op,
                      norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                      dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
        self.conv1 = ConvDropoutNormNonlin(input_channels, output_channels, **kwargs)
        for i in range(2, num_layers + 1):
            setattr(self, f"conv{i}", ConvDropoutNormNonlin(output_channels, output_channels, **kwargs))

    def forward(self, x):
        x = self.conv1(x)
        for i in range(2, self.num_layers + 1):
            x = getattr(self, f"conv{i}")(x)
        return x

    @staticmethod
    def get_block_constructor(n_layers):
        def _block(input_channels, output_channels, **kwargs):
            return MultiLayerConvDropoutNormNonlin(input_channels, output_channels, num_layers=n_layers, **kwargs)
        return _block


# ── Default block kwargs for 3D ────────────────────────────────────────────────

_BK = dict(
    conv_op=nn.Conv3d,
    conv_kwargs={"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "bias": True},
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
    dropout_op=nn.Dropout3d,
    dropout_op_kwargs={"p": 0.0, "inplace": True},
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
)

enc_block = MultiLayerConvDropoutNormNonlin.get_block_constructor(2)  # 2-layer encoder blocks
dec_block = MultiLayerConvDropoutNormNonlin.get_block_constructor(1)  # 1-layer decoder blocks


# ── UNet Encoder (2-layer blocks → keys: conv1+conv2 per level) ───────────────

class UNetEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, starting_filters: int = 32) -> None:
        super().__init__()
        f = starting_filters
        self.in_conv       = enc_block(input_channels, f,      **_BK)
        self.pool1         = nn.MaxPool3d(2)
        self.encoder_conv1 = enc_block(f,      f * 2,  **_BK)
        self.pool2         = nn.MaxPool3d(2)
        self.encoder_conv2 = enc_block(f * 2,  f * 4,  **_BK)
        self.pool3         = nn.MaxPool3d(2)
        self.encoder_conv3 = enc_block(f * 4,  f * 8,  **_BK)
        self.pool4         = nn.MaxPool3d(2)
        self.encoder_conv4 = enc_block(f * 8,  f * 16, **_BK)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.encoder_conv1(self.pool1(x0))
        x2 = self.encoder_conv2(self.pool2(x1))
        x3 = self.encoder_conv3(self.pool3(x2))
        x4 = self.encoder_conv4(self.pool4(x3))
        return [x0, x1, x2, x3, x4]


# ── UNet Decoder (1-layer blocks → keys: conv1 only per level) ────────────────

class UNetDecoder(nn.Module):
    def __init__(self, output_channels: int = 1, starting_filters: int = 32) -> None:
        super().__init__()
        f = starting_filters
        self.upsample1    = nn.ConvTranspose3d(f * 16, f * 8,  kernel_size=2, stride=2)
        self.decoder_conv1 = dec_block(f * 8,  f * 8,  **_BK)
        self.upsample2    = nn.ConvTranspose3d(f * 8,  f * 4,  kernel_size=2, stride=2)
        self.decoder_conv2 = dec_block(f * 4,  f * 4,  **_BK)
        self.upsample3    = nn.ConvTranspose3d(f * 4,  f * 2,  kernel_size=2, stride=2)
        self.decoder_conv3 = dec_block(f * 2,  f * 2,  **_BK)
        self.upsample4    = nn.ConvTranspose3d(f * 2,  f,      kernel_size=2, stride=2)
        self.decoder_conv4 = dec_block(f,      f,      **_BK)
        self.out_conv     = nn.Conv3d(f, output_channels, kernel_size=1)

    def forward(self, xs):
        assert isinstance(xs, list) and len(xs) == 5
        x5 = self.decoder_conv1(self.upsample1(xs[4]))
        x6 = self.decoder_conv2(self.upsample2(x5))
        x7 = self.decoder_conv3(self.upsample3(x6))
        x8 = self.decoder_conv4(self.upsample4(x7))
        return self.out_conv(x8)


# ── Task head ──────────────────────────────────────────────────────────────────

class ClsRegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_channels // 2, num_classes),
        )

    def forward(self, x):
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── VAE helpers ────────────────────────────────────────────────────────────────

def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


# ── Main model ─────────────────────────────────────────────────────────────────

class MultiModalUNetVAE(nn.Module):
    """
    Multi-modal UNet VAE for scalar or classification downstream tasks.
    forward(x) accepts [B, C, D, H, W] and returns [B, num_classes].
    Pretrained keys: model.encoder.*, model.decoder.*, model.conv_mu/logvar_*
    """
    def __init__(
        self,
        input_channels: int = 1,
        starting_filters: int = 32,
        use_vae: bool = True,
        num_classes: int = 1,
        mode: str = "regression",
    ):
        super().__init__()
        f = starting_filters
        self.mode = mode
        self.encoder  = UNetEncoder(input_channels=1, starting_filters=f)
        decoder_outputs = num_classes if mode == "segmentation" else 1
        self.decoder  = UNetDecoder(output_channels=decoder_outputs, starting_filters=f)

        self.num_modalities = input_channels
        self.use_vae        = use_vae
        self.num_classes    = num_classes

        # Bottleneck: pool all 5 encoder levels to same spatial size and concat
        bottleneck_ch   = sum(f * (2 ** i) for i in range(5))  # f+2f+4f+8f+16f = 31f
        latent_shared   = (f * 16) // 2   # 8f
        latent_modality = (f * 16) - latent_shared  # 8f

        self.conv_mu_shared       = nn.Conv3d(bottleneck_ch, latent_shared,   kernel_size=1)
        self.conv_logvar_shared   = nn.Conv3d(bottleneck_ch, latent_shared,   kernel_size=1)
        self.conv_mu_modality     = nn.Conv3d(bottleneck_ch, latent_modality, kernel_size=1)
        self.conv_logvar_modality = nn.Conv3d(bottleneck_ch, latent_modality, kernel_size=1)

        out_dim = latent_modality * input_channels + latent_shared
        self.decoder_task = (
            None if mode == "segmentation"
            else ClsRegHead(in_channels=out_dim, num_classes=num_classes)
        )

    def _encode(self, x):
        skips   = self.encoder(x)
        ref_sz  = skips[-1].shape[2:]
        pooled  = torch.cat([F.adaptive_avg_pool3d(s, ref_sz) for s in skips], dim=1)
        mu_s    = self.conv_mu_shared(pooled)
        logvar_s = self.conv_logvar_shared(pooled)
        mu_m    = self.conv_mu_modality(pooled)
        logvar_m = self.conv_logvar_modality(pooled)
        if self.training and self.use_vae:
            return reparameterize(mu_s, logvar_s), reparameterize(mu_m, logvar_m)
        return mu_s, mu_m

    def _encode_skips_multimodal(self, x):
        per_modality = [self.encoder(x[:, [ix]]) for ix in range(self.num_modalities)]
        fused = []
        for level in range(len(per_modality[0])):
            stacked = torch.stack([mod[level] for mod in per_modality], dim=0)
            fused.append(stacked.mean(dim=0))
        return fused

    def _forward_segmentation(self, x):
        fused_skips = self._encode_skips_multimodal(x)
        return self.decoder(fused_skips)

    def _forward_scalar(self, x):
        z_s_all, z_m_all = [], []
        for ix in range(self.num_modalities):
            z_s, z_m = self._encode(x[:, [ix]])
            z_s_all.append(z_s)
            z_m_all.append(z_m)
        z_s = torch.stack(z_s_all).mean(dim=0)
        z_m = torch.cat(z_m_all, dim=1)
        return self.decoder_task(torch.cat([z_s, z_m], dim=1))

    def forward(self, x):
        if self.mode == "segmentation":
            return self._forward_segmentation(x)
        return self._forward_scalar(x)

    def _predict_patch_with_mirror(self, patch, mirror):
        if not mirror:
            return self.forward(patch)
        preds = [self.forward(patch)]
        flip_dims = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
        for dims in flip_dims:
            flipped = torch.flip(patch, dims=dims)
            pred = self.forward(flipped)
            preds.append(torch.flip(pred, dims=dims))
        return torch.stack(preds, dim=0).mean(dim=0)

    def _patch_segmentation_predict(self, data, patch_size, overlap, mirror):
        b, _, d, h, w = data.shape
        x_steps, y_steps, z_steps = get_steps_for_sliding_window((d, h, w), patch_size, overlap)
        px, py, pz = patch_size
        logits_sum = torch.zeros(
            (b, self.num_classes, d, h, w),
            device=data.device,
            dtype=data.dtype,
        )
        count_map = torch.zeros((1, 1, d, h, w), device=data.device, dtype=data.dtype)

        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    patch = data[:, :, xs:xs + px, ys:ys + py, zs:zs + pz]
                    logits = self._predict_patch_with_mirror(patch, mirror=mirror)
                    logits_sum[:, :, xs:xs + px, ys:ys + py, zs:zs + pz] += logits
                    count_map[:, :, xs:xs + px, ys:ys + py, zs:zs + pz] += 1

        if not torch.all(count_map > 0):
            raise RuntimeError("Sliding-window count map contains uncovered voxels")
        return logits_sum / count_map

    def predict(
        self,
        mode,
        data,
        patch_size,
        overlap,
        sliding_window_prediction=True,
        mirror=False,
        device="cpu",
    ):
        if not sliding_window_prediction:
            return self.forward(data)
        if self.mode == "segmentation":
            return self._patch_segmentation_predict(data, patch_size, overlap, mirror)
        return self._patch_regression_predict(data, patch_size, overlap)

    def _patch_regression_predict(self, data, patch_size, overlap):
        logits_list = []
        x_steps, y_steps, z_steps = get_steps_for_sliding_window(
            data.shape[2:], patch_size, overlap
        )
        px, py, pz = patch_size

        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    patch = data[:, :, xs : xs + px, ys : ys + py, zs : zs + pz]
                    logits = self.forward(patch)
                    logits_list.append(logits)

        if not logits_list:
            raise RuntimeError("No patches were generated for MMUNetVAE prediction")

        logits_stack = torch.stack(logits_list, dim=0)
        return torch.median(logits_stack, dim=0).values


# ── Public constructor ─────────────────────────────────────────────────────────

def mmunetvae(
    input_channels: int = 1,
    output_channels: int = 1,
    mode: str = "regression",
    use_vae: bool = True,
    use_skip_connections: bool = False,
    num_classes: int = 1,
):
    task_outputs = num_classes or output_channels or 1
    return MultiModalUNetVAE(
        input_channels=input_channels,
        starting_filters=32,
        use_vae=use_vae,
        num_classes=task_outputs,
        mode=mode,
    )
