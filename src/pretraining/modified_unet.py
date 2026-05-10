"""
Modified 3D U-Net with Fixed Upsampling (Intervention 1)

This module implements a modified DiffusionModelUNet that replaces all 
transposed convolutions with Upsample + Conv3d to eliminate checkerboard artifacts.

Root Cause Target: H2 (U-Net Upsampling Artifacts)
Expected Impact: 50-70% reduction in Periodicity Ratio
"""

import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet
from typing import Sequence


class UpConvBlock(nn.Module):
    """
    Replacement for ConvTranspose3d using interpolation + convolution.

    This eliminates the "uneven overlap" problem that causes checkerboard
    artifacts in transposed convolutions.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = "trilinear",
        align_corners: bool = False,
    ):
        """
        Args:
            spatial_dims: Number of spatial dimensions (must be 3 for 3D)
            in_channels: Number of input channels
            out_channels: Number of output channels
            scale_factor: Upsampling factor (default: 2)
            mode: Interpolation mode ('trilinear' for 3D)
            align_corners: Whether to align corners in interpolation
        """
        super().__init__()

        if spatial_dims != 3:
            raise ValueError(
                f"This implementation is for 3D only, got spatial_dims={spatial_dims}"
            )

        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C_in, D, H, W]

        Returns:
            Upsampled tensor [B, C_out, 2*D, 2*H, 2*W]
        """
        x = self.upsample(x)
        x = self.conv(x)
        return x


class ArtifactFreeDiffusionUNet(nn.Module):
    """
    Wrapper around DiffusionModelUNet that replaces decoder upsampling layers.

    This class creates a standard DiffusionModelUNet but then surgically
    replaces all ConvTranspose3d layers in the decoder with UpConvBlock
    to eliminate checkerboard artifacts.

    Usage:
        model = ArtifactFreeDiffusionUNet(
            spatial_dims=3,
            in_channels=64,
            out_channels=64,
            num_channels=[128, 128, 256, 256, 512],
            attention_levels=[False, False, False, True, True],
            num_head_channels=[0, 0, 0, 32, 32],
            num_res_blocks=2,
            with_conditioning=True,
            cross_attention_dim=1
        )
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 64,
        out_channels: int = 64,
        num_channels: Sequence[int] = (128, 128, 256, 256, 512),
        attention_levels: Sequence[bool] = (False, False, False, True, True),
        num_head_channels: Sequence[int] = (0, 0, 0, 32, 32),
        num_res_blocks: int = 2,
        use_flash_attention: bool = False,
        with_conditioning: bool = False,
    ):
        """
        Initialize the artifact-free U-Net.

        Replaces ConvTranspose3d with Upsample+Conv3d in the decoder
        to eliminate checkerboard artifacts.

        Compatible with generative.networks.nets.DiffusionModelUNet
        """
        super().__init__()

        unet_kwargs = {
            "spatial_dims": spatial_dims,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_channels": num_channels,
            "attention_levels": attention_levels,
            "num_head_channels": num_head_channels,
            "num_res_blocks": num_res_blocks,
            "use_flash_attention": use_flash_attention,
            "with_conditioning": with_conditioning,
        }

        if with_conditioning:
            unet_kwargs["cross_attention_dim"] = 1

        self.unet = DiffusionModelUNet(**unet_kwargs)

        self._replace_transposed_convolutions()

    def _replace_transposed_convolutions(self):
        """
        Surgically replace all nn.ConvTranspose3d layers with UpConvBlock.

        This function walks through the U-Net's module tree and replaces
        transposed convolutions in the decoder (upsampling path).
        """
        replaced_count = 0

        def replace_conv_transpose_recursive(module, prefix=""):
            nonlocal replaced_count

            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(child, nn.ConvTranspose3d):
                    scale_factor = 2
                    if hasattr(child, "stride"):
                        stride = child.stride
                        if isinstance(stride, tuple):
                            scale_factor = stride[0]
                        else:
                            scale_factor = stride

                    replacement = UpConvBlock(
                        spatial_dims=3,
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        scale_factor=scale_factor,
                        mode="trilinear",
                        align_corners=False,
                    )

                    setattr(module, name, replacement)
                    replaced_count += 1
                    print(f"✓ Replaced ConvTranspose3d at: {full_name}")
                else:
                    replace_conv_transpose_recursive(child, full_name)

        replace_conv_transpose_recursive(self.unet)

        print(f"\n{'='*70}")
        print(f"INTERVENTION 1 APPLIED: Artifact-Free Upsampling")
        print(f"✓ Replaced {replaced_count} ConvTranspose3d layers with UpConvBlock")
        print(f"  Strategy: Replace with Upsample(trilinear) + Conv3d")
        print(f"  Expected: 50-70% reduction in Periodicity Ratio (5,019 → <2,000)")
        print(f"{'='*70}\n")

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the modified U-Net.

        Args:
            x: Input tensor [B, C, D, H, W] (wavelet coefficients)
            timesteps: Diffusion timesteps [B]
            context: Conditioning information (e.g., age) [B, 1, 1]

        Returns:
            Predicted noise [B, C, D, H, W]
        """
        return self.unet(x, timesteps, context=context)


class UnetWrapper(nn.Module):
    """
    Simple wrapper to match the original training script's API.

    This ensures compatibility with existing training code that uses
    UnetWrapper with cross-attention conditioning.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, timesteps, context):
        return self.unet(x, timesteps, context=context)


def create_artifact_free_unet(
    channels: int = 64,
    with_conditioning: bool = True,
    cross_attention_dim: int = 1,
    device: str = "cuda",
) -> UnetWrapper:
    """
    Factory function to create artifact-free U-Net with default parameters.

    Args:
        channels: Number of wavelet coefficient channels (default: 64)
        with_conditioning: Whether to use conditioning (default: True)
        cross_attention_dim: Dimension of cross-attention conditioning (default: 1)
        device: Device to place model on (default: "cuda")

    Returns:
        UnetWrapper containing ArtifactFreeDiffusionUNet

    Example:
        >>> model = create_artifact_free_unet(channels=64)
        >>> model = model.to(device)
        >>> output = model(x, timesteps, context)
    """
    base_model = ArtifactFreeDiffusionUNet(
        spatial_dims=3,
        in_channels=channels,
        out_channels=channels,
        num_channels=[128, 128, 256, 256, 512],
        attention_levels=[False, False, False, True, True],
        num_head_channels=[0, 0, 0, 32, 32],
        num_res_blocks=2,
        use_flash_attention=False,
        with_conditioning=with_conditioning,
        cross_attention_dim=cross_attention_dim,
    )

    wrapped_model = UnetWrapper(base_model)
    wrapped_model = wrapped_model.to(device)

    return wrapped_model


if __name__ == "__main__":
    print("Testing ArtifactFreeDiffusionUNet...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = create_artifact_free_unet(channels=64, device=device)

    batch_size = 2
    x = torch.randn(batch_size, 64, 48, 64, 48).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    context = torch.randn(batch_size, 1, 1).to(device)

    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps: {timesteps.shape}")
    print(f"Context: {context.shape}")

    with torch.no_grad():
        output = model(x, timesteps, context)

    print(f"Output shape: {output.shape}")
    print("\nModel test passed!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
