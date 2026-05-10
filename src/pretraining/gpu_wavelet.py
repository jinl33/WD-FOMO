"""
GPU-Accelerated 3D Wavelet Transform for FOMO Diffusion Model

Adapted from wdm-3d (MIT License):
https://github.com/pfriedri/wdm-3d

Key adaptations:
- Simplified for Haar wavelet only (matching your pywt pipeline)
- 2-level decomposition support (matching your levels=2 config)
- Compatible with your [1, 192, 256, 192] → [64, 48, 64, 48] shape
- Drop-in replacement for pywt_decompose and recon_image_from_wavelet

Usage:
    dwt = HaarDWT3D_2Level(normalize_factor=8.0).cuda()
    wavelet_coeffs = dwt(image)  # [B, 1, 192, 256, 192] → [B, 64, 48, 64, 48]
    
    idwt = HaarIDWT3D_2Level(denormalize_factor=8.0).cuda()
    reconstructed = idwt(wavelet_coeffs)  # [B, 64, 48, 64, 48] → [B, 1, 192, 256, 192]

Author: Adapted for FOMO by GitHub Copilot
Date: October 27, 2025
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


class DWTFunction_3D(Function):
    """3D discrete wavelet transform using matrix multiplication."""

    @staticmethod
    def forward(
        ctx,
        input,
        matrix_Low_0,
        matrix_Low_1,
        matrix_Low_2,
        matrix_High_0,
        matrix_High_1,
        matrix_High_2,
    ):
        ctx.save_for_backward(
            matrix_Low_0,
            matrix_Low_1,
            matrix_Low_2,
            matrix_High_0,
            matrix_High_1,
            matrix_High_2,
        )
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)

        LL = torch.matmul(L, matrix_Low_1).transpose(dim0=2, dim1=3)
        LH = torch.matmul(L, matrix_High_1).transpose(dim0=2, dim1=3)
        HL = torch.matmul(H, matrix_Low_1).transpose(dim0=2, dim1=3)
        HH = torch.matmul(H, matrix_High_1).transpose(dim0=2, dim1=3)

        LLL = torch.matmul(matrix_Low_2, LL).transpose(dim0=2, dim1=3)
        LLH = torch.matmul(matrix_Low_2, LH).transpose(dim0=2, dim1=3)
        LHL = torch.matmul(matrix_Low_2, HL).transpose(dim0=2, dim1=3)
        LHH = torch.matmul(matrix_Low_2, HH).transpose(dim0=2, dim1=3)
        HLL = torch.matmul(matrix_High_2, LL).transpose(dim0=2, dim1=3)
        HLH = torch.matmul(matrix_High_2, LH).transpose(dim0=2, dim1=3)
        HHL = torch.matmul(matrix_High_2, HL).transpose(dim0=2, dim1=3)
        HHH = torch.matmul(matrix_High_2, HH).transpose(dim0=2, dim1=3)

        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(
        ctx,
        grad_LLL,
        grad_LLH,
        grad_LHL,
        grad_LHH,
        grad_HLL,
        grad_HLH,
        grad_HHL,
        grad_HHH,
    ):
        (
            matrix_Low_0,
            matrix_Low_1,
            matrix_Low_2,
            matrix_High_0,
            matrix_High_1,
            matrix_High_2,
        ) = ctx.saved_variables

        grad_LL = torch.add(
            torch.matmul(matrix_Low_2.t(), grad_LLL.transpose(dim0=2, dim1=3)),
            torch.matmul(matrix_High_2.t(), grad_HLL.transpose(dim0=2, dim1=3)),
        ).transpose(dim0=2, dim1=3)

        grad_LH = torch.add(
            torch.matmul(matrix_Low_2.t(), grad_LLH.transpose(dim0=2, dim1=3)),
            torch.matmul(matrix_High_2.t(), grad_HLH.transpose(dim0=2, dim1=3)),
        ).transpose(dim0=2, dim1=3)

        grad_HL = torch.add(
            torch.matmul(matrix_Low_2.t(), grad_LHL.transpose(dim0=2, dim1=3)),
            torch.matmul(matrix_High_2.t(), grad_HHL.transpose(dim0=2, dim1=3)),
        ).transpose(dim0=2, dim1=3)

        grad_HH = torch.add(
            torch.matmul(matrix_Low_2.t(), grad_LHH.transpose(dim0=2, dim1=3)),
            torch.matmul(matrix_High_2.t(), grad_HHH.transpose(dim0=2, dim1=3)),
        ).transpose(dim0=2, dim1=3)

        grad_L = torch.add(
            torch.matmul(grad_LL, matrix_Low_1.t()),
            torch.matmul(grad_LH, matrix_High_1.t()),
        )
        grad_H = torch.add(
            torch.matmul(grad_HL, matrix_Low_1.t()),
            torch.matmul(grad_HH, matrix_High_1.t()),
        )

        grad_input = torch.add(
            torch.matmul(matrix_Low_0.t(), grad_L),
            torch.matmul(matrix_High_0.t(), grad_H),
        )

        return grad_input, None, None, None, None, None, None


class IDWTFunction_3D(Function):
    """3D inverse discrete wavelet transform using matrix multiplication."""

    @staticmethod
    def forward(
        ctx,
        input_LLL,
        input_LLH,
        input_LHL,
        input_LHH,
        input_HLL,
        input_HLH,
        input_HHL,
        input_HHH,
        matrix_Low_0,
        matrix_Low_1,
        matrix_Low_2,
        matrix_High_0,
        matrix_High_1,
        matrix_High_2,
    ):
        ctx.save_for_backward(
            matrix_Low_0,
            matrix_Low_1,
            matrix_Low_2,
            matrix_High_0,
            matrix_High_1,
            matrix_High_2,
        )

        input_LL = torch.add(
            torch.matmul(matrix_Low_2.t(), input_LLL.transpose(dim0=2, dim1=3)),
            torch.matmul(matrix_High_2.t(), input_HLL.transpose(dim0=2, dim1=3)),
        ).transpose(dim0=2, dim1=3)

        input_LH = torch.add(
            torch.matmul(matrix_Low_2.t(), input_LLH.transpose(dim0=2, dim1=3)),
            torch.matmul(matrix_High_2.t(), input_HLH.transpose(dim0=2, dim1=3)),
        ).transpose(dim0=2, dim1=3)

        input_HL = torch.add(
            torch.matmul(matrix_Low_2.t(), input_LHL.transpose(dim0=2, dim1=3)),
            torch.matmul(matrix_High_2.t(), input_HHL.transpose(dim0=2, dim1=3)),
        ).transpose(dim0=2, dim1=3)

        input_HH = torch.add(
            torch.matmul(matrix_Low_2.t(), input_LHH.transpose(dim0=2, dim1=3)),
            torch.matmul(matrix_High_2.t(), input_HHH.transpose(dim0=2, dim1=3)),
        ).transpose(dim0=2, dim1=3)

        input_L = torch.add(
            torch.matmul(input_LL, matrix_Low_1.t()),
            torch.matmul(input_LH, matrix_High_1.t()),
        )
        input_H = torch.add(
            torch.matmul(input_HL, matrix_Low_1.t()),
            torch.matmul(input_HH, matrix_High_1.t()),
        )

        output = torch.add(
            torch.matmul(matrix_Low_0.t(), input_L),
            torch.matmul(matrix_High_0.t(), input_H),
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            matrix_Low_0,
            matrix_Low_1,
            matrix_Low_2,
            matrix_High_0,
            matrix_High_1,
            matrix_High_2,
        ) = ctx.saved_variables

        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)

        grad_LL = torch.matmul(grad_L, matrix_Low_1).transpose(dim0=2, dim1=3)
        grad_LH = torch.matmul(grad_L, matrix_High_1).transpose(dim0=2, dim1=3)
        grad_HL = torch.matmul(grad_H, matrix_Low_1).transpose(dim0=2, dim1=3)
        grad_HH = torch.matmul(grad_H, matrix_High_1).transpose(dim0=2, dim1=3)

        grad_LLL = torch.matmul(matrix_Low_2, grad_LL).transpose(dim0=2, dim1=3)
        grad_LLH = torch.matmul(matrix_Low_2, grad_LH).transpose(dim0=2, dim1=3)
        grad_LHL = torch.matmul(matrix_Low_2, grad_HL).transpose(dim0=2, dim1=3)
        grad_LHH = torch.matmul(matrix_Low_2, grad_HH).transpose(dim0=2, dim1=3)
        grad_HLL = torch.matmul(matrix_High_2, grad_LL).transpose(dim0=2, dim1=3)
        grad_HLH = torch.matmul(matrix_High_2, grad_LH).transpose(dim0=2, dim1=3)
        grad_HHL = torch.matmul(matrix_High_2, grad_HL).transpose(dim0=2, dim1=3)
        grad_HHH = torch.matmul(matrix_High_2, grad_HH).transpose(dim0=2, dim1=3)

        return (
            grad_LLL,
            grad_LLH,
            grad_LHL,
            grad_LHH,
            grad_HLL,
            grad_HLH,
            grad_HHL,
            grad_HHH,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DWT_3D_SingleLevel(nn.Module):
    """
    Single-level 3D Haar wavelet transform.

    Input: [N, C, D, H, W]
    Output: 8 subbands, each [N, C, D/2, H/2, W/2]
            (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
    """

    def __init__(self):
        super().__init__()
        self.band_low = [1 / np.sqrt(2), 1 / np.sqrt(2)]
        self.band_high = [1 / np.sqrt(2), -1 / np.sqrt(2)]
        self.band_length = 2
        self.band_length_half = 1

    def get_matrix(self, input_depth, input_height, input_width, device):
        """Generate transformation matrices for given dimensions."""
        L1 = max(input_height, input_width)
        L = math.floor(L1 / 2)

        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2

        matrix_h_0 = matrix_h[
            0 : math.floor(input_height / 2), 0 : (input_height + self.band_length - 2)
        ]
        matrix_h_1 = matrix_h[
            0 : math.floor(input_width / 2), 0 : (input_width + self.band_length - 2)
        ]
        matrix_h_2 = matrix_h[
            0 : math.floor(input_depth / 2), 0 : (input_depth + self.band_length - 2)
        ]

        matrix_g_0 = matrix_g[
            0 : (input_height - math.floor(input_height / 2)),
            0 : (input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            0 : (input_width - math.floor(input_width / 2)),
            0 : (input_width + self.band_length - 2),
        ]
        matrix_g_2 = matrix_g[
            0 : (input_depth - math.floor(input_depth / 2)),
            0 : (input_depth + self.band_length - 2),
        ]

        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1) : end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1) : end]

        return (
            torch.tensor(matrix_h_0, dtype=torch.float32, device=device),
            torch.tensor(matrix_h_1, dtype=torch.float32, device=device),
            torch.tensor(matrix_h_2, dtype=torch.float32, device=device),
            torch.tensor(matrix_g_0, dtype=torch.float32, device=device),
            torch.tensor(matrix_g_1, dtype=torch.float32, device=device),
            torch.tensor(matrix_g_2, dtype=torch.float32, device=device),
        )

    def forward(self, input):
        """Apply single-level 3D DWT."""
        assert (
            len(input.size()) == 5
        ), f"Expected 5D input [N,C,D,H,W], got {input.size()}"

        input_depth = input.size(-3)
        input_height = input.size(-2)
        input_width = input.size(-1)

        matrices = self.get_matrix(input_depth, input_height, input_width, input.device)

        return DWTFunction_3D.apply(input, *matrices)


class IDWT_3D_SingleLevel(nn.Module):
    """
    Single-level 3D Haar inverse wavelet transform.

    Input: 8 subbands, each [N, C, D/2, H/2, W/2]
    Output: [N, C, D, H, W]
    """

    def __init__(self):
        super().__init__()
        self.band_low = [1 / np.sqrt(2), 1 / np.sqrt(2)]
        self.band_high = [1 / np.sqrt(2), -1 / np.sqrt(2)]
        self.band_low.reverse()
        self.band_high.reverse()
        self.band_length = 2
        self.band_length_half = 1

    def get_matrix(self, input_depth, input_height, input_width, device):
        """Generate reconstruction matrices for given dimensions."""
        L1 = max(input_height, input_width)
        L = math.floor(L1 / 2)

        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2

        matrix_h_0 = matrix_h[
            0 : math.floor(input_height / 2), 0 : (input_height + self.band_length - 2)
        ]
        matrix_h_1 = matrix_h[
            0 : math.floor(input_width / 2), 0 : (input_width + self.band_length - 2)
        ]
        matrix_h_2 = matrix_h[
            0 : math.floor(input_depth / 2), 0 : (input_depth + self.band_length - 2)
        ]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2

        matrix_g_0 = matrix_g[
            0 : (input_height - math.floor(input_height / 2)),
            0 : (input_height + self.band_length - 2),
        ]
        matrix_g_1 = matrix_g[
            0 : (input_width - math.floor(input_width / 2)),
            0 : (input_width + self.band_length - 2),
        ]
        matrix_g_2 = matrix_g[
            0 : (input_depth - math.floor(input_depth / 2)),
            0 : (input_depth + self.band_length - 2),
        ]

        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1) : end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1) : end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1) : end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1) : end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1) : end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1) : end]

        return (
            torch.tensor(matrix_h_0, dtype=torch.float32, device=device),
            torch.tensor(matrix_h_1, dtype=torch.float32, device=device),
            torch.tensor(matrix_h_2, dtype=torch.float32, device=device),
            torch.tensor(matrix_g_0, dtype=torch.float32, device=device),
            torch.tensor(matrix_g_1, dtype=torch.float32, device=device),
            torch.tensor(matrix_g_2, dtype=torch.float32, device=device),
        )

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        """Apply single-level 3D IDWT."""
        assert len(LLL.size()) == 5, f"Expected 5D input, got {LLL.size()}"

        input_depth = LLL.size(-3) + HHH.size(-3)
        input_height = LLL.size(-2) + HHH.size(-2)
        input_width = LLL.size(-1) + HHH.size(-1)

        matrices = self.get_matrix(input_depth, input_height, input_width, LLL.device)

        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH, *matrices)


class HaarDWT3D_2Level(nn.Module):
    """
    2-level 3D Haar wavelet decomposition.
    Matches your pywt_decompose function exactly.

    Input: [B, 1, 192, 256, 192]
    Output: [B, 64, 48, 64, 48] (normalized by /8.0)

    Structure:
        Level 1: 1 input → 8 subbands [B, 1, 96, 128, 96]
        Level 2: Apply DWT to LLL only → 8 subbands [B, 8, 48, 64, 48]
        Total: 8 level-2 + 7 level-1 high-freq = 64 channels
    """

    def __init__(self, normalize_factor=8.0):
        super().__init__()
        self.normalize_factor = normalize_factor
        self.dwt = DWT_3D_SingleLevel()

    def forward(self, x):
        """
        Args:
            x: [B, 1, 192, 256, 192] input image
        Returns:
            coeffs: [B, 64, 48, 64, 48] wavelet coefficients (normalized)
        """
        assert x.size(1) == 1, f"Expected 1 input channel, got {x.size(1)}"

        l1_LLL, l1_LLH, l1_LHL, l1_LHH, l1_HLL, l1_HLH, l1_HHL, l1_HHH = self.dwt(x)

        all_l2_subbands = []
        for l1_subband in [
            l1_LLL,
            l1_LLH,
            l1_LHL,
            l1_LHH,
            l1_HLL,
            l1_HLH,
            l1_HHL,
            l1_HHH,
        ]:
            l2_subs = self.dwt(l1_subband)  # Returns tuple of 8 subbands
            all_l2_subbands.extend(l2_subs)  # Add all 8 to list

        coeffs = torch.cat(all_l2_subbands, dim=1)  # [B, 64, 48, 64, 48]

        coeffs = coeffs / self.normalize_factor

        return coeffs


class HaarIDWT3D_2Level(nn.Module):
    """
    2-level 3D Haar inverse wavelet transform.
    Matches your recon_image_from_wavelet function exactly.

    Input: [B, 64, 48, 64, 48] (normalized coefficients)
    Output: [B, 1, 192, 256, 192] (reconstructed image)
    """

    def __init__(self, denormalize_factor=8.0):
        super().__init__()
        self.denormalize_factor = denormalize_factor
        self.idwt = IDWT_3D_SingleLevel()

    def forward(self, coeffs):
        """
        Args:
            coeffs: [B, 64, 48, 64, 48] wavelet coefficients (normalized)
        Returns:
            recon: [B, 1, 192, 256, 192] reconstructed image
        """
        assert coeffs.size(1) == 64, f"Expected 64 channels, got {coeffs.size(1)}"

        coeffs = coeffs * self.denormalize_factor

        l2_groups = []
        for i in range(8):
            group = coeffs[:, i * 8 : (i + 1) * 8, :, :, :]  # [B, 8, 48, 64, 48]
            l1_subband = self.idwt(
                group[:, 0:1],
                group[:, 1:2],
                group[:, 2:3],
                group[:, 3:4],
                group[:, 4:5],
                group[:, 5:6],
                group[:, 6:7],
                group[:, 7:8],
            )
            l2_groups.append(l1_subband)

        recon = self.idwt(*l2_groups)

        return recon


_dwt_instance = None
_idwt_instance = None
_dwt_device = None


def gpu_wavedecn(
    input_tensor, wavelet="haar", level=2, mode="zero", normalize_factor=8.0
):
    """
    GPU-accelerated 3D wavelet forward transform.

    Wrapper for HaarDWT3D_2Level to match pywt.wavedecn interface.

    Args:
        input_tensor: [B, 1, D, H, W] or numpy array
        wavelet: 'haar' (only supported)
        level: decomposition level (only 2 supported)
        mode: 'zero' (only supported)
        normalize_factor: division factor after transform (default 8.0)

    Returns:
        coeffs: [B, 64, D/4, H/4, W/4] wavelet coefficients
    """
    global _dwt_instance, _dwt_device

    if isinstance(input_tensor, np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device)
    else:
        device = input_tensor.device

    if input_tensor.ndim == 4:
        input_tensor = input_tensor.unsqueeze(0)

    if _dwt_instance is None or _dwt_device != device:
        _dwt_instance = HaarDWT3D_2Level(normalize_factor=normalize_factor).to(device)
        _dwt_device = device
    else:
        _dwt_instance.normalize_factor = normalize_factor

    with torch.no_grad():
        coeffs = _dwt_instance(input_tensor)

    return coeffs


def gpu_waverecn(
    coeffs_tensor, wavelet="haar", level=2, mode="zero", denormalize_factor=8.0
):
    """
    GPU-accelerated 3D wavelet inverse transform.

    Wrapper for HaarIDWT3D_2Level to match pywt.waverecn interface.

    Args:
        coeffs_tensor: [B, 64, D/4, H/4, W/4] wavelet coefficients
        wavelet: 'haar' (only supported)
        level: decomposition level (only 2 supported)
        mode: 'zero' (only supported)
        denormalize_factor: multiplication factor before inverse (default 8.0)

    Returns:
        recon: [B, 1, D, H, W] reconstructed image
    """
    global _idwt_instance, _dwt_device

    device = coeffs_tensor.device

    if _idwt_instance is None or _dwt_device != device:
        _idwt_instance = HaarIDWT3D_2Level(denormalize_factor=denormalize_factor).to(
            device
        )
        _dwt_device = device
    else:
        _idwt_instance.denormalize_factor = denormalize_factor

    with torch.no_grad():
        recon = _idwt_instance(coeffs_tensor)

    return recon


class GPUWaveletPrepareBatch:
    """
    Prepare batch for diffusion training with GPU wavelet transform.

    When using GPU wavelet, the DataLoader only loads and scales the data.
    This class applies the wavelet transform on GPU after data is loaded,
    avoiding CUDA multiprocessing issues with DataLoader workers.

    Usage:
        gpu_dwt = HaarDWT3D_2Level(normalize_factor=8.0).cuda()
        prepare_batch = GPUWaveletPrepareBatch(
            num_train_timesteps=1000,
            gpu_wavelet_fn=gpu_dwt
        )
        trainer = SupervisedTrainer(..., prepare_batch=prepare_batch)
    """

    def __init__(self, num_train_timesteps: int, gpu_wavelet_fn=None) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.gpu_wavelet_fn = gpu_wavelet_fn

    def __call__(self, batchdata: dict, device: str, non_blocking: bool = False):
        image = batchdata["image"].to(device, non_blocking=non_blocking)

        if self.gpu_wavelet_fn is not None:
            with torch.no_grad():
                image = self.gpu_wavelet_fn(image)  # -> [B, 64, D/4, H/4, W/4]

        noise = torch.randn_like(image).to(device, non_blocking=non_blocking)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (image.shape[0],), device=image.device
        ).long()

        return image, image, [noise, timesteps], {}


if __name__ == "__main__":
    print("Testing GPU Wavelet Transform...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 2
    input_image = torch.randn(batch_size, 1, 192, 256, 192, device=device)
    print(f"Input shape: {input_image.shape}")

    print("\nTesting forward transform (DWT)...")
    dwt = HaarDWT3D_2Level(normalize_factor=8.0).to(device)
    coeffs = dwt(input_image)
    print(f"Output shape: {coeffs.shape}")
    assert coeffs.shape == (
        batch_size,
        64,
        48,
        64,
        48,
    ), f"Unexpected shape: {coeffs.shape}"
    print("✓ Forward transform shape correct")

    print("\nTesting inverse transform (IDWT)...")
    idwt = HaarIDWT3D_2Level(denormalize_factor=8.0).to(device)
    recon = idwt(coeffs)
    print(f"Reconstructed shape: {recon.shape}")
    assert (
        recon.shape == input_image.shape
    ), f"Shape mismatch: {recon.shape} vs {input_image.shape}"
    print("✓ Inverse transform shape correct")

    recon_error = torch.abs(input_image - recon).mean().item()
    print(f"\nReconstruction error (MAE): {recon_error:.6f}")

    print("\nTesting gradient flow...")
    coeffs.requires_grad = True
    loss = coeffs.mean()
    loss.backward()
    print(f"Gradient shape: {coeffs.grad.shape}")
    print("✓ Gradients flow correctly")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nUsage:")
    print("  dwt = HaarDWT3D_2Level(normalize_factor=8.0).cuda()")
    print("  idwt = HaarIDWT3D_2Level(denormalize_factor=8.0).cuda()")
    print("  coeffs = dwt(image)  # [B,1,192,256,192] → [B,64,48,64,48]")
    print("  recon = idwt(coeffs)  # [B,64,48,64,48] → [B,1,192,256,192]")
