from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import Inferer
from monai.utils import optional_import
from functools import partial
import glob
import os
import random

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


def get_fomo_data(
    data_path,
    pattern="*t1*skull_stripped.npy",
    min_volume=1e6,
    min_dim=60,
    split_ratios=(0.8, 0.2),
):
    """Load FOMO T1 images with volume filtering"""
    image_files = glob.glob(os.path.join(data_path, pattern))

    valid_files = []
    rejected_truncated = 0
    rejected_small = 0
    errors = 0

    dimensions_count = {}

    for f in image_files:
        try:
            data = np.load(f, mmap_mode="r")
            volume = np.prod(data.shape)

            dim_key = str(data.shape)
            if dim_key in dimensions_count:
                dimensions_count[dim_key] += 1
            else:
                dimensions_count[dim_key] = 1

            if volume < min_volume:
                rejected_small += 1
                continue

            if any(dim < min_dim for dim in data.shape):
                rejected_truncated += 1
                continue

            valid_files.append(f)
        except Exception as e:
            errors += 1
            continue

    if dimensions_count:
        print("Most common dimensions:")
        sorted_dims = sorted(dimensions_count.items(), key=lambda x: x[1], reverse=True)
        for dim, count in sorted_dims[:5]:  # Top 5 most common dimensions
            print(f"  {dim}: {count} files ({count/len(image_files)*100:.1f}%)")

    random.Random(42).shuffle(valid_files)

    n_train = int(len(valid_files) * split_ratios[0])

    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:]

    print(f"Valid files: {len(valid_files)}")
    print(
        f"Rejected files: {rejected_truncated} truncated, {rejected_small} small, {errors} errors"
    )
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    return train_files, val_files


class SamplePredictionPrepareBatch:
    """
    Prepare batch for unconditional diffusion training.
    Matches original implementation - prediction_type="sample" means target is the original image.
    """

    def __init__(self, num_train_timesteps: int) -> None:
        self.num_train_timesteps = num_train_timesteps

    def __call__(self, batchdata: dict, device: str, non_blocking: bool = False):
        image = batchdata["image"].to(device, non_blocking=non_blocking)
        noise = torch.randn_like(image).to(device, non_blocking=non_blocking)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (image.shape[0],), device=device
        ).long()

        return image, image, [noise, timesteps], {}


class UnconditionalDiffusionInferer(Inferer):
    """
    Unconditional diffusion inferer matching the original paper implementation.
    No conditioning - pure unconditional generation.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: callable,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            inputs: Original images
            diffusion_model: UNet model
            noise: Random noise
            timesteps: Training timesteps

        Returns:
            Model prediction (x_0 when prediction_type="sample")
        """
        noisy_image = self.scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps
        )

        prediction = diffusion_model(x=noisy_image, timesteps=timesteps)

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: callable,
        scheduler: callable | None = None,
        save_intermediates: bool = False,
        intermediate_steps: int = 100,
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Sample from the model using DDIM or DDPM.

        Args:
            input_noise: Random noise to start from
            diffusion_model: Trained UNet
            scheduler: DDIM or DDPM scheduler
            save_intermediates: Whether to save intermediate steps
            intermediate_steps: Frequency of saving intermediates
            verbose: Show progress bar

        Returns:
            Generated sample (and intermediates if requested)
        """
        if not scheduler:
            scheduler = self.scheduler

        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)

        intermediates = []
        debug_first_step = True
        for t in progress_bar:
            model_output = diffusion_model(
                image, timesteps=torch.Tensor((t,)).to(input_noise.device)
            )

            if debug_first_step:
                print(f"  [DEBUG] First step t={t.item()}")
                print(
                    f"    Input image: min={image.min().item():.6f}, max={image.max().item():.6f}, mean={image.mean().item():.6f}"
                )
                print(
                    f"    Model output: min={model_output.min().item():.6f}, max={model_output.max().item():.6f}, mean={model_output.mean().item():.6f}"
                )
                debug_first_step = False

            image, _ = scheduler.step(model_output, t, image)

            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)

        if save_intermediates:
            return image, intermediates
        else:
            return image


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    temp_filename = f"{filename}.tmp"
    torch.save(checkpoint, temp_filename)
    os.replace(temp_filename, filename)

    print(f"Checkpoint saved to {filename}")
    return filename


def load_checkpoint(model, optimizer=None, scheduler=None, filename=None):
    """Load checkpoint for resuming training"""
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return 0, float("inf")

    checkpoint = torch.load(filename, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, loss
