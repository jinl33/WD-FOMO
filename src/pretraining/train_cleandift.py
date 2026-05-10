import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import gc
import math
import pywt
import monai
from tqdm import tqdm
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
import utils

CHANNELS = 64
LEVELS = 2
INITIAL_SIZE = [192, 256, 192]
TARGET_SIZE = [48, 64, 48]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NumpyLoader(monai.data.ImageReader):
    def read(self, data, **kwargs):
        img = np.load(data).astype(np.float32)
        return img, {"spatial_shape": img.shape}

    def get_data(self, img):
        return img[0], img[1]

    def verify_suffix(self, filename):
        return filename.endswith(".npy")


def pywt_decompose(x):
    packet = pywt.WaveletPacketND(
        x.get_array(), "haar", axes=(-3, -2, -1), maxlevel=LEVELS
    )
    x.set_array(
        torch.tensor(
            np.concatenate([y.data.view() for y in packet.get_level(LEVELS)], axis=0)
            / 8.0
        )
    )
    gc.collect()
    return x


lambd_pywt = monai.transforms.Lambdad(keys=["image"], func=pywt_decompose)

train_transforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(
            keys=["image"], reader=NumpyLoader(), ensure_channel_first=True
        ),
        monai.transforms.ResizeWithPadOrCropd(
            keys=["image"], spatial_size=INITIAL_SIZE
        ),
        monai.transforms.ScaleIntensityRangePercentilesd(
            keys="image", lower=1, upper=99, b_min=-1, b_max=1, clip=True
        ),
        lambd_pywt,
        monai.transforms.ToTensord(keys=["image"], track_meta=False),
    ]
)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ProjectionHead(nn.Module):
    def __init__(self, feature_dim, time_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, feature_dim),
        )
        self.projector = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=1),
            nn.GroupNorm(32, feature_dim),
            nn.SiLU(),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=1),
            nn.GroupNorm(32, feature_dim),
            nn.SiLU(),
            nn.Conv3d(feature_dim, feature_dim, kernel_size=1),
        )
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)  # [B, C]
        t_emb = t_emb[:, :, None, None, None]
        h = x + t_emb
        return self.projector(h)


class CleanDIFTDistiller(nn.Module):
    def __init__(self, teacher_checkpoint):
        super().__init__()
        print(f"Initializing Multi-Layer CleanDIFT with Teacher: {teacher_checkpoint}")
        self.teacher = self._get_unet()
        self.student = self._get_unet()

        self.projectors = nn.ModuleDict(
            {
                "middle": ProjectionHead(feature_dim=512),
                "up1": ProjectionHead(feature_dim=256),
            }
        )

        ckpt = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        self.teacher.load_state_dict(state_dict)
        self.student.load_state_dict(state_dict)

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.teacher_feats = {}
        self.student_feats = {}

        self._register_hooks(self.teacher, self.teacher_feats)
        self._register_hooks(self.student, self.student_feats)

    def _get_unet(self):
        return DiffusionModelUNet(
            spatial_dims=3,
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            num_channels=[32, 64, 128, 256],
            attention_levels=[False, False, True, True],
            num_head_channels=[0, 0, 32, 32],
            num_res_blocks=2,
            use_flash_attention=False,
            with_conditioning=False,
        )

    def _register_hooks(self, model, storage):
        """Attach hooks to target layers"""

        def hook_mid(module, input, output):
            storage["middle"] = output

        model.middle_block.register_forward_hook(hook_mid)

        def hook_up1(module, input, output):
            storage["up1"] = output

        model.up_blocks[1].register_forward_hook(hook_up1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default="./cleandift_multilayer_checkpoints"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_files, _ = utils.get_fomo_data(args.data_path, pattern="*t1*.npy")
    train_ds = monai.data.Dataset(
        data=[{"image": f} for f in train_files], transform=train_transforms
    )
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    distiller = CleanDIFTDistiller(args.teacher_ckpt).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(distiller.student.parameters()) + list(distiller.projectors.parameters()),
        lr=args.lr,
    )

    scaler = torch.amp.GradScaler("cuda")

    criterion = nn.MSELoss()

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    print(f"Starting Multi-Layer Distillation for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        distiller.student.train()
        for p in distiller.projectors.values():
            p.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            clean_images = batch["image"].to(DEVICE)
            bs = clean_images.shape[0]

            timesteps = torch.randint(0, 1000, (bs,), device=DEVICE).long()
            noise = torch.randn_like(clean_images).to(DEVICE)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    distiller.teacher(noisy_images, timesteps=timesteps)

                t_zeros = torch.zeros_like(timesteps)
                distiller.student(clean_images, timesteps=t_zeros)

                total_loss = 0
                for layer_name, projector in distiller.projectors.items():
                    student_feat = distiller.student_feats[layer_name]
                    teacher_feat = distiller.teacher_feats[layer_name].detach()

                    projected_feat = projector(student_feat, timesteps)

                    total_loss += criterion(projected_feat, teacher_feat)

            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(distiller.student.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": f"{total_loss.item():.6f}"})

        save_path = os.path.join(
            args.output_dir, f"cleandift_student_epoch_{epoch+1}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": distiller.student.state_dict(),
                "projector_state_dict": {
                    k: v.state_dict() for k, v in distiller.projectors.items()
                },
            },
            save_path,
        )
        print(f"Checkpoint saved: {save_path}")


if __name__ == "__main__":
    main()
