import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gpu_wavelet import HaarDWT3D_2Level
from paper_configs import (
    BETA_END,
    BETA_START,
    DISTILL_TAPS,
    NUM_TRAIN_TIMESTEPS,
    SCALE_CHANNELS,
    VOLUME_SHAPE,
    WAVELET_NORMALIZE,
    build_unet,
)

TIME_EMBED_DIM = 256

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher_ckpt", required=True, help="Pretrained teacher checkpoint (paper_232m profile)")
    p.add_argument("--data_path", required=True, help="Directory with preprocessed .npy volumes")
    p.add_argument("--split_json", required=True, help="JSON with a 'distillation' list of held-out files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_prefix", default="cleandift")
    p.add_argument("--teacher_scale", default="paper_232m", choices=list(SCALE_CHANNELS))
    p.add_argument("--student_scale", default="s23", choices=list(SCALE_CHANNELS))
    p.add_argument("--student_init", default="random", choices=["random", "copy"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr_peak", type=float, default=5e-7)
    p.add_argument("--lr_min", type=float, default=2e-7)
    p.add_argument("--warmup_epochs", type=float, default=2.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_epoch", type=int, default=None, help="Resume from cleandift_epoch_{n}.pt")
    p.add_argument("--max_steps", type=int, default=None, help="Stop after this many steps (smoke tests)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def robust_scale(volume: np.ndarray) -> np.ndarray:
    if not np.isfinite(volume).all():
        raise ValueError("non-finite voxels in input volume; exclude this file")
    p1, p99 = np.percentile(volume, [1, 99])
    if abs(p99 - p1) < 1e-6:
        raise ValueError("degenerate intensity range (flat volume); exclude this file")
    scaled = np.clip((volume - p1) / (p99 - p1), 0.0, 1.0)
    return (scaled * 2.0 - 1.0).astype(np.float32)

def pad_or_crop_center(volume: np.ndarray, target=VOLUME_SHAPE) -> np.ndarray:
    out = np.zeros(target, dtype=np.float32)
    src_slices, dst_slices = [], []
    for s, t in zip(volume.shape, target):
        n = min(s, t)
        s0 = (s - n) // 2
        d0 = (t - n) // 2
        src_slices.append(slice(s0, s0 + n))
        dst_slices.append(slice(d0, d0 + n))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out

class VolumeDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        vol = np.load(self.files[i]).astype(np.float32)
        vol = pad_or_crop_center(vol)
        vol = robust_scale(vol)
        return torch.from_numpy(vol)[None]

def distillation_files(data_path, split_json):
    with open(split_json) as f:
        split = json.load(f)
    files = []
    for f in split["distillation"]:
        candidate = os.path.join(data_path, os.path.basename(f))
        if os.path.exists(candidate):
            files.append(candidate)
    if not files:
        raise RuntimeError(f"No distillation files found under {data_path}")
    return files

def alpha_bar_table(device):
    betas = torch.linspace(BETA_START, BETA_END, NUM_TRAIN_TIMESTEPS, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0).to(device=device, dtype=torch.float32)

def add_noise(x0, noise, t, alpha_bar):
    ab = alpha_bar[t].view(-1, 1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

def sinusoidal_embedding(t, dim=TIME_EMBED_DIM):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)

def _groups(channels):
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1

class FiLMResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.film1 = nn.Linear(TIME_EMBED_DIM, 2 * channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.film2 = nn.Linear(TIME_EMBED_DIM, 2 * channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)

    @staticmethod
    def _film(x, emb, film):
        gamma, beta = film(emb).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]

    def forward(self, x, emb):
        h = self._film(self.norm1(x), emb, self.film1)
        h = self.conv1(F.silu(h))
        h = self._film(self.norm2(h), emb, self.film2)
        h = self.conv2(F.silu(h))
        return x + h

class ProjectionHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM), nn.SiLU(), nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM)
        )
        self.blocks = nn.ModuleList([FiLMResBlock(in_channels), FiLMResBlock(in_channels)])
        self.out = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        emb = self.time_mlp(sinusoidal_embedding(t))
        for block in self.blocks:
            x = block(x, emb)
        return self.out(x)

def register_tap_hooks(unet, storage):
    def make(name):
        def hook(_m, _i, out):
            storage[name] = out
        return hook

    unet.middle_block.register_forward_hook(make("middle"))
    unet.up_blocks[0].register_forward_hook(make("up0"))
    unet.up_blocks[1].register_forward_hook(make("up1"))

def learning_rate(epoch_float, args):
    if epoch_float < args.warmup_epochs:
        return args.lr_peak * epoch_float / args.warmup_epochs
    progress = (epoch_float - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1e-8)
    progress = min(max(progress, 0.0), 1.0)
    return args.lr_min + (args.lr_peak - args.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

def load_teacher(args, device):
    teacher = build_unet(args.teacher_scale).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher

def build_student(args, teacher, device):
    student = build_unet(args.student_scale).to(device)
    if args.student_init == "copy":
        student.load_state_dict(teacher.state_dict(), strict=True)
    return student

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "distill_log.jsonl")

    teacher = load_teacher(args, device)
    student = build_student(args, teacher, device)
    dwt = HaarDWT3D_2Level(normalize_factor=WAVELET_NORMALIZE).to(device)
    alpha_bar = alpha_bar_table(device)

    teacher_feats, student_feats = {}, {}
    register_tap_hooks(teacher, teacher_feats)
    register_tap_hooks(student, student_feats)

    heads = nn.ModuleDict(
        {
            "middle": ProjectionHead(SCALE_CHANNELS[args.student_scale][-1], SCALE_CHANNELS[args.teacher_scale][-1]),
            "up0": ProjectionHead(SCALE_CHANNELS[args.student_scale][-1], SCALE_CHANNELS[args.teacher_scale][-1]),
            "up1": ProjectionHead(SCALE_CHANNELS[args.student_scale][-2], SCALE_CHANNELS[args.teacher_scale][-2]),
        }
    ).to(device)

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(heads.parameters()), lr=args.lr_peak)

    start_epoch = 0
    if args.resume_epoch is not None:
        ck_path = os.path.join(args.output_dir, f"{args.model_prefix}_epoch_{args.resume_epoch}.pt")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["model_state_dict"], strict=True)
        heads.load_state_dict(ck["projector_state_dict"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from {ck_path} (next epoch {start_epoch})", flush=True)

    files = distillation_files(args.data_path, args.split_json)
    loader = DataLoader(
        VolumeDataset(files),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    steps_per_epoch = len(loader)
    print(f"Distillation files: {len(files)} | steps/epoch: {steps_per_epoch} | device: {device}", flush=True)

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        student.train()
        heads.train()
        epoch_loss, epoch_steps, t0 = 0.0, 0, time.time()
        for batch_idx, volumes in enumerate(loader):
            epoch_float = epoch + batch_idx / steps_per_epoch
            lr = learning_rate(epoch_float, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            volumes = volumes.to(device, non_blocking=True)
            with torch.no_grad():
                x0 = dwt(volumes)
            bsz = x0.shape[0]
            t = torch.randint(0, NUM_TRAIN_TIMESTEPS, (bsz,), device=device)
            noise = torch.randn_like(x0)
            xt = add_noise(x0, noise, t, alpha_bar)

            with torch.no_grad():
                teacher(xt, timesteps=t)
            t_zero = torch.zeros_like(t)
            student(x0, timesteps=t_zero)

            loss = 0.0
            for name in DISTILL_TAPS:
                projected = heads[name](student_feats[name], t)
                target = teacher_feats[name].detach()
                loss = loss - F.cosine_similarity(projected, target, dim=1).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite distillation loss at epoch {epoch} step {batch_idx}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            if batch_idx % 50 == 0:
                print(f"epoch {epoch + 1} step {batch_idx}/{steps_per_epoch} loss {loss.item():.6f} lr {lr:.3e}", flush=True)
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        mean_loss = epoch_loss / max(epoch_steps, 1)
        record = {
            "epoch": epoch + 1,
            "mean_loss": mean_loss,
            "lr_end": lr,
            "seconds": round(time.time() - t0, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch {epoch + 1} done: mean loss {mean_loss:.6f}", flush=True)

        ck_path = os.path.join(args.output_dir, f"{args.model_prefix}_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "projector_state_dict": heads.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "teacher_scale": args.teacher_scale,
                    "student_scale": args.student_scale,
                    "student_init": args.student_init,
                    "taps": list(DISTILL_TAPS),
                    "lr_peak": args.lr_peak,
                    "lr_min": args.lr_min,
                    "warmup_epochs": args.warmup_epochs,
                    "epochs": args.epochs,
                    "seed": args.seed,
                },
            },
            ck_path,
        )
        print(f"Checkpoint saved: {ck_path}", flush=True)

        if args.max_steps is not None and global_step >= args.max_steps:
            break

if __name__ == "__main__":
    main()
