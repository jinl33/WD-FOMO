import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src" / "downstream"))

import paper_configs as pc
import train_cleandift as td
from gpu_wavelet import HaarDWT3D_2Level
from models.networks.cleandift_notime import strip_time_conditioning

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True)
    p.add_argument("--split_json", required=True, help="JSON with a 'pretrain' list (distillation files excluded)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_prefix", default="masked")
    p.add_argument("--scale", default="s23", choices=list(pc.SCALE_CHANNELS))
    p.add_argument("--mask_ratio", type=float, default=0.6)
    p.add_argument("--mask_patch_size", type=int, default=1, help="Mask block side on the wavelet grid")
    p.add_argument("--mask_token", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def build_time_independent_unet(scale: str, device):
    net = pc.build_unet(scale)
    removed = strip_time_conditioning(net)
    print(f"Removed {removed:,} time-pathway parameters from the {scale} U-Net", flush=True)
    return net.to(device)

def sample_mask(batch, shape, ratio, patch, device):
    d, h, w = shape
    gd, gh, gw = -(-d // patch), -(-h // patch), -(-w // patch)
    coarse = torch.rand(batch, 1, gd, gh, gw, device=device) < ratio
    mask = F.interpolate(coarse.float(), scale_factor=patch, mode="nearest")
    return mask[:, :, :d, :h, :w]

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split_json) as f:
        split = json.load(f)
    files = []
    for name in split["pretrain"]:
        candidate = os.path.join(args.data_path, os.path.basename(name))
        if os.path.exists(candidate):
            files.append(candidate)
    loader = DataLoader(
        td.VolumeDataset(files),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    net = build_time_independent_unet(args.scale, device)
    dwt = HaarDWT3D_2Level(normalize_factor=pc.WAVELET_NORMALIZE).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    log_path = os.path.join(args.output_dir, "masked_log.jsonl")

    global_step = 0
    for epoch in range(args.epochs):
        net.train()
        losses, t0 = [], time.time()
        for volumes in loader:
            with torch.no_grad():
                x0 = dwt(volumes.to(device))
                mask = sample_mask(x0.shape[0], pc.WAVELET_SHAPE, args.mask_ratio, args.mask_patch_size, device)
                x_in = x0 * (1 - mask) + args.mask_token * mask
            zeros = torch.zeros(x0.shape[0], dtype=torch.long, device=device)
            pred = net(x_in, timesteps=zeros)
            sq = (pred - x0) ** 2 * mask
            loss = sq.sum() / (mask.sum() * pc.WAVELET_CHANNELS).clamp_min(1.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite masked-reconstruction loss at epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        record = {"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(record, flush=True)

        ck = os.path.join(args.output_dir, f"{args.model_prefix}_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "objective": "masked_reconstruction",
                    "scale": args.scale,
                    "mask_ratio": args.mask_ratio,
                    "mask_patch_size": args.mask_patch_size,
                    "mask_token": args.mask_token,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "seed": args.seed,
                },
            },
            ck,
        )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

if __name__ == "__main__":
    main()
