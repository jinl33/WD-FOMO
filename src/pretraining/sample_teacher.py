import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paper_configs as pc
from gpu_wavelet import HaarIDWT3D_2Level

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--scale", default="paper_232m", choices=list(pc.SCALE_CHANNELS))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def ddpm_posterior_step(model, x_t, t_int, alpha_bar, betas):
    t_batch = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    x0_hat = model(x_t, timesteps=t_batch)

    ab_t = alpha_bar[t_int]
    ab_prev = alpha_bar[t_int - 1] if t_int > 0 else torch.tensor(1.0, device=x_t.device)
    beta_t = betas[t_int]
    alpha_t = 1.0 - beta_t

    coef_x0 = ab_prev.sqrt() * beta_t / (1.0 - ab_t)
    coef_xt = alpha_t.sqrt() * (1.0 - ab_prev) / (1.0 - ab_t)
    mean = coef_x0 * x0_hat + coef_xt * x_t
    if t_int == 0:
        return mean, x0_hat
    var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t)
    return mean + var.sqrt() * torch.randn_like(x_t), x0_hat

def save_views(volume: np.ndarray, path_png: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    d, h, w = volume.shape
    views = [volume[d // 2], volume[:, h // 2, :], volume[:, :, w // 2]]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, name in zip(axes, views, ("axial", "coronal", "sagittal")):
        ax.imshow(img.T, cmap="gray", origin="lower")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path_png, dpi=120)
    plt.close(fig)

def main():
    print(
        "WARNING: HaarIDWT3D_2Level does not currently invert the forward wavelet transform "
        "(docs/PRE_PUSH_REVIEW.md item 4; documented round-trip max abs error ~5.5 on unit-scale "
        "input). The images this run writes are NOT verified to be the correct inverse of the "
        "sampled coefficients. Fix the subband-ordering bug before trusting this output.",
        flush=True,
    )
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model = pc.build_unet(args.scale).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=True)
    model.eval()

    betas = torch.linspace(pc.BETA_START, pc.BETA_END, pc.NUM_TRAIN_TIMESTEPS, dtype=torch.float64).to(device)
    betas = betas.float()
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    idwt = HaarIDWT3D_2Level(denormalize_factor=pc.WAVELET_NORMALIZE).to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    for i in range(args.n_samples):
        x_t = torch.randn((1, pc.WAVELET_CHANNELS, *pc.WAVELET_SHAPE), generator=gen, device=device)
        with torch.no_grad():
            for t_int in reversed(range(pc.NUM_TRAIN_TIMESTEPS)):
                x_t, x0_hat = ddpm_posterior_step(model, x_t, t_int, alpha_bar, betas)
            image = idwt(x_t)
        volume = image[0, 0].float().cpu().numpy()
        np.save(os.path.join(args.output_dir, f"sample_{i}.npy"), volume)
        save_views(volume, os.path.join(args.output_dir, f"sample_{i}.png"))
        print(f"sample {i}: shape {volume.shape}, min {volume.min():.3f}, max {volume.max():.3f}", flush=True)

if __name__ == "__main__":
    main()
