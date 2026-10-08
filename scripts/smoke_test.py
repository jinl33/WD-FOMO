#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/downstream", "src/pretraining"):
    path = str(REPO_ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

DEFAULT_SHAPE = (32, 32, 32)
ALIGN = 64

def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")

def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")

def _pad_to_align(vol, align: int = ALIGN):
    import torch.nn.functional as F

    d, h, w = vol.shape[-3:]
    pads = []
    for size in (w, h, d):
        target = ((size + align - 1) // align) * align
        pads.extend([0, target - size])
    return F.pad(vol, pads)

def stage_dwpt(volume_path: str | None, tolerance: float, device_str: str) -> bool:
    print("\n[stage: dwpt] wavelet transform: shape + Parseval energy preservation")
    import torch

    try:
        from gpu_wavelet import HaarDWT3D_2Level
    except ImportError as exc:
        _fail(f"cannot import gpu_wavelet ({exc}); check PYTHONPATH / environment")
        return False

    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")

    if volume_path:
        import numpy as np

        arr = np.load(volume_path)
        if arr.ndim == 4:
            print(f"  note  input has {arr.shape[0]} channels; using channel 0")
            arr = arr[0]
        if arr.ndim != 3:
            _fail(f"expected a 3D or 4D array, got shape {arr.shape}")
            return False
        vol = torch.from_numpy(arr).float()[None, None]
        print(f"  input  {volume_path}  shape {tuple(arr.shape)}")
    else:
        vol = torch.randn(1, 1, *DEFAULT_SHAPE)
        print(f"  input  synthetic random volume, shape {DEFAULT_SHAPE}")

    vol = _pad_to_align(vol).to(device)
    print(f"  padded to {tuple(vol.shape[-3:])} (multiple of {ALIGN}); device {device}")
    if device.type == "cpu" and vol.shape[-1] > 64:
        print("  note  this transform is matrix-based and slow on CPU at full volume size;")
        print("        pass --device cuda inside a GPU job for realistic volumes.")

    nf = 8.0
    dwt = HaarDWT3D_2Level(normalize_factor=nf).to(device)
    with torch.no_grad():
        coeffs = dwt(vol)

    passed = True

    expected_spatial = tuple(s // 4 for s in vol.shape[-3:])
    actual_spatial = tuple(coeffs.shape[-3:])
    if coeffs.shape[1] == 64 and actual_spatial == expected_spatial:
        _ok(f"decomposition {tuple(vol.shape[-3:])} -> {coeffs.shape[1]} ch x {actual_spatial}")
    else:
        _fail(
            f"expected 64 channels at {expected_spatial}, "
            f"got {coeffs.shape[1]} channels at {actual_spatial}"
        )
        passed = False

    e_in = (vol.double() ** 2).sum().item()
    e_out = ((coeffs.double() * nf) ** 2).sum().item()
    if e_in == 0:
        _fail("input volume has zero energy; cannot test preservation")
        return False
    ratio = e_out / e_in
    if abs(ratio - 1.0) < tolerance:
        _ok(f"Parseval energy preserved: ratio {ratio:.8f} (|1 - ratio| < {tolerance:g}) -- transform is lossless")
    else:
        _fail(
            f"energy ratio {ratio:.8f} deviates from 1 by more than {tolerance:g} -- the front end is NOT "
            "information-preserving on this data; every downstream result would inherit the loss"
        )
        passed = False

    n_in = vol.numel()
    n_out = coeffs.numel()
    if n_in == n_out:
        _ok(f"element count preserved ({n_in:,}) -- a rearrangement, not a compression")
    else:
        _fail(f"element count changed: {n_in:,} -> {n_out:,}")
        passed = False

    return passed

def stage_model(scale: str) -> bool:
    print("\n[stage: model] instantiation, parameter count, attention head partition")
    import torch

    try:
        from models.networks.cleandift import CleanDIFTBase
    except ImportError as exc:
        _fail(f"cannot import the model ({exc}); check PYTHONPATH / environment")
        return False

    profiles = {
        "s23": [40, 40, 80, 80, 160],
        "s33": [48, 48, 96, 96, 192],
    }
    if scale not in profiles:
        _fail(f"unknown scale {scale!r}; choose from {sorted(profiles)}")
        return False
    widths = profiles[scale]

    passed = True
    counts = {}
    for head_channels in ([0, 0, 0, 16, 16], [0, 0, 0, 32, 32]):
        try:
            model = CleanDIFTBase(
                num_modalities=1,
                backbone_num_channels=widths,
                backbone_num_res_blocks=2,
                backbone_num_head_channels=head_channels,
            )
            backbone = model.backbone
        except TypeError as exc:
            _fail(f"could not construct CleanDIFTBase with explicit head channels: {exc}")
            return False
        except Exception as exc:
            _fail(f"model construction failed: {exc}")
            return False

        n_params = sum(p.numel() for p in backbone.parameters())
        counts[tuple(head_channels)] = n_params
        heads = [c // hc if hc else 0 for c, hc in zip(widths, head_channels)]
        print(f"  head_channels={head_channels}  params={n_params:,}  heads per level={heads}")

    _ok(f"backbone '{scale}' instantiated, widths {widths}")

    distinct = set(counts.values())
    if len(distinct) == 1:
        _ok(
            f"both head partitions give the SAME parameter count ({distinct.pop():,}) -- "
            "this is why a mismatch loads silently under strict=True"
        )
        print(
            "  ----> ACTION: pin num_head_channels explicitly in your config and assert it matches\n"
            "        the value used at pretraining. Nothing downstream can detect a mismatch.\n"
            "        See docs/architecture.md (footgun) and docs/porting_to_new_modality.md 1b."
        )
    else:
        _fail(f"parameter counts differ between head partitions: {counts}")
        passed = False

    return passed

def stage_train(steps: int, scale: str, wavelet_size: int) -> bool:
    print(f"\n[stage: train] {steps} optimizer steps on synthetic data")
    import torch

    try:
        from models.networks.cleandift import CleanDIFTBase
    except ImportError as exc:
        _fail(f"cannot import the model ({exc})")
        return False

    widths = {"s23": [40, 40, 80, 80, 160], "s33": [48, 48, 96, 96, 192]}[scale]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device {device}")

    wrapper = CleanDIFTBase(
        num_modalities=1,
        backbone_num_channels=widths,
        backbone_num_res_blocks=2,
        backbone_num_head_channels=[0, 0, 0, 16, 16],
    )
    model = wrapper.backbone.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    x = torch.randn(1, 64, wavelet_size, wavelet_size, wavelet_size, device=device)
    target = torch.randn_like(x)

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        t = torch.zeros(x.shape[0], device=device).long()
        out = model(x, t)
        loss = torch.nn.functional.mse_loss(out, target)
        if not torch.isfinite(loss):
            _fail(f"loss became non-finite at step {step} -- check the data for non-finite values")
            return False
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step}  loss {loss.item():.6f}")

    _ok(f"{steps} steps completed, all losses finite (first {losses[0]:.6f} -> last {losses[-1]:.6f})")
    if torch.cuda.is_available():
        print(f"  peak GPU memory reserved: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    print(
        "  note  this only proves the training path runs. It does not validate the objective:\n"
        "        the real pipeline trains with prediction_type='sample' (predict the clean signal),\n"
        "        NOT epsilon-prediction. See docs/architecture.md 3."
    )
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["dwpt", "model", "train", "all"], default="all")
    parser.add_argument("--volume", default=None, help="path to a .npy volume for the dwpt round-trip")
    parser.add_argument("--steps", type=int, default=5, help="optimizer steps for --stage train")
    parser.add_argument("--scale", default="s23", choices=["s23", "s33"])
    parser.add_argument("--tolerance", type=float, default=1e-6, help="tolerance on the Parseval energy ratio")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device for the dwpt stage")
    parser.add_argument("--wavelet-size", type=int, default=8, help="cube size in the wavelet domain for --stage train")
    args = parser.parse_args()

    print("=" * 78)
    print("WD-FOMO smoke test")
    print("=" * 78)

    results = {}
    if args.stage in ("dwpt", "all"):
        results["dwpt"] = stage_dwpt(args.volume, args.tolerance, args.device)
    if args.stage in ("model", "all"):
        results["model"] = stage_model(args.scale)
    if args.stage in ("train", "all"):
        results["train"] = stage_train(args.steps, args.scale, args.wavelet_size)

    print("\n" + "=" * 78)
    for name, passed in results.items():
        print(f"  {name:6s}  {'PASS' if passed else 'FAIL'}")
    all_passed = all(results.values())
    print(f"  overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 78)
    return 0 if all_passed else 1

if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
