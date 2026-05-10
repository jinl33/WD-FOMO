#!/usr/bin/env python3
"""Evaluate one Task10 segmentation fold from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--fold_idx", type=int, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--patch_size_dhw", type=int, nargs=3, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    return p.parse_args()


def load_case(npy_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.load(npy_path, allow_pickle=True)
    n_channels = len(raw) - 1
    image = np.stack(
        [np.asarray(raw[i]).astype(np.float32) for i in range(n_channels)], axis=0
    )
    label = np.asarray(raw[-1]).astype(np.uint8)
    image = image.transpose(0, 3, 1, 2).copy()  # (C, D, H, W)
    label = label.transpose(2, 0, 1).copy()  # (D, H, W)
    return image, label


def pad_to_patch(
    image: torch.Tensor, patch_size: Tuple[int, int, int]
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _, d, h, w = image.shape
    pd, ph, pw = patch_size
    orig = (d, h, w)
    pad = []
    for size, target in zip(reversed((d, h, w)), reversed((pd, ph, pw))):
        before = max(0, (target - size) // 2)
        after = max(0, target - size - before)
        pad.extend([before, after])
    if any(p > 0 for p in pad):
        image = F.pad(image, pad, mode="constant", value=0.0)
    return image, orig


def crop_back(arr: np.ndarray, orig_shape: Tuple[int, int, int]) -> np.ndarray:
    d, h, w = arr.shape
    od, oh, ow = orig_shape
    sd = max(0, (d - od) // 2)
    sh = max(0, (h - oh) // 2)
    sw = max(0, (w - ow) // 2)
    return arr[sd : sd + od, sh : sh + oh, sw : sw + ow]


def dice_score(pred_fg: np.ndarray, gt_fg: np.ndarray) -> float:
    pred_fg = pred_fg.astype(bool)
    gt_fg = gt_fg.astype(bool)
    if not pred_fg.any() and not gt_fg.any():
        return 1.0
    if not pred_fg.any() or not gt_fg.any():
        return 0.0
    inter = int(np.sum(pred_fg & gt_fg))
    denom = int(np.sum(pred_fg) + np.sum(gt_fg))
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def load_model(checkpoint: Path):
    from models.supervised_seg import SupervisedSegModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedSegModel.load_from_checkpoint(checkpoint_path=str(checkpoint))
    model.eval().to(device)
    return model, device


def predict_seg(
    model, device, image_np: np.ndarray, patch_size: Tuple[int, int, int]
) -> np.ndarray:
    tensor = torch.from_numpy(image_np).float()
    tensor, orig_shape = pad_to_patch(tensor, patch_size)
    with torch.no_grad():
        pred = model.model.predict(
            data=tensor.unsqueeze(0).to(device),
            mode="3D",
            mirror=False,
            overlap=0.5,
            patch_size=patch_size,
            sliding_window_prediction=True,
        )

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)

    if pred.ndim == 5 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim == 4:
        if pred.shape[0] > 1:
            pred_seg = np.argmax(pred, axis=0).astype(np.uint8)
        else:
            pred_seg = (pred[0] > 0.5).astype(np.uint8)
    elif pred.ndim == 3:
        pred_seg = (pred > 0).astype(np.uint8)
    else:
        raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

    return crop_back(pred_seg, orig_shape)


def main() -> None:
    args = parse_args()

    splits = pickle.load(open(args.task_dir / "splits.pkl", "rb"))
    folds = splits["kfold"][args.num_folds]
    fold = folds[args.fold_idx]
    val_ids = sorted(fold["val"])
    patch_size = tuple(args.patch_size_dhw)

    model, device = load_model(args.checkpoint)

    results: List[Dict[str, float]] = []
    errors: List[Dict[str, str]] = []
    for idx, fid in enumerate(val_ids, 1):
        npy_path = args.task_dir / f"{fid}.npy"
        if not npy_path.exists():
            errors.append({"id": fid, "reason": "missing_npy"})
            continue
        try:
            image_np, label_np = load_case(npy_path)
            pred_seg = predict_seg(model, device, image_np, patch_size)
            gt_fg = label_np > 0
            pred_fg = pred_seg > 0
            dice = dice_score(pred_fg, gt_fg)
            results.append(
                {
                    "id": fid,
                    "dice": float(dice),
                    "gt_voxels": int(np.sum(gt_fg)),
                    "pred_voxels": int(np.sum(pred_fg)),
                }
            )
            if idx % 10 == 0 or idx == len(val_ids):
                print(f"[{idx}/{len(val_ids)}] {fid}: dice={dice:.4f}", flush=True)
        except Exception as exc:
            errors.append({"id": fid, "reason": str(exc)})

    dices = (
        np.array([r["dice"] for r in results], dtype=float)
        if results
        else np.array([], dtype=float)
    )
    payload = {
        "model_name": args.model_name,
        "fold_idx": args.fold_idx,
        "n_subjects": int(len(results)),
        "mean_dice": float(np.mean(dices)) if len(dices) else float("nan"),
        "std_dice": float(np.std(dices)) if len(dices) else float("nan"),
        "median_dice": float(np.median(dices)) if len(dices) else float("nan"),
        "checkpoint": str(args.checkpoint),
        "task_dir": str(args.task_dir),
        "predictions": results,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                k: payload[k]
                for k in ("model_name", "fold_idx", "n_subjects", "mean_dice")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
