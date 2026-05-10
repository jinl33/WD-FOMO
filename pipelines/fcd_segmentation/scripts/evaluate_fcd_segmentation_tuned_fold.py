#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--fold_idx", type=int, required=True)
    p.add_argument("--split_method", type=str, default="nested_kfold")
    p.add_argument("--split_param", type=int, default=5)
    p.add_argument("--patch_size_dhw", type=int, nargs=3, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--thresholds", type=float, nargs="*", default=None)
    p.add_argument("--min_component_voxels", type=int, nargs="*", default=None)
    p.add_argument("--keep_largest_options", type=int, nargs="*", default=(0, 1))
    return p.parse_args()


def default_thresholds() -> List[float]:
    return [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]


def default_min_components() -> List[int]:
    return [0, 16, 32, 64, 128, 256, 512, 1024]


def load_case(npy_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.load(npy_path, allow_pickle=True)
    n_channels = len(raw) - 1
    image = np.stack([np.asarray(raw[i]).astype(np.float32) for i in range(n_channels)], axis=0)
    label = np.asarray(raw[-1]).astype(np.uint8)
    image = image.transpose(0, 3, 1, 2).copy()
    label = label.transpose(2, 0, 1).copy()
    return image, label


def pad_to_patch(image: torch.Tensor, patch_size: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
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
    return arr[sd:sd + od, sh:sh + oh, sw:sw + ow]


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


def predict_fg_prob(
    model,
    device,
    image_np: np.ndarray,
    patch_size: Tuple[int, int, int],
    mirror: bool,
) -> np.ndarray:
    tensor = torch.from_numpy(image_np).float()
    tensor, orig_shape = pad_to_patch(tensor, patch_size)
    with torch.no_grad():
        pred = model.model.predict(
            data=tensor.unsqueeze(0).to(device),
            mode="3D",
            mirror=mirror,
            overlap=0.5,
            patch_size=patch_size,
            sliding_window_prediction=True,
        )

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred)

    if pred.ndim == 5 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim == 4:
        if pred.shape[0] > 1:
            scores = pred - pred.max(axis=0, keepdims=True)
            probs = np.exp(scores)
            probs /= np.clip(probs.sum(axis=0, keepdims=True), 1e-12, None)
            fg_prob = probs[1]
        else:
            fg_prob = 1.0 / (1.0 + np.exp(-pred[0]))
    elif pred.ndim == 3:
        fg_prob = 1.0 / (1.0 + np.exp(-pred))
    else:
        raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")

    return crop_back(fg_prob.astype(np.float32), orig_shape)


def connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num = ndimage.label(mask.astype(np.uint8), structure=structure)
    return labeled, int(num)


def postprocess_mask(prob: np.ndarray, threshold: float, min_component_voxels: int, keep_largest: bool) -> np.ndarray:
    mask = prob > threshold
    if not mask.any():
        return mask.astype(np.uint8)

    labeled, num = connected_components(mask)
    if num == 0:
        return mask.astype(np.uint8)

    counts = np.bincount(labeled.ravel())
    keep = np.ones_like(counts, dtype=bool)
    keep[0] = False

    if min_component_voxels > 0:
        keep &= counts >= int(min_component_voxels)

    if keep_largest:
        valid = np.where(keep)[0]
        if len(valid) == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        largest = valid[np.argmax(counts[valid])]
        keep = np.zeros_like(counts, dtype=bool)
        keep[largest] = True

    out = keep[labeled]
    return out.astype(np.uint8)


def collect_predictions(
    model,
    device,
    task_dir: Path,
    case_ids: Sequence[str],
    patch_size: Tuple[int, int, int],
    mirror: bool,
) -> List[Dict]:
    rows = []
    for idx, fid in enumerate(case_ids, 1):
        image_np, label_np = load_case(task_dir / f"{fid}.npy")
        fg_prob = predict_fg_prob(model, device, image_np, patch_size, mirror=mirror)
        rows.append(
            {
                "id": fid,
                "fg_prob": fg_prob,
                "gt_fg": (label_np > 0).astype(np.uint8),
            }
        )
        if idx % 5 == 0 or idx == len(case_ids):
            print(f"[predict {idx}/{len(case_ids)}] {fid}", flush=True)
    return rows


def evaluate_rows(rows: Sequence[Dict], threshold: float, min_component_voxels: int, keep_largest: bool) -> Dict:
    preds = []
    for row in rows:
        pred_fg = postprocess_mask(row["fg_prob"], threshold, min_component_voxels, keep_largest)
        gt_fg = row["gt_fg"]
        preds.append(
            {
                "id": row["id"],
                "dice": float(dice_score(pred_fg, gt_fg)),
                "gt_voxels": int(gt_fg.sum()),
                "pred_voxels": int(pred_fg.sum()),
            }
        )
    dices = np.array([r["dice"] for r in preds], dtype=float) if preds else np.array([], dtype=float)
    return {
        "mean_dice": float(np.mean(dices)) if len(dices) else float("nan"),
        "std_dice": float(np.std(dices)) if len(dices) else float("nan"),
        "median_dice": float(np.median(dices)) if len(dices) else float("nan"),
        "predictions": preds,
    }


def tune_params(
    val_rows: Sequence[Dict],
    thresholds: Iterable[float],
    min_component_voxels_grid: Iterable[int],
    keep_largest_options: Iterable[bool],
) -> Dict:
    best = None
    for thr in thresholds:
        for min_vox in min_component_voxels_grid:
            for keep_largest in keep_largest_options:
                metrics = evaluate_rows(val_rows, thr, min_vox, keep_largest)
                score = metrics["mean_dice"]
                candidate = {
                    "threshold": float(thr),
                    "min_component_voxels": int(min_vox),
                    "keep_largest": bool(keep_largest),
                    "val_mean_dice": float(score),
                    "val_std_dice": float(metrics["std_dice"]),
                    "val_median_dice": float(metrics["median_dice"]),
                }
                if best is None:
                    best = candidate
                    continue
                cur_key = (
                    candidate["val_mean_dice"],
                    candidate["val_median_dice"],
                    -candidate["min_component_voxels"],
                    -abs(candidate["threshold"] - 0.5),
                    -int(candidate["keep_largest"]),
                )
                best_key = (
                    best["val_mean_dice"],
                    best["val_median_dice"],
                    -best["min_component_voxels"],
                    -abs(best["threshold"] - 0.5),
                    -int(best["keep_largest"]),
                )
                if cur_key > best_key:
                    best = candidate
    return best


def main() -> None:
    args = parse_args()
    thresholds = args.thresholds if args.thresholds else default_thresholds()
    min_component_grid = args.min_component_voxels if args.min_component_voxels else default_min_components()
    keep_largest_options = [bool(x) for x in args.keep_largest_options]

    splits = pickle.load(open(args.task_dir / "splits.pkl", "rb"))
    if args.split_method not in splits or args.split_param not in splits[args.split_method]:
        raise RuntimeError(f"Missing {args.split_method}[{args.split_param}] in splits.pkl")
    folds = splits[args.split_method][args.split_param]
    fold = folds[args.fold_idx]
    if "test" not in fold:
        raise RuntimeError(f"Split {args.split_method}[{args.split_param}] fold {args.fold_idx} has no test set")

    patch_size = tuple(args.patch_size_dhw)
    val_ids = sorted(fold["val"])
    test_ids = sorted(fold["test"])

    model, device = load_model(args.checkpoint)
    val_rows = collect_predictions(model, device, args.task_dir, val_ids, patch_size, mirror=args.mirror)
    test_rows = collect_predictions(model, device, args.task_dir, test_ids, patch_size, mirror=args.mirror)

    best = tune_params(
        val_rows=val_rows,
        thresholds=thresholds,
        min_component_voxels_grid=min_component_grid,
        keep_largest_options=keep_largest_options,
    )
    test_metrics = evaluate_rows(
        test_rows,
        threshold=best["threshold"],
        min_component_voxels=best["min_component_voxels"],
        keep_largest=best["keep_largest"],
    )

    payload = {
        "model_name": args.model_name,
        "fold_idx": args.fold_idx,
        "split_method": args.split_method,
        "split_param": args.split_param,
        "n_val_subjects": len(val_rows),
        "n_test_subjects": len(test_rows),
        "checkpoint": str(args.checkpoint),
        "task_dir": str(args.task_dir),
        "mirror": bool(args.mirror),
        "tuned_on": "val",
        "evaluated_on": "test",
        "best_postprocess": best,
        "test_mean_dice": test_metrics["mean_dice"],
        "test_std_dice": test_metrics["std_dice"],
        "test_median_dice": test_metrics["median_dice"],
        "predictions": test_metrics["predictions"],
        "threshold_grid": [float(x) for x in thresholds],
        "min_component_voxels_grid": [int(x) for x in min_component_grid],
        "keep_largest_options": keep_largest_options,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "model_name": payload["model_name"],
                "fold_idx": payload["fold_idx"],
                "best_postprocess": payload["best_postprocess"],
                "test_mean_dice": payload["test_mean_dice"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
