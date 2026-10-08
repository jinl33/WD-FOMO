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
    p.add_argument("--seg_head_variant", type=str, default=None)
    p.add_argument("--fold_idx", type=int, required=True)
    p.add_argument("--split_method", type=str, default="stratified_train_val_test_split")
    p.add_argument("--split_param", type=str, default="0.8")
    p.add_argument("--patch_size_dhw", type=int, nargs=3, required=True)
    p.add_argument("--input_modality_indices", type=int, nargs="+", default=None)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--thresholds", type=float, nargs="*", default=None)
    p.add_argument("--min_component_voxels", type=int, nargs="*", default=None)
    p.add_argument("--keep_largest_options", type=int, nargs="*", default=(0, 1))
    p.add_argument("--fixed_threshold", type=float, default=None)
    p.add_argument("--fixed_min_component_voxels", type=int, default=None)
    p.add_argument("--fixed_keep_largest", type=int, choices=(0, 1), default=None)
    p.add_argument("--restrict_to_image_foreground", action="store_true")
    p.add_argument("--argmax_no_tune", action="store_true")
    return p.parse_args()

def resolve_split_key(
    splits: Dict,
    split_method: str,
    split_param_raw: str,
):
    if split_method not in splits:
        raise RuntimeError(f"Missing split method {split_method} in splits.pkl")

    candidate_params = [split_param_raw]
    try:
        candidate_params.append(int(split_param_raw))
    except (TypeError, ValueError):
        pass
    try:
        float_value = float(split_param_raw)
        candidate_params.append(float_value)
        if float_value.is_integer():
            candidate_params.append(str(int(float_value)))
        candidate_params.append(str(float_value))
    except (TypeError, ValueError):
        pass

    seen = []
    for candidate in candidate_params:
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate in splits[split_method]:
            return candidate

    raise RuntimeError(
        f"Missing {split_method}[{split_param_raw}] in splits.pkl "
        f"(tried keys: {seen})"
    )

def default_thresholds() -> List[float]:
    return [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]

def default_min_components() -> List[int]:
    return [0, 16, 32, 64, 128, 256, 512, 1024]

def load_case(
    npy_path: Path,
    modality_indices: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.load(npy_path, allow_pickle=True)
    n_channels = len(raw) - 1
    selected = list(range(n_channels)) if modality_indices is None else list(modality_indices)
    bad = [idx for idx in selected if idx < 0 or idx >= n_channels]
    if bad:
        raise IndexError(
            f"Requested modality indices {bad}, but {npy_path.name} has {n_channels} modalities"
        )
    image = np.stack([np.asarray(raw[i]).astype(np.float32) for i in selected], axis=0)
    label = np.asarray(raw[-1]).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(label)

def image_foreground_mask(image: np.ndarray, dilation_iters: int = 2) -> np.ndarray:
    mask = np.any(np.abs(image) > 1e-6, axis=0)
    if mask.any() and dilation_iters > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilation_iters)
    return mask.astype(np.uint8)

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

def surface_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    surface = mask & ~eroded
    return surface if surface.any() else mask

def normalized_surface_dice(
    pred_fg: np.ndarray,
    gt_fg: np.ndarray,
    tolerance_mm: float = 1.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    pred_fg = pred_fg.astype(bool)
    gt_fg = gt_fg.astype(bool)
    if not pred_fg.any() and not gt_fg.any():
        return 1.0
    if not pred_fg.any() or not gt_fg.any():
        return 0.0

    pred_surface = surface_mask(pred_fg)
    gt_surface = surface_mask(gt_fg)
    pred_count = int(pred_surface.sum())
    gt_count = int(gt_surface.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    dist_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    pred_close = int((dist_to_gt[pred_surface] <= tolerance_mm).sum())
    gt_close = int((dist_to_pred[gt_surface] <= tolerance_mm).sum())
    return float((pred_close + gt_close) / (pred_count + gt_count))

def load_model(checkpoint: Path, expected_seg_head_variant: str | None = None):
    from models.supervised_seg import SupervisedSegModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    init_kwargs = dict(checkpoint_obj.get("hyper_parameters", {}))
    model = SupervisedSegModel(**init_kwargs)
    load_result = torch.nn.Module.load_state_dict(
        model, checkpoint_obj["state_dict"], strict=False
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "[load_model] checkpoint load warnings:",
            {
                "missing_keys": load_result.missing_keys,
                "unexpected_keys": load_result.unexpected_keys,
            },
        )
    actual_variant = getattr(model.model, "seg_head_variant", None)
    if expected_seg_head_variant and actual_variant and actual_variant != expected_seg_head_variant:
        raise RuntimeError(
            "Loaded segmentation head variant does not match requested variant: "
            f"checkpoint={actual_variant}, requested={expected_seg_head_variant}"
        )
    model.eval().to(device)
    return model, device

def predict_logits(model, device, image_np: np.ndarray, patch_size: Tuple[int, int, int], mirror: bool) -> np.ndarray:
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
    return pred, orig_shape

def predict_fg_prob(model, device, image_np: np.ndarray, patch_size: Tuple[int, int, int], mirror: bool) -> np.ndarray:
    pred, orig_shape = predict_logits(model, device, image_np, patch_size, mirror)
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

def predict_argmax_mask(model, device, image_np: np.ndarray, patch_size: Tuple[int, int, int], mirror: bool) -> np.ndarray:
    pred, orig_shape = predict_logits(model, device, image_np, patch_size, mirror)
    if pred.ndim == 4 and pred.shape[0] > 1:
        pred_seg = np.argmax(pred, axis=0).astype(np.uint8)
    elif pred.ndim == 4 and pred.shape[0] == 1:
        pred_seg = (pred[0] > 0).astype(np.uint8)
    elif pred.ndim == 3:
        pred_seg = (pred > 0).astype(np.uint8)
    else:
        raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")
    return crop_back(pred_seg, orig_shape).astype(np.uint8)

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
    return keep[labeled].astype(np.uint8)

def summarize_predictions(preds: Sequence[Dict]) -> Dict:
    dices = np.array([r["dice"] for r in preds], dtype=float) if preds else np.array([], dtype=float)
    nsds = np.array([r["nsd"] for r in preds], dtype=float) if preds else np.array([], dtype=float)
    return {
        "mean_dice": float(np.mean(dices)) if len(dices) else float("nan"),
        "std_dice": float(np.std(dices)) if len(dices) else float("nan"),
        "median_dice": float(np.median(dices)) if len(dices) else float("nan"),
        "mean_nsd": float(np.mean(nsds)) if len(nsds) else float("nan"),
        "std_nsd": float(np.std(nsds)) if len(nsds) else float("nan"),
        "median_nsd": float(np.median(nsds)) if len(nsds) else float("nan"),
        "predictions": list(preds),
    }

def evaluate_argmax_cases_streaming(
    model,
    device,
    task_dir: Path,
    case_ids: Sequence[str],
    patch_size: Tuple[int, int, int],
    mirror: bool,
    modality_indices: Sequence[int] | None = None,
    restrict_to_image_fg: bool = False,
) -> Dict:
    preds = []
    for idx, fid in enumerate(case_ids, 1):
        image_np, label_np = load_case(task_dir / f"{fid}.npy", modality_indices)
        pred_fg = predict_argmax_mask(model, device, image_np, patch_size, mirror=mirror)
        if restrict_to_image_fg:
            pred_fg = (pred_fg.astype(bool) & image_foreground_mask(image_np).astype(bool)).astype(np.uint8)
        gt_fg = (label_np > 0).astype(np.uint8)
        preds.append(
            {
                "id": fid,
                "dice": float(dice_score(pred_fg, gt_fg)),
                "nsd": float(normalized_surface_dice(pred_fg, gt_fg)),
                "gt_voxels": int(gt_fg.sum()),
                "pred_voxels": int(pred_fg.sum()),
            }
        )
        if idx % 5 == 0 or idx == len(case_ids):
            print(f"[argmax evaluate {idx}/{len(case_ids)}] {fid}", flush=True)
    return summarize_predictions(preds)

def evaluate_cases_streaming(
    model,
    device,
    task_dir: Path,
    case_ids: Sequence[str],
    patch_size: Tuple[int, int, int],
    mirror: bool,
    threshold: float,
    min_component_voxels: int,
    keep_largest: bool,
    modality_indices: Sequence[int] | None = None,
    restrict_to_image_fg: bool = False,
) -> Dict:
    preds = []
    for idx, fid in enumerate(case_ids, 1):
        image_np, label_np = load_case(task_dir / f"{fid}.npy", modality_indices)
        fg_prob = predict_fg_prob(model, device, image_np, patch_size, mirror=mirror)
        pred_fg = postprocess_mask(fg_prob, threshold, min_component_voxels, keep_largest)
        if restrict_to_image_fg:
            pred_fg = (pred_fg.astype(bool) & image_foreground_mask(image_np).astype(bool)).astype(np.uint8)
        gt_fg = (label_np > 0).astype(np.uint8)
        preds.append(
            {
                "id": fid,
                "dice": float(dice_score(pred_fg, gt_fg)),
                "nsd": float(normalized_surface_dice(pred_fg, gt_fg)),
                "gt_voxels": int(gt_fg.sum()),
                "pred_voxels": int(pred_fg.sum()),
            }
        )
        if idx % 5 == 0 or idx == len(case_ids):
            print(f"[evaluate {idx}/{len(case_ids)}] {fid}", flush=True)
    return summarize_predictions(preds)

def tune_params(
    model,
    device,
    task_dir: Path,
    case_ids: Sequence[str],
    patch_size: Tuple[int, int, int],
    mirror: bool,
    thresholds: Iterable[float],
    min_component_voxels_grid: Iterable[int],
    keep_largest_options: Iterable[bool],
    modality_indices: Sequence[int] | None = None,
    restrict_to_image_fg: bool = False,
) -> Dict:
    rows = []
    for idx, fid in enumerate(case_ids, 1):
        image_np, label_np = load_case(task_dir / f"{fid}.npy", modality_indices)
        rows.append(
            {
                "id": fid,
                "fg_prob": predict_fg_prob(model, device, image_np, patch_size, mirror=mirror),
                "gt_fg": (label_np > 0).astype(np.uint8),
                "image_fg": image_foreground_mask(image_np),
            }
        )
        if idx % 5 == 0 or idx == len(case_ids):
            print(f"[tune collect {idx}/{len(case_ids)}] {fid}", flush=True)

    best = None
    for thr in thresholds:
        for min_vox in min_component_voxels_grid:
            for keep_largest in keep_largest_options:
                preds = []
                for row in rows:
                    pred_fg = postprocess_mask(
                        row["fg_prob"], thr, min_vox, keep_largest
                    )
                    if restrict_to_image_fg:
                        pred_fg = (pred_fg.astype(bool) & row["image_fg"].astype(bool)).astype(np.uint8)
                    gt_fg = row["gt_fg"]
                    preds.append(float(dice_score(pred_fg, gt_fg)))
                mean_dice = float(np.mean(preds)) if preds else float("nan")
                median_dice = float(np.median(preds)) if preds else float("nan")
                candidate = {
                    "threshold": float(thr),
                    "min_component_voxels": int(min_vox),
                    "keep_largest": bool(keep_largest),
                    "val_mean_dice": mean_dice,
                    "val_median_dice": median_dice,
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
    fixed_values = (
        args.fixed_threshold,
        args.fixed_min_component_voxels,
        args.fixed_keep_largest,
    )
    fixed_postprocess = any(value is not None for value in fixed_values)
    if fixed_postprocess and not all(value is not None for value in fixed_values):
        raise ValueError(
            "Fixed postprocessing requires --fixed_threshold, "
            "--fixed_min_component_voxels, and --fixed_keep_largest."
        )

    splits = pickle.load(open(args.task_dir / "splits.pkl", "rb"))
    resolved_split_param = resolve_split_key(
        splits,
        args.split_method,
        args.split_param,
    )
    folds = splits[args.split_method][resolved_split_param]
    fold = folds[args.fold_idx]
    if "test" not in fold:
        raise RuntimeError(
            f"Split {args.split_method}[{resolved_split_param}] "
            f"fold {args.fold_idx} has no test set"
        )

    patch_size = tuple(args.patch_size_dhw)
    val_ids = sorted(fold["val"])
    test_ids = sorted(fold["test"])

    model, device = load_model(args.checkpoint, args.seg_head_variant)
    if args.argmax_no_tune:
        test_metrics = evaluate_argmax_cases_streaming(
            model,
            device,
            args.task_dir,
            test_ids,
            patch_size,
            mirror=args.mirror,
            modality_indices=args.input_modality_indices,
            restrict_to_image_fg=args.restrict_to_image_foreground,
        )
        payload = {
            "model_name": args.model_name,
            "seg_head_variant": args.seg_head_variant,
            "fold_idx": args.fold_idx,
            "split_method": args.split_method,
            "split_param": resolved_split_param,
            "patch_size_dhw": list(patch_size),
            "n_val_subjects": len(val_ids),
            "n_test_subjects": len(test_metrics["predictions"]),
            "checkpoint": str(args.checkpoint),
            "task_dir": str(args.task_dir),
            "input_modality_indices": args.input_modality_indices,
            "mirror": bool(args.mirror),
            "evaluation_mode": "argmax_no_tune",
            "restrict_to_image_foreground": bool(args.restrict_to_image_foreground),
            "best_postprocess": None,
            "test_mean_dice": test_metrics["mean_dice"],
            "test_std_dice": test_metrics["std_dice"],
            "test_median_dice": test_metrics["median_dice"],
            "test_mean_nsd": test_metrics["mean_nsd"],
            "test_std_nsd": test_metrics["std_nsd"],
            "test_median_nsd": test_metrics["median_nsd"],
            "predictions": test_metrics["predictions"],
        }
    else:
        if fixed_postprocess:
            best = {
                "threshold": float(args.fixed_threshold),
                "min_component_voxels": int(args.fixed_min_component_voxels),
                "keep_largest": bool(args.fixed_keep_largest),
                "val_mean_dice": None,
                "val_median_dice": None,
            }
        else:
            best = tune_params(
                model,
                device,
                args.task_dir,
                val_ids,
                patch_size,
                mirror=args.mirror,
                thresholds=thresholds,
                min_component_voxels_grid=min_component_grid,
                keep_largest_options=keep_largest_options,
                modality_indices=args.input_modality_indices,
                restrict_to_image_fg=args.restrict_to_image_foreground,
            )
        test_metrics = evaluate_cases_streaming(
            model,
            device,
            args.task_dir,
            test_ids,
            patch_size,
            mirror=args.mirror,
            threshold=best["threshold"],
            min_component_voxels=best["min_component_voxels"],
            keep_largest=best["keep_largest"],
            modality_indices=args.input_modality_indices,
            restrict_to_image_fg=args.restrict_to_image_foreground,
        )
        payload = {
            "model_name": args.model_name,
            "seg_head_variant": args.seg_head_variant,
            "fold_idx": args.fold_idx,
            "split_method": args.split_method,
            "split_param": resolved_split_param,
            "patch_size_dhw": list(patch_size),
            "n_val_subjects": len(val_ids),
            "n_test_subjects": len(test_metrics["predictions"]),
            "checkpoint": str(args.checkpoint),
            "task_dir": str(args.task_dir),
            "input_modality_indices": args.input_modality_indices,
            "mirror": bool(args.mirror),
            "evaluation_mode": "fixed_postprocess" if fixed_postprocess else "tuned",
            "restrict_to_image_foreground": bool(args.restrict_to_image_foreground),
            "best_postprocess": best,
            "test_mean_dice": test_metrics["mean_dice"],
            "test_std_dice": test_metrics["std_dice"],
            "test_median_dice": test_metrics["median_dice"],
            "test_mean_nsd": test_metrics["mean_nsd"],
            "test_std_nsd": test_metrics["std_nsd"],
            "test_median_nsd": test_metrics["median_nsd"],
            "predictions": test_metrics["predictions"],
            "threshold_grid": None if fixed_postprocess else [float(x) for x in thresholds],
            "min_component_voxels_grid": None if fixed_postprocess else [int(x) for x in min_component_grid],
            "keep_largest_options": None if fixed_postprocess else keep_largest_options,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "model_name": payload["model_name"],
                "fold_idx": payload["fold_idx"],
                "evaluation_mode": payload["evaluation_mode"],
                "best_postprocess": payload["best_postprocess"],
                "test_mean_dice": payload["test_mean_dice"],
                "test_mean_nsd": payload["test_mean_nsd"],
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
