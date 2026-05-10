#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr


REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_SRC = REPO_ROOT / "src" / "downstream"
PRETRAINING_SRC = REPO_ROOT / "src" / "pretraining"
for path in (str(DOWNSTREAM_SRC), str(PRETRAINING_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.supervised_reg import SupervisedRegModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task_dir", type=Path, required=True)
    parser.add_argument("--split_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--patch_size_dhw", type=int, nargs=3, required=True)
    return parser.parse_args()


def pad_or_center_crop(volume, patch_size):
    vol = torch.from_numpy(volume).float()
    pad = []
    for size, target in zip(reversed(vol.shape[-3:]), reversed(patch_size)):
        before = max(0, (target - size) // 2)
        after = max(0, target - size - before)
        pad.extend([before, after])
    if any(pad):
        vol = F.pad(vol, pad, mode="constant", value=0)

    d, h, w = vol.shape[-3:]
    target_d, target_h, target_w = patch_size
    sd = max(0, (d - target_d) // 2)
    sh = max(0, (h - target_h) // 2)
    sw = max(0, (w - target_w) // 2)
    return vol[..., sd : sd + target_d, sh : sh + target_h, sw : sw + target_w]


def dataset_name(case_id):
    if case_id.startswith("GSP_"):
        return "GSP"
    if case_id.startswith("BrainLat_"):
        return "BrainLat"
    if case_id.startswith("Task3_"):
        return "Task3"
    return "UNK"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedRegModel.load_from_checkpoint(checkpoint_path=str(args.checkpoint))
    model.eval().to(device)

    with args.split_json.open() as f:
        test_ids = sorted(json.load(f)["test"])

    predictions = []
    for idx, case_id in enumerate(test_ids, 1):
        npy_path = args.task_dir / f"{case_id}.npy"
        txt_path = args.task_dir / f"{case_id}.txt"
        if not npy_path.exists() or not txt_path.exists():
            continue

        ground_truth = float(txt_path.read_text().strip())
        volume = np.load(npy_path).astype(np.float32)
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]

        inputs = pad_or_center_crop(volume, tuple(args.patch_size_dhw))[None].to(device)
        with torch.no_grad():
            outputs = (
                model.run_predict(inputs.float())
                if model.model_name == "mmunetvae"
                else model(inputs.float())
            )
        pred = float(outputs[0, 0])
        predictions.append(
            {
                "id": case_id,
                "dataset": dataset_name(case_id),
                "ground_truth": ground_truth,
                "predicted_age": pred,
                "absolute_error": abs(pred - ground_truth),
            }
        )
        if idx % 100 == 0 or idx == len(test_ids):
            print(f"[{idx}/{len(test_ids)}] {case_id}: pred={pred:.2f}, gt={ground_truth:.2f}")

    gt = np.array([row["ground_truth"] for row in predictions], dtype=float)
    pred = np.array([row["predicted_age"] for row in predictions], dtype=float)
    mae = float(np.mean(np.abs(gt - pred)))
    rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
    ss_res = float(np.sum((gt - pred) ** 2))
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pearson_r, pearson_p = pearsonr(gt, pred)

    per_dataset_rows = defaultdict(list)
    for row in predictions:
        per_dataset_rows[row["dataset"]].append(row)

    metrics_by_dataset = {}
    for dataset, rows in sorted(per_dataset_rows.items()):
        ds_gt = np.array([row["ground_truth"] for row in rows], dtype=float)
        ds_pred = np.array([row["predicted_age"] for row in rows], dtype=float)
        ds_mae = float(np.mean(np.abs(ds_gt - ds_pred)))
        ds_rmse = float(np.sqrt(np.mean((ds_gt - ds_pred) ** 2)))
        ds_ss_res = float(np.sum((ds_gt - ds_pred) ** 2))
        ds_ss_tot = float(np.sum((ds_gt - np.mean(ds_gt)) ** 2))
        ds_r2 = float(1 - ds_ss_res / ds_ss_tot) if ds_ss_tot > 0 else float("nan")
        ds_pearson_r, ds_pearson_p = pearsonr(ds_gt, ds_pred)
        metrics_by_dataset[dataset] = {
            "count": len(rows),
            "mae": ds_mae,
            "rmse": ds_rmse,
            "r2": ds_r2,
            "pearson_r": float(ds_pearson_r),
            "pearson_p": float(ds_pearson_p),
        }

    payload = {
        "n_test": len(predictions),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "metrics_by_dataset": metrics_by_dataset,
        "predictions": predictions,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
