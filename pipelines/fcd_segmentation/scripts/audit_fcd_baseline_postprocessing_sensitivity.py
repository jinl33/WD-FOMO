#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_SRC = REPO_ROOT / "src" / "downstream"
PRETRAINING_SRC = REPO_ROOT / "src" / "pretraining"
for path in [DOWNSTREAM_SRC, PRETRAINING_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

EVAL_SCRIPT = Path(__file__).with_name("evaluate_fcd_segmentation_tuned_fold.py")
spec = importlib.util.spec_from_file_location("fcd_eval", EVAL_SCRIPT)
fcd_eval = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fcd_eval)

DEFAULT_TASK_DIR = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/openneuro_task10_t1flairseg_1mm_20260507/"
    "Task010_OpenNeuro_ds004199_FCDSeg_T1FLAIR_1mm"
)
DEFAULT_RESULT_ROOT = Path("/nfs/turbo/umms-wilms1/FOMO/inference_outputs")
DEFAULT_OUTPUT_DIR = Path(
    "/nfs/turbo/umms-wilms1/FOMO/qc/fcd_segmentation_baseline_postprocess_audit_20260520"
)

MODEL_SPECS = {
    "unet_b": {
        "label": "UNet-B",
        "result_dir": "task10_qc76_unetb_reseed_20260519",
        "stem": "unet_b_qc76_unetb_b4_all_dicece_best_100b_seed20260519_v1",
        "patch_size": (96, 96, 96),
    },
    "unet_xl": {
        "label": "UNet-XL",
        "result_dir": "task10_qc76_unetxl_reseed_20260519",
        "stem": "unet_xl_qc76_unetxl_b4_all_dicece_best_100b_seed20260519_v1",
        "patch_size": (96, 96, 96),
    },
}

VARIANTS = {
    "raw_argmax": {"foreground": False, "keep_largest": False, "min_voxels": 0},
    "foreground_argmax": {"foreground": True, "keep_largest": False, "min_voxels": 0},
    "largest_component": {"foreground": False, "keep_largest": True, "min_voxels": 0},
    "foreground_largest_component": {"foreground": True, "keep_largest": True, "min_voxels": 0},
    "foreground_min64": {"foreground": True, "keep_largest": False, "min_voxels": 64},
    "foreground_min256": {"foreground": True, "keep_largest": False, "min_voxels": 256},
    "foreground_min1024": {"foreground": True, "keep_largest": False, "min_voxels": 1024},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    p.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def load_payloads(result_root: Path, model_key: str, model_spec: dict[str, Any]) -> list[dict[str, Any]]:
    result_dir = result_root / model_spec["result_dir"]
    rows = []
    for path in sorted(result_dir.glob(f"{model_spec['stem']}_fold*_test.json")):
        payload = json.loads(path.read_text())
        if payload.get("model_name") != model_key:
            continue
        if payload.get("evaluation_mode") != "argmax_no_tune":
            continue
        rows.append({"path": str(path), "payload": payload})
    return sorted(rows, key=lambda x: int(x["payload"]["fold_idx"]))


def component_filter(mask: np.ndarray, keep_largest: bool, min_voxels: int) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask.astype(np.uint8)
    labeled, num = ndimage.label(mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    valid = np.arange(1, len(counts))
    if min_voxels > 0:
        valid = valid[counts[valid] >= min_voxels]
    if keep_largest and len(valid):
        valid = np.array([valid[np.argmax(counts[valid])]])
    keep[valid] = True
    return keep[labeled].astype(np.uint8)


def percentile(values: list[float], pct: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize(preds: list[dict[str, Any]]) -> dict[str, Any]:
    dices = [float(x["dice"]) for x in preds]
    nsds = [float(x["nsd"]) for x in preds]
    return {
        "n": len(preds),
        "mean_dice": statistics.mean(dices),
        "std_dice": statistics.stdev(dices) if len(dices) > 1 else 0.0,
        "median_dice": statistics.median(dices),
        "case_dice_q1": percentile(dices, 25),
        "case_dice_q3": percentile(dices, 75),
        "zero_dice_cases": sum(1 for x in dices if x <= 0.0),
        "dice_ge_0_1_cases": sum(1 for x in dices if x >= 0.1),
        "dice_ge_0_2_cases": sum(1 for x in dices if x >= 0.2),
        "mean_nsd": statistics.mean(nsds),
    }


def evaluate_model(
    task_dir: Path,
    model_key: str,
    model_spec: dict[str, Any],
    payload_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    variant_preds: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    for row in payload_rows:
        payload = row["payload"]
        fold = int(payload["fold_idx"])
        checkpoint = Path(payload["checkpoint"])
        print(f"Loading {model_spec['label']} fold {fold}: {checkpoint}", flush=True)
        model, device = fcd_eval.load_model(checkpoint, "earlyfusion")
        case_ids = [pred["id"] for pred in payload["predictions"]]
        for idx, case_id in enumerate(case_ids, 1):
            image_np, label_np = fcd_eval.load_case(task_dir / f"{case_id}.npy", None)
            raw_mask = fcd_eval.predict_argmax_mask(
                model,
                device,
                image_np,
                tuple(model_spec["patch_size"]),
                mirror=False,
            )
            image_fg = fcd_eval.image_foreground_mask(image_np)
            gt = (label_np > 0).astype(np.uint8)
            for variant_name, variant in VARIANTS.items():
                pred = raw_mask
                if variant["foreground"]:
                    pred = (pred.astype(bool) & image_fg.astype(bool)).astype(np.uint8)
                pred = component_filter(
                    pred,
                    keep_largest=bool(variant["keep_largest"]),
                    min_voxels=int(variant["min_voxels"]),
                )
                variant_preds[variant_name].append(
                    {
                        "id": case_id,
                        "fold": fold,
                        "dice": float(fcd_eval.dice_score(pred, gt)),
                        "nsd": float(fcd_eval.normalized_surface_dice(pred, gt)),
                        "gt_voxels": int(gt.sum()),
                        "pred_voxels": int(pred.sum()),
                    }
                )
            if idx % 5 == 0 or idx == len(case_ids):
                print(f"  {model_spec['label']} fold {fold}: {idx}/{len(case_ids)}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "model": model_key,
        "variants": {
            name: {
                "summary": summarize(preds),
                "predictions": preds,
            }
            for name, preds in variant_preds.items()
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {"task_dir": str(args.task_dir), "models": {}}
    for model_key, model_spec in MODEL_SPECS.items():
        payloads = load_payloads(args.result_root, model_key, model_spec)
        if len(payloads) != 5:
            raise RuntimeError(f"Expected 5 folds for {model_key}, found {len(payloads)}")
        out["models"][model_key] = evaluate_model(args.task_dir, model_key, model_spec, payloads)
    out_path = args.output_dir / "baseline_postprocess_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}", flush=True)
    for model_key, model_result in out["models"].items():
        print(f"\n{model_key}", flush=True)
        for variant, variant_result in model_result["variants"].items():
            s = variant_result["summary"]
            print(
                f"  {variant}: mean={s['mean_dice']:.6f}, median={s['median_dice']:.6f}, "
                f"zero={s['zero_dice_cases']}/{s['n']}, >=0.2={s['dice_ge_0_2_cases']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
