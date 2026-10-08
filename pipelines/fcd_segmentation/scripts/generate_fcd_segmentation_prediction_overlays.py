#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DEFAULT_OUTPUT_DIR = Path("/nfs/turbo/umms-wilms1/FOMO/qc/fcd_segmentation_unetxl_sanity_20260520")

MODEL_SPECS = {
    "ours": {
        "label": "Ours",
        "model_name": "cleandift_s23",
        "result_dir": "task10_qc76_20260517",
        "stem": "cleandift_s23_qc76_flaironly_b4tb25_imagerefine_focaltversky_fullLR_best_nw2_v2",
        "mode": "tuned",
        "seg_head_variant": "image_refine",
        "patch_size": (192, 256, 192),
        "mirror": True,
        "input_modalities": [1],
    },
    "unet_b": {
        "label": "UNet-B",
        "model_name": "unet_b",
        "result_dir": "task10_qc76_unetb_reseed_20260519",
        "stem": "unet_b_qc76_unetb_b4_all_dicece_best_100b_seed20260519_v1",
        "mode": "argmax_no_tune",
        "seg_head_variant": "earlyfusion",
        "patch_size": (96, 96, 96),
        "mirror": False,
        "input_modalities": None,
    },
    "unet_xl": {
        "label": "UNet-XL",
        "model_name": "unet_xl",
        "result_dir": "task10_qc76_unetxl_reseed_20260519",
        "stem": "unet_xl_qc76_unetxl_b4_all_dicece_best_100b_seed20260519_v1",
        "mode": "argmax_no_tune",
        "seg_head_variant": "earlyfusion",
        "patch_size": (96, 96, 96),
        "mirror": False,
        "input_modalities": None,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    p.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--include-case", action="append", default=[])
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def result_jsons(result_root: Path, spec: dict[str, Any]) -> list[Path]:
    result_dir = result_root / spec["result_dir"]
    return sorted(result_dir.glob(f"{spec['stem']}*fold*_test.json"))


def load_result_index(result_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    index: dict[str, dict[int, dict[str, Any]]] = {}
    for key, model_spec in MODEL_SPECS.items():
        by_fold: dict[int, dict[str, Any]] = {}
        for path in result_jsons(result_root, model_spec):
            payload = read_json(path)
            if payload.get("model_name") != model_spec["model_name"]:
                continue
            if payload.get("evaluation_mode") != model_spec["mode"]:
                continue
            fold = int(payload["fold_idx"])
            by_fold[fold] = {"path": path, "payload": payload}
        index[key] = by_fold
    return index


def select_cases(index: dict[str, dict[int, dict[str, Any]]], top_n: int, include_case: list[str]) -> list[dict[str, Any]]:
    candidates = []
    for fold, row in index["unet_xl"].items():
        for pred in row["payload"]["predictions"]:
            candidates.append(
                {
                    "id": pred["id"],
                    "fold": fold,
                    "unet_xl_dice": float(pred["dice"]),
                    "unet_xl_pred_voxels": int(pred["pred_voxels"]),
                    "gt_voxels": int(pred["gt_voxels"]),
                }
            )
    selected = sorted(candidates, key=lambda x: x["unet_xl_dice"], reverse=True)[:top_n]
    seen = {row["id"] for row in selected}
    for case_id in include_case:
        if case_id in seen:
            continue
        match = next((row for row in candidates if row["id"] == case_id), None)
        if match:
            selected.append(match)
            seen.add(case_id)
    return selected


def prediction_lookup(index: dict[str, dict[int, dict[str, Any]]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    out = {}
    for model_key, by_fold in index.items():
        for fold, row in by_fold.items():
            for pred in row["payload"]["predictions"]:
                out[(model_key, fold, pred["id"])] = pred
    return out


def robust_scale(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    data = arr[finite]
    if mask is not None and np.any(mask):
        data = arr[finite & mask.astype(bool)]
    if data.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(data, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def lesion_center(mask: np.ndarray) -> tuple[int, int, int]:
    if np.any(mask):
        center = ndimage.center_of_mass(mask.astype(np.float32))
        return tuple(int(np.clip(round(c), 0, mask.shape[i] - 1)) for i, c in enumerate(center))
    return tuple(int(x // 2) for x in mask.shape)


def plane_slice(arr: np.ndarray, center: tuple[int, int, int], plane: str) -> np.ndarray:
    d, h, w = center
    if plane == "axial":
        return np.rot90(arr[:, :, w])
    if plane == "coronal":
        return np.rot90(arr[:, h, :])
    if plane == "sagittal":
        return np.rot90(arr[d, :, :])
    raise ValueError(plane)


def draw_contour(ax, mask2d: np.ndarray, color: str, linewidth: float = 1.5) -> None:
    if not np.any(mask2d):
        return
    ax.contour(mask2d.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def load_case(task_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(task_dir / f"{case_id}.npy", allow_pickle=True)
    t1 = np.asarray(raw[0], dtype=np.float32)
    flair = np.asarray(raw[1], dtype=np.float32)
    label = np.asarray(raw[-1], dtype=np.uint8)
    return t1, flair, label


def predict_cases(
    task_dir: Path,
    selected: list[dict[str, Any]],
    index: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, np.ndarray]]:
    predictions: dict[str, dict[str, np.ndarray]] = {row["id"]: {} for row in selected}
    by_fold: dict[int, list[str]] = {}
    for row in selected:
        by_fold.setdefault(int(row["fold"]), []).append(row["id"])

    for model_key, model_spec in MODEL_SPECS.items():
        for fold, case_ids in sorted(by_fold.items()):
            result = index[model_key].get(fold)
            if result is None:
                raise RuntimeError(f"Missing {model_key} fold {fold} result JSON")
            payload = result["payload"]
            checkpoint = Path(payload["checkpoint"])
            print(f"Loading {model_spec['label']} fold {fold}: {checkpoint}", flush=True)
            model, device = fcd_eval.load_model(checkpoint, model_spec["seg_head_variant"])
            patch_size = tuple(model_spec["patch_size"])
            for case_id in case_ids:
                image_np, _ = fcd_eval.load_case(
                    task_dir / f"{case_id}.npy",
                    model_spec["input_modalities"],
                )
                if model_spec["mode"] == "argmax_no_tune":
                    mask = fcd_eval.predict_argmax_mask(
                        model,
                        device,
                        image_np,
                        patch_size,
                        mirror=bool(model_spec["mirror"]),
                    )
                else:
                    fg_prob = fcd_eval.predict_fg_prob(
                        model,
                        device,
                        image_np,
                        patch_size,
                        mirror=bool(model_spec["mirror"]),
                    )
                    best = payload["best_postprocess"]
                    mask = fcd_eval.postprocess_mask(
                        fg_prob,
                        float(best["threshold"]),
                        int(best["min_component_voxels"]),
                        bool(best["keep_largest"]),
                    )
                    image_fg = fcd_eval.image_foreground_mask(image_np)
                    mask = (mask.astype(bool) & image_fg.astype(bool)).astype(np.uint8)
                predictions[case_id][model_key] = mask.astype(np.uint8)
                print(f"  predicted {model_spec['label']} {case_id}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return predictions


def make_overlay(
    out_path: Path,
    task_dir: Path,
    case_row: dict[str, Any],
    predictions: dict[str, np.ndarray],
    lookup: dict[tuple[str, int, str], dict[str, Any]],
) -> None:
    case_id = case_row["id"]
    fold = int(case_row["fold"])
    _, flair, label = load_case(task_dir, case_id)
    label_fg = label > 0
    center = lesion_center(label_fg)
    foreground = flair > 1e-6
    flair_scaled = robust_scale(flair, foreground)

    planes = ["axial", "coronal", "sagittal"]
    columns = ["gt", "ours", "unet_b", "unet_xl"]
    titles = {
        "gt": "GT ROI",
        "ours": "Ours",
        "unet_b": "UNet-B",
        "unet_xl": "UNet-XL",
    }
    fig, axes = plt.subplots(len(planes), len(columns), figsize=(13, 9), constrained_layout=True)
    for r, plane in enumerate(planes):
        base = plane_slice(flair_scaled, center, plane)
        gt2d = plane_slice(label_fg, center, plane)
        for c, column in enumerate(columns):
            ax = axes[r, c]
            ax.imshow(base, cmap="gray", interpolation="nearest")
            draw_contour(ax, gt2d, "lime", 1.6)
            if column != "gt":
                pred2d = plane_slice(predictions[column] > 0, center, plane)
                draw_contour(ax, pred2d, "red", 1.4)
            if r == 0:
                title = titles[column]
                if column != "gt":
                    pred = lookup[(column, fold, case_id)]
                    title += f"\nDice={float(pred['dice']):.3f}, vox={int(pred['pred_voxels'])}"
                else:
                    title += f"\nGT vox={int(label_fg.sum())}"
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(plane, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"{case_id} | fold {fold} | green=GT, red=prediction | selected by UNet-XL Dice={case_row['unet_xl_dice']:.3f}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_index(out_dir: Path, selected: list[dict[str, Any]], lookup: dict[tuple[str, int, str], dict[str, Any]]) -> None:
    rows = [
        "<html><head><meta charset='utf-8'><title>FCD Segmentation Sanity Overlays</title>",
        "<style>body{font-family:Arial,sans-serif} table{border-collapse:collapse} "
        "td,th{border:1px solid #ddd;padding:5px;vertical-align:top} img{width:520px}</style></head><body>",
        "<h1>FCD Segmentation Sanity Overlays</h1>",
        "<p>Cases are selected from the highest UNet-XL QC76 test Dice values. Green contour is GT ROI; red contour is the model prediction.</p>",
        "<table><tr><th>Case</th><th>Fold</th><th>Ours Dice</th><th>UNet-B Dice</th><th>UNet-XL Dice</th><th>Overlay</th></tr>",
    ]
    for row in selected:
        case_id = row["id"]
        fold = int(row["fold"])
        img = f"{case_id}_fold{fold}_overlay.png"
        rows.append(
            "<tr>"
            f"<td>{html.escape(case_id)}</td>"
            f"<td>{fold}</td>"
            f"<td>{lookup[('ours', fold, case_id)]['dice']:.3f}</td>"
            f"<td>{lookup[('unet_b', fold, case_id)]['dice']:.3f}</td>"
            f"<td>{lookup[('unet_xl', fold, case_id)]['dice']:.3f}</td>"
            f"<td><a href='overlays/{img}'><img src='overlays/{img}'></a></td>"
            "</tr>"
        )
    rows.append("</table></body></html>")
    (out_dir / "index.html").write_text("\n".join(rows))


def write_csv(out_dir: Path, selected: list[dict[str, Any]], lookup: dict[tuple[str, int, str], dict[str, Any]]) -> None:
    with (out_dir / "selected_case_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "fold",
                "gt_voxels",
                "ours_dice",
                "ours_pred_voxels",
                "unet_b_dice",
                "unet_b_pred_voxels",
                "unet_xl_dice",
                "unet_xl_pred_voxels",
            ]
        )
        for row in selected:
            case_id = row["id"]
            fold = int(row["fold"])
            writer.writerow(
                [
                    case_id,
                    fold,
                    row["gt_voxels"],
                    lookup[("ours", fold, case_id)]["dice"],
                    lookup[("ours", fold, case_id)]["pred_voxels"],
                    lookup[("unet_b", fold, case_id)]["dice"],
                    lookup[("unet_b", fold, case_id)]["pred_voxels"],
                    lookup[("unet_xl", fold, case_id)]["dice"],
                    lookup[("unet_xl", fold, case_id)]["pred_voxels"],
                ]
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    index = load_result_index(args.result_root)
    selected = select_cases(index, args.top_n, args.include_case)
    lookup = prediction_lookup(index)
    predictions = predict_cases(args.task_dir, selected, index)
    for row in selected:
        case_id = row["id"]
        fold = int(row["fold"])
        out_path = overlay_dir / f"{case_id}_fold{fold}_overlay.png"
        make_overlay(out_path, args.task_dir, row, predictions[case_id], lookup)
        print(f"Wrote {out_path}", flush=True)
    write_index(args.output_dir, selected, lookup)
    write_csv(args.output_dir, selected, lookup)
    print(f"Wrote {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
