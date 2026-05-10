#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


WB_PATH = Path(
    "/nfs/turbo/umms-wilms1/FOMO/inference_outputs/"
    "FOMO300K_results_v3_siteagebalanced_20260504.xlsx"
)
RESULT_DIRS = [
    Path("/nfs/turbo/umms-wilms1/FOMO/inference_outputs/task10_nested_tuned_20260509"),
    Path("/nfs/turbo/umms-wilms1/FOMO/inference_outputs/task10_nested_tuned_20260510"),
]

THIN = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
TITLE_FONT = Font(bold=True, size=13)
HDR_FONT = Font(bold=True, size=11)
FILL_BLUE = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

MODEL_LABELS = {
    "cleandift_s23": "Ours-23M (bs=1)",
    "mmunetvae": "MMUNetVAE (bs=2)",
}
GROUP_LABELS = {
    "cleandift_s23": "Ours",
    "mmunetvae": "Baseline",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def section_title(ws, text: str, cols: int) -> None:
    ws.append([text])
    ws.merge_cells(
        start_row=ws.max_row,
        start_column=1,
        end_row=ws.max_row,
        end_column=cols,
    )
    cell = ws.cell(ws.max_row, 1)
    cell.font = TITLE_FONT
    cell.alignment = Alignment(horizontal="left")


def header_row(ws, values: list[str]) -> None:
    ws.append(values)
    for cell in ws[ws.max_row]:
        cell.font = HDR_FONT
        cell.fill = FILL_BLUE
        cell.border = THIN
        cell.alignment = Alignment(horizontal="center")


def format_last_row(ws) -> None:
    for cell in ws[ws.max_row]:
        cell.border = THIN
        if isinstance(cell.value, float):
            cell.number_format = "0.0000"


def autosize(ws) -> None:
    widths = {}
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            widths[cell.column_letter] = max(widths.get(cell.column_letter, 0), len(str(cell.value)))
    for col, width in widths.items():
        ws.column_dimensions[col].width = min(max(width + 2, 12), 36)


def describe_setting(path: Path) -> str:
    stem = path.stem
    if "fullall_p192ftv" in stem:
        return "aug=all, full 192x256x192, focaltversky"
    if "fullbasic_p192ftv" in stem:
        return "aug=basic, full 192x256x192, focaltversky"
    if "fullnone_p192ftv" in stem:
        return "aug=none, full 192x256x192, focaltversky"
    if "all_ftv" in stem:
        return "aug=all, focaltversky"
    if "basic_dicece" in stem:
        return "aug=basic, dicece"
    if "none_ftv" in stem:
        return "aug=none, focaltversky"
    return stem


def collect_result_files() -> list[Path]:
    files: list[Path] = []
    for result_dir in RESULT_DIRS:
        if result_dir.exists():
            files.extend(sorted(result_dir.glob("*_fold0_test.json")))
    return files


def main() -> None:
    result_files = collect_result_files()
    if not result_files:
        raise SystemExit("No Task10 fold-0 result JSONs found.")

    wb = load_workbook(WB_PATH)
    for name in ["FCD_Seg_Metrics", "FCD_Seg_Postproc", "FCD_Seg_Subjects"]:
        if name in wb.sheetnames:
            del wb[name]

    rows = []
    subject_rows = []
    for path in result_files:
        payload = load_json(path)
        model_name = payload["model_name"]
        best_pp = payload["best_postprocess"]
        rows.append(
            {
                "group": GROUP_LABELS.get(model_name, ""),
                "model": MODEL_LABELS.get(model_name, model_name),
                "setting": describe_setting(path),
                "fold_idx": payload["fold_idx"],
                "n_val_subjects": payload["n_val_subjects"],
                "n_test_subjects": payload["n_test_subjects"],
                "test_mean_dice": payload["test_mean_dice"],
                "test_std_dice": payload["test_std_dice"],
                "test_median_dice": payload["test_median_dice"],
                "threshold": best_pp["threshold"],
                "min_component_voxels": best_pp["min_component_voxels"],
                "keep_largest": best_pp["keep_largest"],
                "val_mean_dice": best_pp["val_mean_dice"],
                "val_std_dice": best_pp["val_std_dice"],
                "val_median_dice": best_pp["val_median_dice"],
                "path": str(path),
            }
        )
        for pred in payload.get("predictions", []):
            subject_rows.append(
                {
                    "group": GROUP_LABELS.get(model_name, ""),
                    "model": MODEL_LABELS.get(model_name, model_name),
                    "setting": describe_setting(path),
                    "fold_idx": payload["fold_idx"],
                    "id": pred["id"],
                    "dice": pred["dice"],
                    "gt_voxels": pred["gt_voxels"],
                    "pred_voxels": pred["pred_voxels"],
                }
            )

    rows.sort(key=lambda r: (r["model"], r["setting"]))
    subject_rows.sort(key=lambda r: (r["model"], r["setting"], r["id"]))

    ws = wb.create_sheet("FCD_Seg_Metrics")
    section_title(ws, "ds004199 FCD Segmentation — Nested Fold-0 Outer-Test Summary", 10)
    ws.append([])
    header_row(
        ws,
        [
            "Group",
            "Model",
            "Setting",
            "Fold",
            "N Val",
            "N Test",
            "Test Mean Dice",
            "Test Std Dice",
            "Test Median Dice",
            "Result file",
        ],
    )
    for row in rows:
        ws.append(
            [
                row["group"],
                row["model"],
                row["setting"],
                row["fold_idx"],
                row["n_val_subjects"],
                row["n_test_subjects"],
                row["test_mean_dice"],
                row["test_std_dice"],
                row["test_median_dice"],
                row["path"],
            ]
        )
        format_last_row(ws)
    ws.freeze_panes = "A4"
    autosize(ws)

    ws = wb.create_sheet("FCD_Seg_Postproc")
    section_title(ws, "ds004199 FCD Segmentation — Tuned Postprocessing", 10)
    ws.append([])
    header_row(
        ws,
        [
            "Group",
            "Model",
            "Setting",
            "Threshold",
            "Min Component Voxels",
            "Keep Largest",
            "Val Mean Dice",
            "Val Std Dice",
            "Val Median Dice",
            "Fold",
        ],
    )
    for row in rows:
        ws.append(
            [
                row["group"],
                row["model"],
                row["setting"],
                row["threshold"],
                row["min_component_voxels"],
                int(bool(row["keep_largest"])),
                row["val_mean_dice"],
                row["val_std_dice"],
                row["val_median_dice"],
                row["fold_idx"],
            ]
        )
        format_last_row(ws)
    ws.freeze_panes = "A4"
    autosize(ws)

    ws = wb.create_sheet("FCD_Seg_Subjects")
    section_title(ws, "ds004199 FCD Segmentation — Per-Subject Fold-0 Outer-Test Dice", 8)
    ws.append([])
    header_row(
        ws,
        [
            "Group",
            "Model",
            "Setting",
            "Fold",
            "Subject ID",
            "Dice",
            "GT Voxels",
            "Pred Voxels",
        ],
    )
    for row in subject_rows:
        ws.append(
            [
                row["group"],
                row["model"],
                row["setting"],
                row["fold_idx"],
                row["id"],
                row["dice"],
                row["gt_voxels"],
                row["pred_voxels"],
            ]
        )
        format_last_row(ws)
    ws.freeze_panes = "A4"
    autosize(ws)

    wb.save(WB_PATH)
    print(f"Updated workbook: {WB_PATH}")


if __name__ == "__main__":
    main()
