#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from scipy.stats import ttest_rel


OUT_DIR = Path("/nfs/turbo/umms-wilms1/FOMO/inference_outputs")
OUT_XLSX = OUT_DIR / "FOMO300K_results_v3_siteagebalanced_20260504.xlsx"

SPLIT_JSON = Path(
    "/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k_unified/splits/"
    "fomo300k_v2_stratified_ft200_test500_gsp179_brainlat179_task3142_diskvalidated_siteagebalanced.json"
)

OURS_JSON = OUT_DIR / (
    "s23_ft200_t500_matchclean500_exactpretrainfull700_sub100headerfix_"
    "siteagebalanced_p192x256x192_bs4_e100_tb100_lr1e4_augall_mse_"
    "bestvalmae_full100_20260502_test.json"
)
OURS_TIMING = OUT_DIR / (
    "s23_ft200_t500_matchclean500_exactpretrainfull700_sub100headerfix_"
    "siteagebalanced_p192x256x192_bs4_e100_tb100_lr1e4_augall_mse_"
    "bestvalmae_full100_20260502_timing.json"
)
UNET_B_JSON = OUT_DIR / (
    "fomo300k_v2_unet_b_exactpretrainfull700_sub100headerfix_"
    "siteagebalanced_20260502_test.json"
)
UNET_B_TIMING = OUT_DIR / (
    "fomo300k_v2_unet_b_exactpretrainfull700_sub100headerfix_"
    "siteagebalanced_20260502_timing.json"
)
UNET_XL_JSON = OUT_DIR / (
    "fomo300k_v2_unet_xl_exactpretrainfull700_sub100headerfix_"
    "siteagebalanced_20260502_test.json"
)
UNET_XL_TIMING = OUT_DIR / (
    "fomo300k_v2_unet_xl_exactpretrainfull700_sub100headerfix_"
    "siteagebalanced_20260502_timing.json"
)
MMU_JSON = OUT_DIR / (
    "fomo300k_v2_mmunetvae_exactpretrainfull700_sub100headerfix_"
    "siteagebalanced_20260502_test.json"
)
MMU_LOG = Path(
    "/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/logs/ft_mmu_siteage-49222018.log"
)

THIN = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
TITLE_FONT = Font(bold=True, size=13)
HDR_FONT = Font(bold=True, size=11)
FILL_BLUE = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def section_title(ws, text: str, cols: int) -> None:
    ws.append([text])
    ws.merge_cells(
        start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=cols
    )
    c = ws.cell(ws.max_row, 1)
    c.font = TITLE_FONT
    c.alignment = Alignment(horizontal="left")


def header_row(ws, values: list) -> None:
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


def fmt_hms(seconds: int | float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def load_result(path: Path) -> dict:
    d = load_json(path)
    if "overall" in d:
        return {
            "n": d["overall"]["n"],
            "mae": d["overall"]["mae"],
            "pearson_r": d["overall"]["pearson_r"],
            "per_dataset": d["per_dataset"],
            "predictions": d["predictions"],
        }
    return {
        "n": d["n"],
        "mae": d["MAE"],
        "pearson_r": d["Pearson_r"],
        "per_dataset": d["per_dataset"],
        "predictions": d["predictions"],
    }


def timing_mmu_seconds(log_path: Path) -> int:
    text = log_path.read_text()
    m = re.search(r"Fine-tune done in (\d+)s", text)
    if not m:
        raise RuntimeError(
            f"Could not parse MMUNetVAE finetune seconds from {log_path}"
        )
    return int(m.group(1))


def ds_n(per_dataset: dict, name: str) -> int:
    return int(per_dataset[name]["n"])


def sorted_preds(result: dict) -> list[dict]:
    return sorted(result["predictions"], key=lambda r: r["id"])


def paired_abs_errors(
    a_rows: list[dict], b_rows: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    a_map = {r["id"]: r for r in a_rows}
    b_map = {r["id"]: r for r in b_rows}
    ids = sorted(set(a_map) & set(b_map))
    a_err, b_err = [], []
    for sid in ids:
        ga = a_map[sid]["gt"]
        gb = b_map[sid]["gt"]
        if ga != gb or ga is None or ga <= 0:
            continue
        a_err.append(abs(a_map[sid]["pred"] - ga))
        b_err.append(abs(b_map[sid]["pred"] - ga))
    return np.array(a_err, dtype=np.float64), np.array(b_err, dtype=np.float64)


def main() -> None:
    split = load_json(SPLIT_JSON)
    test_ids = split["test"]
    gsp_n = sum(s.startswith("GSP_") for s in test_ids)
    brainlat_n = sum(s.startswith("BrainLat_") for s in test_ids)
    task3_n = sum(s.startswith("Task3_") for s in test_ids)

    ours = load_result(OURS_JSON)
    unet_b = load_result(UNET_B_JSON)
    unet_xl = load_result(UNET_XL_JSON)
    mmu = load_result(MMU_JSON)

    ours_time = load_json(OURS_TIMING)["finetune_seconds"]
    unet_b_time = load_json(UNET_B_TIMING)["finetune_seconds"]
    unet_xl_time = load_json(UNET_XL_TIMING)["finetune_seconds"]
    mmu_time = timing_mmu_seconds(MMU_LOG)

    models = [
        ("Ours", "Ours-23M (bs=4)", ours, ours_time),
        ("Baseline", "UNet-B (bs=2)", unet_b, unet_b_time),
        ("", "UNet-XL (bs=2)", unet_xl, unet_xl_time),
        ("", "MMUNetVAE (bs=4)", mmu, mmu_time),
    ]

    wb = Workbook()

    ws = wb.active
    ws.title = "FOMO300K_Metrics"
    section_title(
        ws,
        "FOMO300K v3 (N=700): Age Regression — Grouped Model Summary",
        9,
    )
    ws.append([])
    header_row(
        ws,
        [
            "Group",
            "Model",
            "N",
            "MAE",
            "Pearson r",
            "GSP (n)",
            "BrainLat (n)",
            "Task3 (n)",
            "Finetune Time",
        ],
    )
    for group, name, result, tsec in models:
        ws.append(
            [
                group,
                name,
                result["n"],
                result["mae"],
                result["pearson_r"],
                gsp_n,
                brainlat_n,
                task3_n,
                fmt_hms(tsec),
            ]
        )
        format_last_row(ws)
    ws.freeze_panes = "A4"

    ws = wb.create_sheet("FOMO300K_Predictions")
    section_title(
        ws,
        "FOMO300K v3: Individual Predictions (matched 500 subjects across all models)",
        6,
    )
    ws.append([])
    pred_headers = [
        "Subject",
        "GT",
        "Ours-23M (bs=4)",
        "UNet-B (bs=2)",
        "UNet-XL (bs=2)",
        "MMUNetVAE (bs=4)",
    ]
    header_row(ws, pred_headers)
    pred_maps = {
        "Ours-23M (bs=4)": {r["id"]: r["pred"] for r in ours["predictions"]},
        "UNet-B (bs=2)": {r["id"]: r["pred"] for r in unet_b["predictions"]},
        "UNet-XL (bs=2)": {r["id"]: r["pred"] for r in unet_xl["predictions"]},
        "MMUNetVAE (bs=4)": {r["id"]: r["pred"] for r in mmu["predictions"]},
    }
    gt_map = {r["id"]: r["gt"] for r in ours["predictions"]}
    common_ids = sorted(
        set(gt_map)
        & set(pred_maps["UNet-B (bs=2)"])
        & set(pred_maps["UNet-XL (bs=2)"])
        & set(pred_maps["MMUNetVAE (bs=4)"])
    )
    for sid in common_ids:
        ws.append(
            [
                sid,
                gt_map[sid],
                pred_maps["Ours-23M (bs=4)"][sid],
                pred_maps["UNet-B (bs=2)"][sid],
                pred_maps["UNet-XL (bs=2)"][sid],
                pred_maps["MMUNetVAE (bs=4)"][sid],
            ]
        )
        format_last_row(ws)
    ws.freeze_panes = "A4"

    ws = wb.create_sheet("FOMO300K_Statistical_Tests")
    bonf = 0.05 / 3
    section_title(
        ws,
        "Paired Statistical Tests — Our Models vs Comparators (FOMO300K TEST set, n=500). "
        f"Test: paired two-sided t-test on per-subject absolute error (|pred-gt|). "
        f"Bonferroni-corrected alpha={bonf:.6f} (0.05/3 tests). YES = Bonferroni-corrected significance.",
        12,
    )
    ws.append([])
    header_row(
        ws,
        [
            "Our Model",
            "Comparator",
            "n (paired)",
            "Our MAE",
            "Comparator MAE",
            "MAE Improvement (yrs)",
            "Our Pearson r",
            "Comparator Pearson r",
            "Paired t-stat",
            "Paired t-test p (MAE)",
            "Bonferroni threshold",
            "Significant (Bonf.)?",
        ],
    )
    ours_rows = sorted_preds(ours)
    for comp_name, comp_res in [
        ("UNet-B (bs=2)", unet_b),
        ("UNet-XL (bs=2)", unet_xl),
        ("MMUNetVAE (bs=4)", mmu),
    ]:
        comp_rows = sorted_preds(comp_res)
        our_err, comp_err = paired_abs_errors(ours_rows, comp_rows)
        stat = ttest_rel(our_err, comp_err)
        pval = float(stat.pvalue)
        ws.append(
            [
                "Ours-23M (bs=4)",
                comp_name,
                int(len(our_err)),
                float(our_err.mean()),
                float(comp_err.mean()),
                float(comp_err.mean() - our_err.mean()),
                ours["pearson_r"],
                comp_res["pearson_r"],
                float(stat.statistic),
                pval,
                bonf,
                "YES" if pval < bonf else "",
            ]
        )
        format_last_row(ws)
    ws.freeze_panes = "A4"

    for sheet in wb.worksheets:
        widths = {}
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is None:
                    continue
                widths[cell.column_letter] = max(
                    widths.get(cell.column_letter, 0), len(str(cell.value))
                )
        for col, width in widths.items():
            sheet.column_dimensions[col].width = min(max(width + 2, 12), 28)

    wb.save(OUT_XLSX)
    print(f"Saved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
