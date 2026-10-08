#!/usr/bin/env python3
"""Create quantitative and visual QC for ds004199 FCD alignment."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import find_contours


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task-dir",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-wilms1/FOMO/Data/openneuro_task10_t1flairseg_1mm_20260507/"
            "Task010_OpenNeuro_ds004199_FCDSeg_T1FLAIR_1mm"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/nfs/turbo/umms-wilms1/FOMO/qc/fcd_alignment_20260514"
        ),
    )
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20260514)
    return p.parse_args()


def robust_scale(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    data = arr[np.isfinite(arr)]
    if mask is not None and np.any(mask):
        data = arr[mask & np.isfinite(arr)]
    if data.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(data, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def normalized_mi(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray, sample_size: int, rng: np.random.Generator
) -> float:
    idx = np.flatnonzero(mask & np.isfinite(a) & np.isfinite(b))
    if idx.size == 0:
        return float("nan")
    if idx.size > sample_size:
        idx = rng.choice(idx, size=sample_size, replace=False)
    av = a.ravel()[idx]
    bv = b.ravel()[idx]
    ah, _ = np.histogram(av, bins=64)
    bh, _ = np.histogram(bv, bins=64)
    ab, _, _ = np.histogram2d(av, bv, bins=64)
    pa = ah / max(ah.sum(), 1)
    pb = bh / max(bh.sum(), 1)
    pab = ab / max(ab.sum(), 1)
    ha = -np.sum(pa[pa > 0] * np.log2(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log2(pb[pb > 0]))
    hab = -np.sum(pab[pab > 0] * np.log2(pab[pab > 0]))
    if hab == 0:
        return float("nan")
    return float((ha + hb) / hab)


def case_ids(aligned_dir: Path) -> list[str]:
    ids = []
    for path in aligned_dir.glob("*_T1w.nii.gz"):
        ids.append(path.name.replace("_T1w.nii.gz", ""))
    return sorted(ids)


def bbox(mask: np.ndarray) -> list[int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return [int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2])]


def centroid(mask: np.ndarray, fallback: np.ndarray) -> tuple[int, int, int]:
    source = mask if np.any(mask) else fallback
    if not np.any(source):
        return tuple(int(x // 2) for x in source.shape)
    center = ndimage.center_of_mass(source.astype(np.float32))
    return tuple(int(np.clip(round(c), 0, source.shape[i] - 1)) for i, c in enumerate(center))


def plane_slice(arr: np.ndarray, center: tuple[int, int, int], plane: str) -> np.ndarray:
    x, y, z = center
    if plane == "sagittal":
        return np.rot90(arr[x, :, :])
    if plane == "coronal":
        return np.rot90(arr[:, y, :])
    if plane == "axial":
        return np.rot90(arr[:, :, z])
    raise ValueError(plane)


def draw_contour(ax, mask2d: np.ndarray, color: str, linewidth: float = 1.3) -> None:
    for contour in find_contours(mask2d.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def make_overlay_png(
    out_path: Path,
    case_id: str,
    t1: np.ndarray,
    flair: np.ndarray,
    roi: np.ndarray,
    t1_mask: np.ndarray,
    row: dict,
) -> None:
    center = centroid(roi, t1_mask)
    t1_scaled = robust_scale(t1, t1_mask)
    flair_scaled = robust_scale(flair, t1_mask)
    planes = ["axial", "coronal", "sagittal"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for r, plane in enumerate(planes):
        t1_2d = plane_slice(t1_scaled, center, plane)
        fl_2d = plane_slice(flair_scaled, center, plane)
        roi_2d = plane_slice(roi > 0.5, center, plane)
        mask_2d = plane_slice(t1_mask, center, plane)

        axes[r, 0].imshow(t1_2d, cmap="gray", interpolation="nearest")
        draw_contour(axes[r, 0], mask_2d, "yellow", 0.8)
        draw_contour(axes[r, 0], roi_2d, "red", 1.5)
        axes[r, 0].set_title(f"T1 + ROI/mask ({plane})", fontsize=9)

        axes[r, 1].imshow(fl_2d, cmap="gray", interpolation="nearest")
        draw_contour(axes[r, 1], mask_2d, "yellow", 0.8)
        draw_contour(axes[r, 1], roi_2d, "red", 1.5)
        axes[r, 1].set_title("FLAIR in T1 + ROI", fontsize=9)

        rgb = np.zeros((*t1_2d.shape, 3), dtype=np.float32)
        rgb[..., 0] = t1_2d
        rgb[..., 1] = fl_2d
        rgb[..., 2] = 0.35 * fl_2d
        axes[r, 2].imshow(rgb, interpolation="nearest")
        draw_contour(axes[r, 2], roi_2d, "red", 1.5)
        axes[r, 2].set_title("T1(red) / FLAIR(green)", fontsize=9)

        for c in range(3):
            axes[r, c].axis("off")

    flag_text = ", ".join(row["flags"]) if row["flags"] else "no flags"
    fig.suptitle(
        f"{case_id} | ROI vox={row['roi_voxels']} | ROI outside mask={row['roi_outside_t1mask_voxels']} "
        f"({row['roi_outside_t1mask_fraction']:.3%}) | {flag_text}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_html(rows: list[dict], out_dir: Path) -> None:
    flagged = [r for r in rows if r["flags"]]
    body = [
        "<html><head><meta charset='utf-8'><title>FCD Alignment QC</title>",
        "<style>body{font-family:Arial,sans-serif} table{border-collapse:collapse} "
        "td,th{border:1px solid #ddd;padding:4px} img{width:360px}</style></head><body>",
        "<h1>FCD T1/FLAIR/ROI Alignment QC</h1>",
        f"<p>Total cases: {len(rows)}. Flagged cases: {len(flagged)}.</p>",
        "<p>Overlays are lesion-centered when ROI is present. Yellow contour is T1-derived mask; red contour is ROI.</p>",
        "<h2>Flagged Cases</h2>",
        "<table><tr><th>Case</th><th>Flags</th><th>ROI outside mask</th><th>FLAIR/T1 mask Dice</th><th>Overlay</th></tr>",
    ]
    for r in flagged:
        img = html.escape(r["overlay_png"])
        body.append(
            "<tr>"
            f"<td>{html.escape(r['case_id'])}</td>"
            f"<td>{html.escape('; '.join(r['flags']))}</td>"
            f"<td>{r['roi_outside_t1mask_voxels']} ({r['roi_outside_t1mask_fraction']:.3%})</td>"
            f"<td>{r['flair_t1mask_dice']:.4f}</td>"
            f"<td><a href='{img}'><img src='{img}'></a></td>"
            "</tr>"
        )
    body.append("</table><h2>All Cases</h2><table><tr><th>Case</th><th>Flags</th><th>Overlay</th></tr>")
    for r in rows:
        img = html.escape(r["overlay_png"])
        body.append(
            "<tr>"
            f"<td>{html.escape(r['case_id'])}</td>"
            f"<td>{html.escape('; '.join(r['flags']) if r['flags'] else '')}</td>"
            f"<td><a href='{img}'><img src='{img}'></a></td>"
            "</tr>"
        )
    body.append("</table></body></html>")
    (out_dir / "index.html").write_text("\n".join(body))


def main() -> None:
    args = parse_args()
    aligned_dir = args.task_dir / "_tmp" / "aligned"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    for cid in case_ids(aligned_dir):
        t1_img = nib.load(str(aligned_dir / f"{cid}_T1w.nii.gz"))
        flair_img = nib.load(str(aligned_dir / f"{cid}_FLAIR_inT1.nii.gz"))
        roi_img = nib.load(str(aligned_dir / f"{cid}_ROI_inT1.nii.gz"))
        t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
        flair = np.asarray(flair_img.dataobj, dtype=np.float32)
        roi = np.asarray(roi_img.dataobj, dtype=np.float32) > 0.5
        t1_mask = t1 != 0
        flair_mask = flair != 0
        shape_match = t1_img.shape == flair_img.shape == roi_img.shape
        affine_match = (
            np.allclose(t1_img.affine, flair_img.affine)
            and np.allclose(t1_img.affine, roi_img.affine)
        )
        flair_outside = int(np.logical_and(flair_mask, ~t1_mask).sum())
        roi_outside = int(np.logical_and(roi, ~t1_mask).sum())
        roi_voxels = int(roi.sum())
        roi_outside_fraction = roi_outside / roi_voxels if roi_voxels else 0.0
        t1_voxels = int(t1_mask.sum())
        flair_voxels = int(flair_mask.sum())
        coverage = flair_voxels / t1_voxels if t1_voxels else 0.0
        fmask_dice = dice(flair_mask, t1_mask)
        nmi = normalized_mi(t1, flair, t1_mask & flair_mask, args.sample_size, rng)

        flags: list[str] = []
        if not shape_match:
            flags.append("shape_mismatch")
        if not affine_match:
            flags.append("affine_mismatch")
        if flair_outside > 0:
            flags.append("flair_outside_t1mask")
        if roi_outside_fraction > 0.01:
            flags.append("roi_outside_t1mask_gt1pct")
        elif roi_outside > 0:
            flags.append("small_roi_outside_t1mask")
        if coverage < 0.95:
            flags.append("low_flair_brain_coverage")
        if roi_voxels == 0:
            flags.append("empty_roi")

        row = {
            "case_id": cid,
            "shape": list(map(int, t1_img.shape)),
            "shape_match": bool(shape_match),
            "affine_match": bool(affine_match),
            "spacing": [float(x) for x in t1_img.header.get_zooms()[:3]],
            "t1mask_voxels": t1_voxels,
            "flair_nonzero_voxels": flair_voxels,
            "flair_outside_t1mask_voxels": flair_outside,
            "flair_t1mask_coverage": float(coverage),
            "flair_t1mask_dice": float(fmask_dice),
            "t1_flair_nmi": float(nmi),
            "roi_voxels": roi_voxels,
            "roi_outside_t1mask_voxels": roi_outside,
            "roi_outside_t1mask_fraction": float(roi_outside_fraction),
            "roi_bbox": bbox(roi),
            "flags": flags,
        }
        out_png = overlay_dir / f"{cid}_overlay.png"
        row["overlay_png"] = str(out_png.relative_to(args.output_dir))
        make_overlay_png(out_png, cid, t1, flair, roi, t1_mask, row)
        rows.append(row)

    flagged = [r for r in rows if r["flags"]]
    summary = {
        "task_dir": str(args.task_dir),
        "output_dir": str(args.output_dir),
        "n_cases": len(rows),
        "n_flagged": len(flagged),
        "flag_counts": {},
        "geometry_mismatch_cases": [
            r["case_id"] for r in rows if not r["shape_match"] or not r["affine_match"]
        ],
        "flair_outside_t1mask_total": int(sum(r["flair_outside_t1mask_voxels"] for r in rows)),
        "roi_outside_t1mask_cases": [
            {
                "case_id": r["case_id"],
                "roi_outside_voxels": r["roi_outside_t1mask_voxels"],
                "roi_outside_fraction": r["roi_outside_t1mask_fraction"],
            }
            for r in rows
            if r["roi_outside_t1mask_voxels"] > 0
        ],
        "min_flair_t1mask_coverage": float(min(r["flair_t1mask_coverage"] for r in rows)),
        "median_flair_t1mask_coverage": float(np.median([r["flair_t1mask_coverage"] for r in rows])),
        "min_flair_t1mask_dice": float(min(r["flair_t1mask_dice"] for r in rows)),
        "median_t1_flair_nmi": float(np.nanmedian([r["t1_flair_nmi"] for r in rows])),
    }
    for row in rows:
        for flag in row["flags"]:
            summary["flag_counts"][flag] = summary["flag_counts"].get(flag, 0) + 1

    with (args.output_dir / "qc_summary.json").open("w") as f:
        json.dump({"summary": summary, "cases": rows}, f, indent=2)
    with (args.output_dir / "qc_cases.csv").open("w", newline="") as f:
        fieldnames = [k for k in rows[0].keys() if k not in {"flags", "roi_bbox"}] + [
            "flags",
            "roi_bbox",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["flags"] = ";".join(row["flags"])
            flat["roi_bbox"] = json.dumps(row["roi_bbox"])
            writer.writerow(flat)
    write_html(rows, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
