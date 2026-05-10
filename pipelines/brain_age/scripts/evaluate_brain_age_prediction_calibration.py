#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression


SITES = ("GSP", "BrainLat", "Task3")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--output-json", required=True)
    return ap.parse_args()


def load_rows(path: str | Path) -> list[dict]:
    d = json.load(open(path))
    rows = d["predictions"]
    for r in rows:
        if "site" not in r:
            rid = r["id"]
            if rid.startswith("GSP_"):
                r["site"] = "GSP"
            elif rid.startswith("BrainLat_"):
                r["site"] = "BrainLat"
            elif rid.startswith("Task3_"):
                r["site"] = "Task3"
            else:
                r["site"] = "UNKNOWN"
    return rows


def metric_block(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["gt"] is not None and float(r["gt"]) > 0]
    gt = np.array([r["gt"] for r in valid], dtype=float)
    pred = np.array([r["pred"] for r in valid], dtype=float)
    err = pred - gt
    out = {
        "n": int(len(valid)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }
    by_site = {}
    for site in SITES:
        sub = [r for r in valid if r["site"] == site]
        g = np.array([r["gt"] for r in sub], dtype=float)
        p = np.array([r["pred"] for r in sub], dtype=float)
        e = p - g
        by_site[site] = {
            "n": int(len(sub)),
            "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
        }
    out["per_site"] = by_site
    return out


def apply_fn(rows: list[dict], fn) -> list[dict]:
    out = []
    for r in rows:
        rr = dict(r)
        rr["pred"] = float(fn(r))
        out.append(rr)
    return out


def main():
    args = parse_args()
    val_rows = load_rows(args.val_json)
    test_rows = load_rows(args.test_json)

    results = {}
    results["raw"] = metric_block(test_rows)

    X = np.array([[r["pred"]] for r in val_rows], dtype=float)
    y = np.array([r["gt"] for r in val_rows], dtype=float)
    glr = LinearRegression().fit(X, y)
    rows = apply_fn(test_rows, lambda r: glr.predict([[r["pred"]]])[0])
    results["global_affine"] = {
        "coef": float(glr.coef_[0]),
        "intercept": float(glr.intercept_),
        "metrics": metric_block(rows),
    }

    gbias = float(np.mean([r["pred"] - r["gt"] for r in val_rows]))
    rows = apply_fn(test_rows, lambda r: r["pred"] - gbias)
    results["global_bias"] = {
        "bias": gbias,
        "metrics": metric_block(rows),
    }

    site_bias = {
        site: float(
            np.mean([r["pred"] - r["gt"] for r in val_rows if r["site"] == site])
        )
        for site in SITES
    }
    rows = apply_fn(test_rows, lambda r: r["pred"] - site_bias[r["site"]])
    results["site_bias"] = {
        "bias_by_site": site_bias,
        "metrics": metric_block(rows),
    }

    site_models = {}
    for site in SITES:
        X = np.array([[r["pred"]] for r in val_rows if r["site"] == site], dtype=float)
        y = np.array([r["gt"] for r in val_rows if r["site"] == site], dtype=float)
        site_models[site] = LinearRegression().fit(X, y)
    rows = apply_fn(
        test_rows, lambda r: site_models[r["site"]].predict([[r["pred"]]])[0]
    )
    results["site_affine"] = {
        "params_by_site": {
            site: {
                "coef": float(site_models[site].coef_[0]),
                "intercept": float(site_models[site].intercept_),
            }
            for site in SITES
        },
        "metrics": metric_block(rows),
    }

    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print(f"Saved: {args.output_json}")
    for name, block in results.items():
        m = block if "mae" in block else block["metrics"]
        print(
            name,
            m["mae"],
            m["per_site"]["BrainLat"]["mae"],
            m["per_site"]["Task3"]["mae"],
        )


if __name__ == "__main__":
    main()
