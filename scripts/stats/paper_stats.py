from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats_tests import (
    auroc,
    bonferroni_adjust,
    delong_auc_test,
    holm_adjust,
    mean_sd,
    paired_t_test,
    wilcoxon_signed_rank,
)

def _check_pairing(df: pd.DataFrame, key: str, models: list[str]):
    sets = {m: set(df.loc[df["model"] == m, key]) for m in models}
    ref = sets[models[0]]
    for m in models[1:]:
        if sets[m] != ref:
            raise ValueError(f"model {m} has a different subject set from {models[0]}; fix inputs before testing")

def stroke_or_fcd_classification(df: pd.DataFrame, reference: str, task_name: str):
    models = sorted(df["model"].unique())
    if reference not in models:
        raise ValueError(f"reference {reference!r} not in classification file")
    rows = []
    others = [m for m in models if m != reference]

    per_fold = {}
    for m in models:
        fold_aucs = []
        for _, sub in df[df["model"] == m].groupby("fold"):
            fold_aucs.append(auroc(sub["label"], sub["score"]))
        per_fold[m] = fold_aucs
    p_fold = {}
    for m in others:
        _, p = paired_t_test(per_fold[reference], per_fold[m])
        p_fold[m] = p
    adj_fold = dict(zip(others, holm_adjust([p_fold[m] for m in others]))) if others else {}

    ref_sub = df[df["model"] == reference].sort_values(["subject_id"])
    for m in models:
        mean, sd = mean_sd(per_fold[m])
        row = {"task": task_name, "model": m, "AUROC_foldmean": mean, "AUROC_foldsd": sd, "AUROC_pooled": None,
               "p_fold_holm": None, "p_delong_holm": None}
        if m == reference:
            row["AUROC_pooled"] = auroc(ref_sub["label"], ref_sub["score"])
        rows.append(row)

    p_delong = {}
    for m in others:
        other_sub = df[df["model"] == m].sort_values(["subject_id"])
        _check_pairing(df, "subject_id", [reference, m])
        merged = ref_sub[["subject_id", "label", "score"]].merge(
            other_sub[["subject_id", "score"]], on="subject_id", suffixes=("_ref", "_other")
        )
        out = delong_auc_test(merged["label"], merged["score_ref"], merged["score_other"])
        p_delong[m] = out["p_value"]
        for row in rows:
            if row["model"] == m:
                row["AUROC_pooled"] = out["auc_b"]
    adj_delong = dict(zip(others, holm_adjust([p_delong[m] for m in others]))) if others else {}

    for row in rows:
        m = row["model"]
        if m in adj_fold:
            row["p_fold_holm"] = adj_fold[m]
        if m in adj_delong:
            row["p_delong_holm"] = adj_delong[m]
    return rows

def brain_age(df: pd.DataFrame, reference: str):
    models = sorted(df["model"].unique())
    if reference not in models:
        raise ValueError(f"reference {reference!r} not in regression file")
    for m in models:
        _check_pairing(df, "subject_id", [reference, m])
    rows, pvals, others = [], {}, [m for m in models if m != reference]
    ref = df[df["model"] == reference].sort_values("subject_id")
    ref_err = np.abs(ref["y_pred"].to_numpy() - ref["y_true"].to_numpy())
    for m in others:
        sub = df[df["model"] == m].sort_values("subject_id")
        err = np.abs(sub["y_pred"].to_numpy() - sub["y_true"].to_numpy())
        _, p = paired_t_test(ref_err, err)
        pvals[m] = p
    adj = dict(zip(others, bonferroni_adjust([pvals[m] for m in others]))) if others else {}
    for m in models:
        sub = df[df["model"] == m].sort_values("subject_id")
        err = np.abs(sub["y_pred"].to_numpy() - sub["y_true"].to_numpy())
        mae_mean, mae_sd = mean_sd(err)
        r = float(np.corrcoef(sub["y_true"], sub["y_pred"])[0, 1])
        rows.append({"task": "brain_age", "model": m, "MAE": mae_mean, "MAE_sd": mae_sd, "pearson_r": r,
                     "p_paired_t_bonferroni": adj.get(m)})
    return rows

def segmentation(df: pd.DataFrame, reference: str):
    models = sorted(df["model"].unique())
    if reference not in models:
        raise ValueError(f"reference {reference!r} not in segmentation file")
    for m in models:
        _check_pairing(df, "subject_id", [reference, m])
    rows = []
    ref = df[df["model"] == reference].sort_values("subject_id")
    for m in models:
        sub = df[df["model"] == m].sort_values("subject_id")
        row = {"task": "segmentation", "model": m}
        for metric in ("dice", "nsd"):
            mean, sd = mean_sd(sub[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            if m != reference:
                _, p = wilcoxon_signed_rank(ref[metric], sub[metric])
                row[f"{metric}_p_wilcoxon"] = p
        rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Model every other model is compared against (e.g. Ours)")
    parser.add_argument("--classification", nargs="*", default=[], help="Entries of the form task_name=path.csv")
    parser.add_argument("--regression", default=None)
    parser.add_argument("--segmentation", nargs="*", default=[], help="Entries of the form task_name=path.csv")
    args = parser.parse_args()

    report = {}
    for entry in args.classification:
        name, path = entry.split("=", 1)
        report[name] = stroke_or_fcd_classification(pd.read_csv(path), args.reference, name)
    if args.regression:
        report["brain_age"] = brain_age(pd.read_csv(args.regression), args.reference)
    for entry in args.segmentation:
        name, path = entry.split("=", 1)
        report[name] = segmentation(pd.read_csv(path), args.reference)

    for name, rows in report.items():
        print(f"\n## {name}")
        frame = pd.DataFrame(rows)
        with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.4g}".format):
            print(frame.to_string(index=False))

if __name__ == "__main__":
    main()
