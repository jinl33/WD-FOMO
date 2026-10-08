#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np

def load_labels(data_dir: Path) -> dict[str, int]:
    labels = {}
    for txt_path in sorted(data_dir.glob("*.txt")):
        subject_id = txt_path.stem
        if not (data_dir / f"{subject_id}.npy").exists():
            continue
        labels[subject_id] = int(float(txt_path.read_text().strip()))
    if not labels:
        raise SystemExit(f"No labelled subjects (*.txt with matching *.npy) found under {data_dir}")
    return labels

def stratified_kfold(ids, y, n_folds, seed):
    rng = np.random.RandomState(seed)
    ids = np.array(ids)
    y = np.array(y)
    folds = [[] for _ in range(n_folds)]
    for cls in sorted(set(y.tolist())):
        cls_ids = ids[y == cls].copy()
        rng.shuffle(cls_ids)
        for i, subject_id in enumerate(cls_ids):
            folds[i % n_folds].append(subject_id)
    return [sorted(f) for f in folds]

def build_splits(labels: dict[str, int], n_folds: int, seed: int) -> dict:
    ids = list(labels.keys())
    y = [labels[i] for i in ids]
    fold_val_ids = stratified_kfold(ids, y, n_folds, seed)

    kfold = []
    for k in range(n_folds):
        val_ids = fold_val_ids[k]
        train_ids = sorted(set(ids) - set(val_ids))
        assert not (set(train_ids) & set(val_ids))
        kfold.append({"train": train_ids, "val": val_ids})

    n_pos = sum(y)
    n_neg = len(y) - n_pos
    for k, fold in enumerate(kfold):
        val_pos = sum(labels[i] for i in fold["val"])
        print(f"fold {k}: train={len(fold['train'])} val={len(fold['val'])} (val positives={val_pos})")
    print(f"total: n={len(ids)} positive={n_pos} negative={n_neg}")

    return {"kfold": {n_folds: kfold}}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory with {id}.npy and {id}.txt per subject")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to <data_dir>/splits.pkl")
    args = parser.parse_args()

    labels = load_labels(args.data_dir)
    splits = build_splits(labels, args.n_folds, args.seed)

    out_path = args.out or (args.data_dir / "splits.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
