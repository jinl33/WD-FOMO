import argparse
import glob
import json
import os
import random

def build_split(files, n_holdout, seed):
    if n_holdout >= len(files):
        raise ValueError(f"n_holdout={n_holdout} must be smaller than the number of files ({len(files)})")
    ordered = sorted(files)
    random.Random(seed).shuffle(ordered)
    distillation = sorted(ordered[:n_holdout])
    pretrain = sorted(ordered[n_holdout:])
    return pretrain, distillation

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--n_holdout", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data_path, args.pattern))
    if not files:
        raise SystemExit(f"no files match {os.path.join(args.data_path, args.pattern)}")
    pretrain, distillation = build_split(files, args.n_holdout, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "pretrain": pretrain,
                "distillation": distillation,
                "meta": {
                    "seed": args.seed,
                    "n_total": len(files),
                    "n_pretrain": len(pretrain),
                    "n_distillation": len(distillation),
                    "pattern": args.pattern,
                    "data_path": os.path.abspath(args.data_path),
                },
            },
            f,
            indent=1,
        )
    print(f"wrote {args.out}: pretrain={len(pretrain)} distillation={len(distillation)}")

if __name__ == "__main__":
    main()
