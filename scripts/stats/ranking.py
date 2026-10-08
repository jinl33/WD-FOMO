from __future__ import annotations

import argparse
import json

from scipy.stats import rankdata

def rank_within_task(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    models = list(values)
    scores = [values[m] for m in models]
    key = [-s for s in scores] if higher_is_better else scores
    ranks = rankdata(key, method="average")
    return {m: float(r) for m, r in zip(models, ranks)}

def mean_rank(tasks: dict[str, dict]) -> dict[str, float]:
    per_model: dict[str, list[float]] = {}
    for task_name, spec in tasks.items():
        higher = _is_higher_better(spec["metric"], task_name)
        ranks = rank_within_task(spec["values"], higher)
        for model, r in ranks.items():
            per_model.setdefault(model, []).append(r)
    return {m: sum(rs) / len(rs) for m, rs in per_model.items()}

def _is_higher_better(direction: str, task_name: str) -> bool:
    if direction not in {"higher", "lower"}:
        raise ValueError(f"task {task_name}: metric direction must be 'higher' or 'lower', got {direction!r}")
    return direction == "higher"

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, help="JSON file with per-task metric values")
    args = parser.parse_args()
    with open(args.table) as f:
        tasks = json.load(f)

    result = mean_rank(tasks)
    print(f"{'model':<14}{'mean rank':>10}")
    for model, r in sorted(result.items(), key=lambda kv: kv[1]):
        print(f"{model:<14}{r:>10.2f}")

if __name__ == "__main__":
    main()
