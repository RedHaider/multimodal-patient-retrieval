from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import distance_matrix, make_cohort, retrieval_metrics


def evaluate(seed: int = 27):
    image, clinical, labels = make_cohort(seed=seed)
    results = {}
    for name, weight in {"clinical_only": 0.0, "fused": 0.6, "image_only": 1.0}.items():
        results[name] = retrieval_metrics(distance_matrix(image, clinical, weight), labels)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = evaluate(args.seed)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
