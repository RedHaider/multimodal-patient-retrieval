from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import make_cohort

SEEDS = (27, 41, 73, 101, 137)
CONDITIONS = ("clean", "image_missing", "clinical_missing", "mixed_missing")


def _standardize(gallery, query):
    mean, std = gallery.mean(0), gallery.std(0) + 1e-8
    return (gallery - mean) / std, (query - mean) / std


def _corrupt(image, clinical, condition, seed):
    image, clinical = image.copy(), clinical.copy()
    rng = np.random.default_rng(seed)
    n = len(image)
    if condition in ("image_missing", "mixed_missing"):
        idx = rng.choice(n, int(.35 * n), replace=False); image[idx] = np.nan
    if condition in ("clinical_missing", "mixed_missing"):
        idx = rng.choice(n, int(.35 * n), replace=False); clinical[idx] = np.nan
    return image, clinical


def _distances(g_img, g_clin, q_img, q_clin, base_weight, adaptive):
    gi, qi = _standardize(g_img, np.nan_to_num(q_img, nan=np.nanmean(g_img, axis=0)))
    gc, qc = _standardize(g_clin, np.nan_to_num(q_clin, nan=np.nanmean(g_clin, axis=0)))
    di = ((qi[:, None] - gi[None, :]) ** 2).mean(2)
    dc = ((qc[:, None] - gc[None, :]) ** 2).mean(2)
    weights = np.full(len(qi), base_weight)
    if adaptive:
        image_available = ~np.isnan(q_img).all(1)
        clinical_available = ~np.isnan(q_clin).all(1)
        weights[~image_available & clinical_available] = 0.0
        weights[image_available & ~clinical_available] = 1.0
        weights[~image_available & ~clinical_available] = .5
    return weights[:, None] * di + (1 - weights[:, None]) * dc


def _metrics(distance, gallery_labels, query_labels, k=5):
    order = np.argsort(distance, axis=1)
    matches = gallery_labels[order] == query_labels[:, None]
    first = np.argmax(matches, axis=1) + 1
    return {"precision_at_5": float(matches[:, :k].mean()),
            "mean_reciprocal_rank": float(np.mean(1.0 / first))}


def _tune(g_img, g_clin, g_y, v_img, v_clin, v_y, adaptive, seed):
    scores = []
    for weight in np.linspace(.1, .9, 9):
        condition_scores = []
        for i, condition in enumerate(CONDITIONS if adaptive else ("clean",)):
            image, clinical = _corrupt(v_img, v_clin, condition, seed + i)
            d = _distances(g_img, g_clin, image, clinical, weight, adaptive)
            condition_scores.append(_metrics(d, g_y, v_y)["precision_at_5"])
        scores.append(np.mean(condition_scores))
    return float(np.linspace(.1, .9, 9)[int(np.argmax(scores))])


def run():
    rows = []
    tuned = []
    for seed in SEEDS:
        image, clinical, labels = make_cohort(1300, seed)
        g_img, v_img, t_img = image[:600], image[600:900], image[900:]
        g_clin, v_clin, t_clin = clinical[:600], clinical[600:900], clinical[900:]
        g_y, v_y, t_y = labels[:600], labels[600:900], labels[900:]
        baseline_weight = _tune(g_img, g_clin, g_y, v_img, v_clin, v_y, False, seed)
        robust_weight = _tune(g_img, g_clin, g_y, v_img, v_clin, v_y, True, seed)
        tuned.append({"seed": seed, "baseline_weight": baseline_weight, "robust_weight": robust_weight})
        for index, condition in enumerate(CONDITIONS):
            qi, qc = _corrupt(t_img, t_clin, condition, seed + 200 + index)
            for model, weight, adaptive in (("global_fusion", baseline_weight, False),
                                            ("availability_aware", robust_weight, True)):
                metrics = _metrics(_distances(g_img, g_clin, qi, qc, weight, adaptive), g_y, t_y)
                rows.append({"seed": seed, "condition": condition, "model": model, **metrics})
    summary = {}
    for model in ("global_fusion", "availability_aware"):
        summary[model] = {}
        for condition in CONDITIONS:
            subset = [r for r in rows if r["model"] == model and r["condition"] == condition]
            summary[model][condition] = {}
            for metric in ("precision_at_5", "mean_reciprocal_rank"):
                values = np.array([r[metric] for r in subset]); margin = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
                summary[model][condition][metric] = {"mean": float(values.mean()),
                    "ci95_low": float(values.mean()-margin), "ci95_high": float(values.mean()+margin)}
    base = summary["global_fusion"]["mixed_missing"]["precision_at_5"]["mean"]
    proposed = summary["availability_aware"]["mixed_missing"]["precision_at_5"]["mean"]
    report = {"protocol": {"seeds": list(SEEDS), "gallery": 600, "validation": 300, "test": 400,
                           "missing_rate": .35, "weight_tuning": "validation only"},
              "tuned_weights": tuned, "summary": summary,
              "primary_comparison": {"metric": "mixed-missing precision@5", "baseline": base,
                                     "proposed": proposed, "absolute_gain": proposed-base}}
    return rows, report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("outputs/heavy"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows, report = run()
    (args.output / "heavy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report["primary_comparison"], indent=2))


if __name__ == "__main__": main()
