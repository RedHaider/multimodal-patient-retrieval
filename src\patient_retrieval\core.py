from __future__ import annotations

import numpy as np


def make_cohort(n: int = 500, seed: int = 27):
    rng = np.random.default_rng(seed)
    subtype = rng.integers(0, 4, n)
    # Moderate separation avoids a nearly perfect toy task and leaves room for
    # each modality to contribute complementary evidence.
    image_centers = rng.normal(size=(4, 12)) * 0.65
    clinical_centers = rng.normal(size=(4, 5)) * 0.6
    image = image_centers[subtype] + rng.normal(scale=1.2, size=(n, 12))
    clinical = clinical_centers[subtype] + rng.normal(scale=1.0, size=(n, 5))
    return image, clinical, subtype


def _normalize(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def distance_matrix(image, clinical, image_weight: float):
    image, clinical = _normalize(image), _normalize(clinical)
    di = ((image[:, None] - image[None, :]) ** 2).mean(axis=2)
    dc = ((clinical[:, None] - clinical[None, :]) ** 2).mean(axis=2)
    d = image_weight * di + (1.0 - image_weight) * dc
    np.fill_diagonal(d, np.inf)
    return d


def retrieval_metrics(distance, labels, k: int = 5):
    order = np.argsort(distance, axis=1)
    matches = labels[order] == labels[:, None]
    precision = matches[:, :k].mean()
    first = np.argmax(matches, axis=1) + 1
    has_match = matches.any(axis=1)
    reciprocal = np.where(has_match, 1.0 / first, 0.0)
    return {"precision_at_5": float(precision), "mean_reciprocal_rank": float(reciprocal.mean())}
