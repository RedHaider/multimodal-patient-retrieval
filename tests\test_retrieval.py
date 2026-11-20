import numpy as np

from patient_retrieval.core import distance_matrix, make_cohort, retrieval_metrics
from patient_retrieval.experiment import evaluate


def test_distance_is_symmetric_and_excludes_self():
    image, clinical, _ = make_cohort(20, 3)
    d = distance_matrix(image, clinical, 0.5)
    assert np.allclose(d, d.T)
    assert np.isinf(np.diag(d)).all()


def test_metrics_are_probabilities():
    results = evaluate(4)
    assert all(0 <= v <= 1 for metrics in results.values() for v in metrics.values())
