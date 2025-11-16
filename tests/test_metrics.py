from __future__ import annotations

import numpy as np
import pytest

from src.modeling.metrics import compute_classification_metrics


def test_compute_classification_metrics_with_proba() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.25, 0.2, 0.85, 0.7])

    result = compute_classification_metrics(y_true, y_pred, y_proba)

    metrics = result.metrics
    assert pytest.approx(metrics["accuracy"], rel=1e-5) == 4 / 6
    assert metrics["roc_auc"] > 0.5
    assert metrics["log_loss"] > 0
    assert set(result.confusion_matrix.columns) == {"pred_0", "pred_1"}
