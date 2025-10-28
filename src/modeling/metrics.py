from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetricsResult:
    metrics: Dict[str, float]
    confusion_matrix: pd.DataFrame


def _build_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    index = [f"actual_{str(lbl)}" for lbl in labels]
    columns = [f"pred_{str(lbl)}" for lbl in labels]
    return pd.DataFrame(cm, index=index, columns=columns)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[Sequence[float]] = None,
) -> ClassificationMetricsResult:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    if y_proba is not None:
        proba_arr = np.asarray(y_proba, dtype=float)
        if proba_arr.ndim == 2:
            if proba_arr.shape[1] < 2:
                proba_arr = proba_arr.squeeze(axis=1)
            else:
                proba_arr = proba_arr[:, 1]
        proba_clipped = np.clip(proba_arr, 1e-15, 1 - 1e-15)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, proba_clipped))
        metrics["log_loss"] = float(log_loss(y_true_arr, proba_clipped, labels=[0, 1]))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")

    cm_df = _build_confusion_matrix(y_true_arr, y_pred_arr)

    return ClassificationMetricsResult(metrics=metrics, confusion_matrix=cm_df)
