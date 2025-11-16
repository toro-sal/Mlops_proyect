from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import tempfile

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .config import ModelConfig
from .models import (
    BaseModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    SupportVectorMachineModel,
    KNeighborsModel,
)
from .metrics import compute_classification_metrics
from src.utils.seed import set_global_seed


MODEL_REGISTRY = {
    "logreg": LogisticRegressionModel,
    "rf": RandomForestModel,
    "gb": GradientBoostingModel,
    "svc": SupportVectorMachineModel,
    "knn": KNeighborsModel,
}


@dataclass
class ModelTrainingResult:
    """Representa los artefactos principales del entrenamiento."""

    model: BaseModel
    config: ModelConfig
    metrics: Dict[str, float]
    classification_report: str
    roc_auc: float
    feature_summary: Optional[pd.DataFrame]
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    confusion_matrix: Optional[pd.DataFrame] = None


class ModelTrainer:
    """
    Orquesta el flujo completo: carga de datos, preparación, split, entrenamiento y evaluación.
    Replica la lógica del notebook 03_modeling.py pero en una API reutilizable.
    """

    def __init__(self, config: ModelConfig, model: Optional[BaseModel] = None) -> None:
        self.config = config
        self.model = model or self._build_model(config.algorithm)
        self._feature_names: Tuple[str, ...] = ()
        self._dropped_columns: Tuple[str, ...] = ()
        set_global_seed(getattr(self.config, "random_state", 42))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_model(self, algorithm: str) -> BaseModel:
        try:
            model_cls = MODEL_REGISTRY[algorithm]
        except KeyError as exc:
            raise ValueError(f"Algoritmo '{algorithm}' no soportado. Opciones: {list(MODEL_REGISTRY)}") from exc
        return model_cls(self.config)

    def load_dataset(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Carga el dataset limpio desde CSV o Parquet."""
        data_path = path or self.config.data_path
        if data_path is None:
            raise ValueError("No se especificó `data_path` en la configuración.")
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"No existe el dataset limpio en {data_path}.")

        if data_path.suffix.lower() == ".parquet":
            return pd.read_parquet(data_path)
        return pd.read_csv(data_path)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Replica la preparación realizada en el notebook (drop label cols, coerción numérica, binarización)."""
        if self.config.target not in df.columns:
            raise ValueError(f"El target '{self.config.target}' no existe en el DataFrame.")

        # Binarización del target
        y = pd.to_numeric(df[self.config.target], errors="coerce").fillna(0)
        if self.config.binarize_target:
            y = (y > self.config.positive_threshold).astype(int)

        # Columnas a descartar (labels, target original, etc.)
        drop_cols: set[str] = set()
        for suf in self.config.drop_suffixes:
            drop_cols.update(c for c in df.columns if c.endswith(suf))
        drop_cols.add(self.config.target)
        self._dropped_columns = tuple(sorted(drop_cols))

        features = df.drop(columns=list(drop_cols), errors="ignore")
        X = features.apply(pd.to_numeric, errors="coerce").fillna(self.config.fillna_value)
        self._feature_names = tuple(X.columns.astype(str))
        return X, y

    def _train_test_split(self, X, y):
        stratify = y if self.config.stratify else None
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )

    def evaluate(self, X_test, y_test) -> ModelTrainingResult:
        y_pred = self.model.predict(X_test)
        report_str = classification_report(y_test, y_pred, digits=3)

        proba = None
        if hasattr(self.model.pipeline, "predict_proba"):
            raw_proba = self.model.predict_proba(X_test)
            if isinstance(raw_proba, np.ndarray):
                if raw_proba.ndim == 2:
                    proba = raw_proba[:, 1]
                else:
                    proba = raw_proba
            else:
                proba = raw_proba

        metrics_result = compute_classification_metrics(y_test, y_pred, proba)
        metrics = metrics_result.metrics
        roc_auc = float(metrics.get("roc_auc", float("nan")))
        try:
            feature_summary = self.model.feature_importances(self._feature_names)
        except NotImplementedError:
            feature_summary = None

        return ModelTrainingResult(
            model=self.model,
            config=self.config,
            metrics=metrics,
            classification_report=report_str,
            roc_auc=roc_auc,
            feature_summary=feature_summary,
            confusion_matrix=metrics_result.confusion_matrix,
        )

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def run(self, df: Optional[pd.DataFrame] = None) -> ModelTrainingResult:
        """
        Ejecuta el flujo end-to-end y devuelve las métricas junto con el modelo entrenado.
        """
        data = df if df is not None else self.load_dataset()
        X, y = self.prepare_features(data)
        X_train, X_test, y_train, y_test = self._train_test_split(X, y)
        self.model.fit(X_train, y_train)
        result = self.evaluate(X_test, y_test)
        self._track_experiment(result)
        return result

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return self._feature_names

    @property
    def dropped_columns(self) -> Tuple[str, ...]:
        return self._dropped_columns

    # ------------------------------------------------------------------ #
    # Tracking helpers (MLflow + registro local)
    # ------------------------------------------------------------------ #
    def _track_experiment(self, result: ModelTrainingResult) -> None:
        mlflow_info = self._log_mlflow(result)
        if mlflow_info:
            result.mlflow_run_id = mlflow_info.get("run_id")
            result.mlflow_experiment_id = mlflow_info.get("experiment_id")
            artifact_uri = mlflow_info.get("artifact_uri")
            if artifact_uri:
                result.artifacts["mlflow_artifact_uri"] = artifact_uri

        registry_path = self._persist_registry_entry(result)
        if registry_path:
            result.artifacts["model_registry_entry"] = str(registry_path)

    def _log_mlflow(self, result: ModelTrainingResult) -> Optional[Dict[str, str]]:
        if not getattr(self.config, "mlflow_enabled", False):
            return None

        try:
            import mlflow
            import mlflow.sklearn  # type: ignore
        except ModuleNotFoundError:
            print("[MLFLOW] mlflow no está instalado; se omite el tracking.")
            return None

        tracking_uri = getattr(self.config, "mlflow_tracking_uri", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = getattr(self.config, "mlflow_experiment_name", None) or "Default"
        _experiment = mlflow.set_experiment(experiment_name)

        run_name = getattr(self.config, "mlflow_run_name", None)
        extra_tags = {k: str(v) for k, v in getattr(self.config, "mlflow_tags", {}).items()}
        tags = {"model.algorithm": self.config.algorithm, **extra_tags}

        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.set_tags(tags)

            mlflow.log_param("target", self.config.target)
            mlflow.log_param("binarize_target", self.config.binarize_target)
            mlflow.log_param("positive_threshold", self.config.positive_threshold)
            mlflow.log_param("test_size", self.config.test_size)
            mlflow.log_param("random_state", self.config.random_state)
            mlflow.log_param("stratify", self.config.stratify)
            if self.config.data_path:
                mlflow.log_param("data_path", str(self.config.data_path))

            for key, value in self.config.algorithm_params.items():
                mlflow.log_param(f"model__{key}", value)

            for metric_name, metric_value in result.metrics.items():
                value = float(metric_value)
                if np.isnan(value):
                    continue
                mlflow.log_metric(metric_name, value)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                report_file = tmp_path / "classification_report.txt"
                report_file.write_text(result.classification_report)
                mlflow.log_artifact(str(report_file), artifact_path="reports")

                if result.confusion_matrix is not None:
                    cm_file = tmp_path / "confusion_matrix.csv"
                    result.confusion_matrix.to_csv(cm_file)
                    mlflow.log_artifact(str(cm_file), artifact_path="reports")
                    result.artifacts["mlflow_confusion_matrix"] = "reports/confusion_matrix.csv"

                meta_file = tmp_path / "config_snapshot.json"
                meta_payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "algorithm": self.config.algorithm,
                    "algorithm_params": self.config.algorithm_params,
                    "dropped_columns": self._dropped_columns,
                    "feature_names": self._feature_names,
                }
                meta_file.write_text(json.dumps(meta_payload, indent=2))
                mlflow.log_artifact(str(meta_file), artifact_path="reports")

                if result.feature_summary is not None:
                    feature_file = tmp_path / "feature_importance.csv"
                    result.feature_summary.to_csv(feature_file, index=False)
                    mlflow.log_artifact(str(feature_file), artifact_path="reports")

            if self.model.pipeline is not None:
                registered_name = getattr(self.config, "mlflow_registered_model_name", None)
                mlflow.sklearn.log_model(
                    self.model.pipeline,
                    artifact_path=getattr(self.config, "mlflow_artifact_path", "model"),
                    registered_model_name=registered_name,
                )

            return {
                "run_id": active_run.info.run_id,
                "experiment_id": active_run.info.experiment_id,
                "artifact_uri": active_run.info.artifact_uri,
            }

    def _persist_registry_entry(self, result: ModelTrainingResult) -> Optional[Path]:
        registry_dir = Path(getattr(self.config, "model_registry_dir", "")) if getattr(
            self.config, "model_registry_dir", None
        ) else None
        if registry_dir is None:
            return None

        registry_dir.mkdir(parents=True, exist_ok=True)
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        payload = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": self.config.algorithm,
            "hyperparameters": self.config.algorithm_params,
            "target": self.config.target,
            "metrics": {k: float(v) for k, v in result.metrics.items()},
            "roc_auc": float(result.roc_auc) if not np.isnan(result.roc_auc) else None,
            "mlflow_run_id": result.mlflow_run_id,
            "mlflow_experiment_id": result.mlflow_experiment_id,
            "mlflow_artifact_uri": result.artifacts.get("mlflow_artifact_uri"),
            "data_path": str(self.config.data_path) if self.config.data_path else None,
            "feature_names": list(map(str, self._feature_names)),
            "dropped_columns": list(map(str, self._dropped_columns)),
            "random_state": getattr(self.config, "random_state", None),
        }
        if result.confusion_matrix is not None:
            payload["confusion_matrix"] = {
                "index": list(map(str, result.confusion_matrix.index)),
                "columns": list(map(str, result.confusion_matrix.columns)),
                "values": result.confusion_matrix.values.tolist(),
            }

        if self.model.pipeline is not None:
            model_file = registry_dir / f"{version}_{self.config.algorithm}.joblib"
            dump(self.model.pipeline, model_file)
            payload["model_artifact"] = str(model_file)
            result.artifacts["model_artifact_path"] = str(model_file)

        file_path = registry_dir / f"{version}_{self.config.algorithm}.json"
        file_path.write_text(json.dumps(payload, indent=2))
        latest_file = registry_dir / "latest.json"
        latest_file.write_text(json.dumps(payload, indent=2))
        result.artifacts["model_registry_latest"] = str(latest_file)
        return file_path
