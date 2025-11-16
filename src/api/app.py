from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from joblib import load
from pydantic import BaseModel, Field, field_validator

from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)
SERVICE_SEED = int(os.getenv("MODEL_SERVICE_SEED", "42"))
set_global_seed(SERVICE_SEED)


class PredictionRequest(BaseModel):
    """Esquema de entrada para /predict."""

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1, description="Lista de registros con las mismas columnas que el dataset limpio."
    )
    return_probabilities: bool = Field(
        default=False,
        description="Si es True y el modelo lo soporta, añade probabilidades a la respuesta.",
    )

    @field_validator("records")
    @classmethod
    def _ensure_records(cls, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not value:
            raise ValueError("Debes enviar al menos un registro en 'records'.")
        return value


class PredictionResponse(BaseModel):
    model_version: str
    algorithm: str
    predictions: List[int]
    probabilities: Optional[List[float]] = None
    feature_names: List[str]
    reference_metrics: Dict[str, float]
    interpretations: List[str]


class ModelServiceState:
    """Carga diferida del modelo/metadata para exponer predicciones vía FastAPI."""

    def __init__(self) -> None:
        metadata_path = os.getenv("MODEL_REGISTRY_METADATA_PATH", "models/registry/latest.json")
        self.metadata_path = Path(metadata_path)
        artifact_override = os.getenv("MODEL_ARTIFACT_PATH")
        self._artifact_override = Path(artifact_override) if artifact_override else None

        self.metadata: Optional[Dict[str, Any]] = None
        self.pipeline = None
        self._metadata_mtime: Optional[float] = None

    def configure(self, metadata_path: Path) -> None:
        """Permite reconfigurar la ruta del registry (útil en pruebas)."""
        self.metadata_path = Path(metadata_path)
        self._metadata_mtime = None
        self.metadata = None
        self.pipeline = None

    def ensure_loaded(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de registry en {self.metadata_path}. Ejecuta el pipeline primero."
            )

        current_mtime = self.metadata_path.stat().st_mtime
        if self.metadata is not None and self._metadata_mtime == current_mtime:
            return

        metadata = json.loads(self.metadata_path.read_text())
        artifact_path = self._resolve_artifact(metadata)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"El artefacto del modelo no existe en {artifact_path}. Ejecuta el entrenamiento nuevamente."
            )

        logger.info("Cargando modelo desde %s", artifact_path)
        self.pipeline = load(artifact_path)
        self.metadata = metadata
        self._metadata_mtime = current_mtime

    def _resolve_artifact(self, metadata: Dict[str, Any]) -> Path:
        if self._artifact_override:
            return self._artifact_override

        artifact = metadata.get("model_artifact") or metadata.get("model_path")
        if not artifact:
            raise ValueError("El registry no contiene la clave 'model_artifact'.")
        return Path(artifact)


state = ModelServiceState()
@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        state.ensure_loaded()
    except FileNotFoundError as exc:
        logger.warning("No se pudo cargar el modelo en el arranque: %s", exc)
    yield


app = FastAPI(
    title="Insurance Model Serving API",
    version="0.1.0",
    description=(
        "Servicio FastAPI que carga el último modelo registrado y expone un endpoint /predict. "
        "Consulta /docs para el schema interactivo (OpenAPI)."
    ),
    lifespan=lifespan,
)


def _build_feature_frame(records: List[Dict[str, Any]], feature_names: List[str]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame = frame.reindex(columns=feature_names, fill_value=np.nan)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    fill_value = float(os.getenv("MODEL_SERVICE_FILL_VALUE", "0"))
    return numeric.fillna(fill_value)


def _interpret_prediction(pred: int, prob: Optional[float]) -> str:
    if prob is not None:
        percentage = f"{prob * 100:.1f}%"
        if pred >= 1:
            return f"Alta probabilidad de contar con póliza caravan (confianza {percentage})."
        return f"Baja probabilidad de póliza caravan (confianza {percentage})."
    if pred >= 1:
        return "Alta probabilidad de contar con póliza caravan."
    return "Baja probabilidad de póliza caravan."


@app.get("/health")
def healthcheck() -> Dict[str, Any]:
    metadata = state.metadata
    return {
        "status": "ok" if metadata else "model_not_loaded",
        "model_version": metadata.get("version") if metadata else None,
        "algorithm": metadata.get("algorithm") if metadata else None,
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        state.ensure_loaded()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    metadata = state.metadata or {}
    pipeline = state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="El modelo no está listo todavía.")

    feature_names: List[str] = metadata.get("feature_names") or []
    if not feature_names:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No hay feature_names en metadata.")

    features = _build_feature_frame(payload.records, feature_names)

    predictions = pipeline.predict(features)
    pred_list = predictions.tolist()
    probabilities: Optional[List[float]] = None
    if payload.return_probabilities and hasattr(pipeline, "predict_proba"):
        raw = pipeline.predict_proba(features)
        if isinstance(raw, np.ndarray):
            if raw.ndim == 2 and raw.shape[1] >= 2:
                probabilities = raw[:, 1].tolist()
            else:
                probabilities = raw.ravel().tolist()
        else:
            probabilities = list(raw)

    interpretations: List[str] = []
    if probabilities is not None:
        for pred_value, prob_value in zip(pred_list, probabilities):
            prob_number = float(prob_value) if prob_value is not None else None
            interpretations.append(_interpret_prediction(int(pred_value), prob_number))
    else:
        for pred_value in pred_list:
            interpretations.append(_interpret_prediction(int(pred_value), None))

    return PredictionResponse(
        model_version=str(metadata.get("version")),
        algorithm=str(metadata.get("algorithm")),
        predictions=[int(v) for v in pred_list],
        probabilities=probabilities,
        feature_names=feature_names,
        reference_metrics=metadata.get("metrics", {}),
        interpretations=interpretations,
    )
