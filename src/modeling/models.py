from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from .config import ModelConfig


class BaseModel(ABC):
    """Interfaz común para los modelos supervisados del proyecto."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.pipeline: Optional[Pipeline] = None

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Devuelve el pipeline sklearn listo para entrenar."""

    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if self.pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("El modelo actual no expone predict_proba().")
        return self.pipeline.predict_proba(X)

    def feature_importances(self, feature_names: Sequence[str], top_n: Optional[int] = 15) -> pd.DataFrame:
        """
        Devuelve un DataFrame ordenado con las variables más influyentes.
        Cada subclase decide cómo calcular la importancia (coeficientes, importancias, etc.).
        """
        raise NotImplementedError("La importancia de variables no está implementada para este modelo.")


class LogisticRegressionModel(BaseModel):
    """Replica la regresión logística balanceada utilizada en el notebook 03_modeling."""

    def build_pipeline(self) -> Pipeline:
        steps = []
        if self.config.scale_numeric:
            steps.append(("scaler", StandardScaler()))

        params = {"max_iter": 500, "class_weight": "balanced"}
        params.update(self.config.algorithm_params)
        clf = LogisticRegression(**params)

        steps.append(("clf", clf))
        return Pipeline(steps)

    def feature_importances(self, feature_names: Sequence[str], top_n: Optional[int] = 15) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        coefs = clf.coef_[0]
        data = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
                "abs_coefficient": np.abs(coefs),
            }
        ).sort_values("abs_coefficient", ascending=False)
        if top_n is not None:
            data = data.head(top_n)
        return data.reset_index(drop=True)


class RandomForestModel(BaseModel):
    """Alternativa basada en RandomForest para extender el baseline rápidamente."""

    def build_pipeline(self) -> Pipeline:
        params = {"n_estimators": 300, "max_depth": None, "class_weight": "balanced"}
        params.update(self.config.algorithm_params)
        clf = RandomForestClassifier(**params)
        return Pipeline([("clf", clf)])

    def feature_importances(self, feature_names: Sequence[str], top_n: Optional[int] = 15) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        clf: RandomForestClassifier = self.pipeline.named_steps["clf"]
        importances = clf.feature_importances_
        data = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "abs_importance": np.abs(importances),
            }
        ).sort_values("abs_importance", ascending=False)
        if top_n is not None:
            data = data.head(top_n)
        return data.reset_index(drop=True)


class GradientBoostingModel(BaseModel):
    """Modelo Gradient Boosting basado en árboles, útil para capturar relaciones no lineales."""

    def build_pipeline(self) -> Pipeline:
        params = {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 3}
        params.update(self.config.algorithm_params)
        clf = GradientBoostingClassifier(**params)
        return Pipeline([("clf", clf)])

    def feature_importances(self, feature_names: Sequence[str], top_n: Optional[int] = 15) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        clf: GradientBoostingClassifier = self.pipeline.named_steps["clf"]
        importances = clf.feature_importances_
        data = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "abs_importance": np.abs(importances),
            }
        ).sort_values("abs_importance", ascending=False)
        if top_n is not None:
            data = data.head(top_n)
        return data.reset_index(drop=True)


class SupportVectorMachineModel(BaseModel):
    """Clasificador SVM con probabilidad habilitada para compatibilidad con métricas probabilísticas."""

    def build_pipeline(self) -> Pipeline:
        steps = []
        if self.config.scale_numeric:
            steps.append(("scaler", StandardScaler()))

        params = {"probability": True}
        params.update(self.config.algorithm_params)
        clf = SVC(**params)

        steps.append(("clf", clf))
        return Pipeline(steps)


class KNeighborsModel(BaseModel):
    """Clasificador k-NN con normalización opcional de variables numéricas."""

    def build_pipeline(self) -> Pipeline:
        steps = []
        if self.config.scale_numeric:
            steps.append(("scaler", StandardScaler()))

        params = {"n_neighbors": 5, "weights": "distance"}
        params.update(self.config.algorithm_params)
        clf = KNeighborsClassifier(**params)

        steps.append(("clf", clf))
        return Pipeline(steps)
