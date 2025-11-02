from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelConfig:
    """
    Configuración central para el entrenamiento de modelos supervisados.

    Permite cargar parámetros directamente desde el `params.yaml` del proyecto
    para mantener una única fuente de verdad entre notebooks y scripts.
    """

    data_path: Optional[Path] = None
    target: str = "caravan"
    drop_suffixes: Tuple[str, ...] = ("_label",)
    binarize_target: bool = True
    positive_threshold: float = 0.0
    fillna_value: float = 0.0
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    algorithm: str = "logreg"
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    scale_numeric: bool = True
    scoring: str = "roc_auc"
    mlflow_enabled: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None
    mlflow_tags: Dict[str, str] = field(default_factory=dict)
    mlflow_registered_model_name: Optional[str] = None
    mlflow_artifact_path: str = "model"
    model_registry_dir: Optional[Path] = None

    @classmethod
    def from_params(cls, params: Dict[str, Any], project_root: Optional[Path] = None) -> "ModelConfig":
        paths_cfg = params.get("paths", {})
        preprocess_cfg = params.get("preprocess", {})
        model_cfg = params.get("model", {})
        eda_cfg = params.get("eda", {})
        tracking_cfg = params.get("tracking", {})
        mlflow_cfg = tracking_cfg.get("mlflow", {})
        registry_cfg = tracking_cfg.get("registry", {})

        data_rel = paths_cfg.get("interim")
        data_path = None
        if data_rel:
            data_path = Path(data_rel)
            if project_root:
                data_path = (project_root / data_path).resolve()

        algorithm = model_cfg.get("algorithm", "logreg")
        algorithm_params = model_cfg.get(algorithm, {})

        target = model_cfg.get("target") or eda_cfg.get("target") or "caravan"
        registry_dir = registry_cfg.get("dir")
        if registry_dir:
            registry_dir = Path(registry_dir)
            if project_root:
                registry_dir = (project_root / registry_dir).resolve()

        return cls(
            data_path=data_path,
            target=target,
            drop_suffixes=tuple(model_cfg.get("drop_label_suffixes", ("_label",))),
            binarize_target=model_cfg.get("binarize_target", True),
            positive_threshold=model_cfg.get("positive_threshold", 0.0),
            fillna_value=model_cfg.get("fillna_value", 0.0),
            test_size=preprocess_cfg.get("test_size", 0.2),
            random_state=preprocess_cfg.get("random_state", 42),
            stratify=preprocess_cfg.get("stratify", True),
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            scale_numeric=preprocess_cfg.get("scale_numeric", True),
            scoring=model_cfg.get("scoring", "roc_auc"),
            mlflow_enabled=mlflow_cfg.get("enabled", False),
            mlflow_tracking_uri=mlflow_cfg.get("tracking_uri"),
            mlflow_experiment_name=mlflow_cfg.get("experiment_name") or mlflow_cfg.get("experiment"),
            mlflow_run_name=mlflow_cfg.get("run_name"),
            mlflow_tags=mlflow_cfg.get("tags", {}),
            mlflow_registered_model_name=mlflow_cfg.get("registered_model_name"),
            mlflow_artifact_path=mlflow_cfg.get("artifact_path", "model"),
            model_registry_dir=registry_dir,
        )

    @classmethod
    def from_yaml(cls, path: Path, project_root: Optional[Path] = None) -> "ModelConfig":
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML es requerido para cargar la configuración desde YAML. Instálalo o usa ModelConfig.from_params."
            ) from exc

        params = yaml.safe_load(Path(path).read_text())
        if project_root is None:
            project_root = Path(path).resolve().parent.parent
        return cls.from_params(params, project_root=project_root)
