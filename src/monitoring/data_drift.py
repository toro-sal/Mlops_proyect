from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

from src.modeling import ModelConfig, ModelTrainer
from src.modeling.metrics import compute_classification_metrics
from src.utils.seed import set_global_seed


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("PyYAML es requerido para el monitoreo de drift.") from exc

    return yaml.safe_load(path.read_text())


def _load_registry(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_clean_dataset(config: ModelConfig) -> pd.DataFrame:
    if config.data_path is None:
        raise ValueError("data_path no está definido en config/params.yaml. Ejecuta el pipeline primero.")
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el dataset limpio en {data_path} para monitoreo de drift.")
    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def _simulate_shift(features: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    drift = features.copy()
    if drift.empty:
        return drift

    numeric_cols = drift.columns.tolist()
    noise = rng.normal(0, severity, size=drift[numeric_cols].shape)
    drift[numeric_cols] = drift[numeric_cols] * (1 + severity) + noise

    # Simula faltantes en las primeras columnas
    cols_to_perturb = numeric_cols[: max(1, len(numeric_cols) // 8)]
    for col in cols_to_perturb:
        mask = rng.random(len(drift)) < min(0.4, severity + 0.1)
        drift.loc[mask, col] = drift[col].median()
    return drift


def _align_features(df: pd.DataFrame, feature_names: Tuple[str, ...]) -> pd.DataFrame:
    return df.reindex(columns=list(feature_names), fill_value=0.0)


def _predict_proba(pipeline, data: pd.DataFrame):
    if not hasattr(pipeline, "predict_proba"):
        return None
    raw = pipeline.predict_proba(data)
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.ravel()


def run_drift_detection(config_path: Path, registry_path: Path, output_dir: Path, severity: float = 0.25) -> Dict[str, Any]:
    params = _load_yaml(config_path)
    project_root = config_path.resolve().parent.parent
    model_config = ModelConfig.from_params(params, project_root=project_root)
    set_global_seed(model_config.random_state)

    clean_df = _load_clean_dataset(model_config)
    trainer = ModelTrainer(model_config)
    features, target = trainer.prepare_features(clean_df)

    registry = _load_registry(registry_path)
    feature_names = tuple(registry.get("feature_names", list(features.columns)))
    artifact_path = Path(registry.get("model_artifact") or registry.get("model_path"))
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No se encontró el artefacto del modelo en {artifact_path}. Corre el pipeline antes de monitorear drift."
        )
    pipeline = load(artifact_path)

    features = _align_features(features, feature_names)
    baseline_metrics = compute_classification_metrics(target, pipeline.predict(features), _predict_proba(pipeline, features))

    rng = np.random.default_rng(model_config.random_state)
    drift_features = _simulate_shift(features, severity, rng)
    drift_metrics = compute_classification_metrics(
        target, pipeline.predict(drift_features), _predict_proba(pipeline, drift_features)
    )

    threshold_cfg = params.get("monitoring", {}).get("drift", {})
    roc_auc_threshold = float(threshold_cfg.get("roc_auc_drop_threshold", 0.05))
    baseline_roc = float(baseline_metrics.metrics.get("roc_auc", float("nan")))
    drift_roc = float(drift_metrics.metrics.get("roc_auc", float("nan")))
    delta_roc = baseline_roc - drift_roc
    alert = delta_roc >= roc_auc_threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(registry_path),
        "artifact": str(artifact_path),
        "severity": severity,
        "baseline_metrics": baseline_metrics.metrics,
        "drift_metrics": drift_metrics.metrics,
        "delta": {"roc_auc": delta_roc},
        "thresholds": {"roc_auc_drop_threshold": roc_auc_threshold},
        "alert": alert,
        "action": "Revisar pipeline de features y planificar reentrenamiento" if alert else "Monitor continuo",
    }

    sample_path = output_dir / "drift_sample.csv"
    drift_features.to_csv(sample_path, index=False)
    payload["drift_sample"] = str(sample_path)

    plot_path = _plot_feature_shift(features, drift_features, feature_names, output_dir)
    if plot_path:
        payload["drift_plot"] = str(plot_path)

    report_path = output_dir / "drift_report.json"
    report_path.write_text(json.dumps(payload, indent=2))
    return payload


def _plot_feature_shift(
    baseline: pd.DataFrame, drifted: pd.DataFrame, feature_names: Tuple[str, ...], output_dir: Path
) -> Optional[Path]:
    if not feature_names:
        return None
    feature = feature_names[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(baseline[feature], bins=30, alpha=0.6, label="baseline", color="#4c72b0")
    ax.hist(drifted[feature], bins=30, alpha=0.6, label="drift", color="#dd8452")
    ax.set_title(f"Distribución de {feature}: baseline vs drifted")
    ax.legend()
    ax.set_xlabel(feature)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "feature_shift.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simula y detecta data drift sobre el modelo desplegable.")
    parser.add_argument("--config", default="config/params.yaml", help="Ruta al archivo params.yaml.")
    parser.add_argument(
        "--registry", default="models/registry/latest.json", help="Ruta al JSON con metadata del último modelo."
    )
    parser.add_argument("--out", default="reports/drift", help="Carpeta donde guardar el reporte de drift.")
    parser.add_argument("--severity", type=float, default=0.25, help="Magnitud del shift simulado (0-1).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run_drift_detection(Path(args.config), Path(args.registry), Path(args.out), severity=args.severity)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}")
        return 1
    print("[OK] Reporte de drift generado.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
