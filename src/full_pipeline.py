from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.data_proc.data_cleaner import CleanConfig, DataCleaner
from src.eda.eda_reporter import EDAReport
from src.modeling import ModelConfig, ModelTrainer


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - entorno sin dependencia
        raise ModuleNotFoundError(
            "PyYAML es requerido para cargar la configuración del pipeline. "
            "Instálalo (pip install pyyaml) o pasa un diccionario ya cargado."
        ) from exc

    return yaml.safe_load(path.read_text())


def _resolve_columns(df: pd.DataFrame, names: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for original in names:
        if original in df.columns:
            resolved.append(original)
            continue
        slug = DataCleaner._slugify(original)  # type: ignore[attr-defined]
        if slug in df.columns:
            resolved.append(slug)
        else:
            resolved.append(original)
    return resolved


def _guess_target_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    """
    Intenta resolver la columna target partiendo de la configuración preferida y
    cayendo en alternativas conocidas del dataset (CARAVAN, caravan, etc.).
    """
    columns = list(df.columns)
    col_set = set(columns)
    lower_map = {c.lower(): c for c in columns}

    candidates: List[Optional[str]] = []
    if preferred:
        candidates.append(preferred)
        candidates.append(DataCleaner._slugify(preferred))  # type: ignore[attr-defined]

    candidates.extend(["caravan", "CARAVAN", "caravan_bin", "CARAVAN_BIN"])

    for cand in candidates:
        if not cand:
            continue
        if cand in col_set:
            return cand
        cand_low = cand.lower()
        if cand_low in lower_map:
            return lower_map[cand_low]

    # Heurística final: busca columnas que contengan 'caravan'
    for col in columns:
        if "caravan" in col.lower():
            return col

    return None


def run_full_pipeline(
    *,
    config_path: Path | str = Path("config/params.yaml"),
    project_root: Path | str | None = None,
    generate_clean_report: bool = True,
    algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline end-to-end:

    1. Carga parámetros del proyecto.
    2. Limpia el dataset crudo con `DataCleaner`.
    3. Genera (opcional) un reporte EDA sobre datos limpios.
    4. Entrena el modelo baseline mediante `ModelTrainer`.

    Devuelve un diccionario con rutas y métricas relevantes.
    """

    config_path = Path(config_path)
    if not config_path.is_absolute():
        cfg_root = project_root if project_root else Path.cwd()
        config_path = (Path(cfg_root) / config_path).resolve()

    if project_root is None:
        project_root = config_path.resolve().parent.parent
    project_root = Path(project_root).resolve()

    params = _load_yaml(config_path)
    paths_cfg = params.get("paths", {})
    eda_cfg = params.get("eda", {})
    clean_cfg = params.get("clean", {})
    model_cfg_raw = params.get("model", {})
    preferred_target = model_cfg_raw.get("target") or eda_cfg.get("target")

    raw_rel = paths_cfg.get("raw")
    if not raw_rel:
        raise ValueError("El archivo RAW no está definido en paths.raw del YAML.")
    raw_path = (project_root / raw_rel).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset crudo en {raw_path}")

    interim_rel = paths_cfg.get("interim", "data/interim/data_clean.parquet")
    interim_path = (project_root / interim_rel).resolve()
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_path, low_memory=False)
    raw_target_col = _guess_target_column(df_raw, preferred_target)
    if raw_target_col is None:
        raw_target_col = _guess_target_column(df_raw, "caravan")

    clean_config = CleanConfig(
        drop_na_thresh=clean_cfg.get("drop_na_thresh", 0.95),
        outlier_z=clean_cfg.get("outlier_z", 4.0),
        numeric_like_ratio=clean_cfg.get("numeric_like_ratio", 0.8),
        extra_missing_tokens=clean_cfg.get("extra_missing_tokens", []),
        coerce_float=clean_cfg.get("coerce_float", True),
    )

    cleaner = DataCleaner(clean_config)
    df_clean = cleaner.clean(df_raw)
    cleaner.save(df_clean, interim_path)
    clean_target_hint = DataCleaner._slugify(raw_target_col) if raw_target_col else preferred_target

    report_path = None
    if generate_clean_report:
        report_rel_raw = paths_cfg.get("report_html", "reports/eda_clean.html")
        report_rel_path = Path(report_rel_raw)
        if report_rel_path.suffix == "":
            report_rel_path = report_rel_path / "eda_clean.html"
        report_path = (project_root / report_rel_path).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fig_dir = report_path.parent / "figures"

        target_resolved = _guess_target_column(df_clean, clean_target_hint or eda_cfg.get("target"))

        id_cols = _resolve_columns(df_clean, eda_cfg.get("id_cols", []))
        datetime_cols = _resolve_columns(df_clean, eda_cfg.get("datetime_cols", []))
        cat_cols = _resolve_columns(df_clean, eda_cfg.get("cat_cols", []))

        reporter = EDAReport(
            target=target_resolved,
            id_cols=id_cols,
            datetime_cols=datetime_cols,
            cat_cols=cat_cols,
            report_title=eda_cfg.get("report_title", "EDA — Dataset limpio"),
            fig_dir=str(fig_dir),
            max_plots=eda_cfg.get("max_plots"),
            drop_suffixes=tuple(eda_cfg.get("drop_suffixes", ("_label",))),
            show_exec_summary_in_html=eda_cfg.get("show_exec_summary_in_html", True),
        )
        reporter.fit(df_clean)
        reporter.generate_html(str(report_path))

    model_config = ModelConfig.from_params(params, project_root=project_root)
    if algorithm:
        model_config.algorithm = algorithm
        algo_params = model_cfg_raw.get(algorithm, {})
        if isinstance(algo_params, dict):
            model_config.algorithm_params = algo_params
        else:
            model_config.algorithm_params = {}
    target_col = _guess_target_column(df_clean, clean_target_hint or model_config.target)
    if target_col is None:
        sample_cols = ", ".join(map(str, df_clean.columns[:8]))
        raise ValueError(
            f"No se pudo resolver la columna target. Configurado: '{model_config.target}'. "
            f"Columnas disponibles (muestra): {sample_cols}. "
            "Verifica config/model.target o eda.target."
        )
    model_config.target = target_col

    trainer = ModelTrainer(model_config)
    result = trainer.run(df_clean)

    return {
        "raw_path": raw_path,
        "clean_path": interim_path,
        "report_path": report_path,
        "model_metrics": result.metrics,
        "classification_report": result.classification_report,
        "feature_summary": result.feature_summary,
        "mlflow_run_id": result.mlflow_run_id,
        "mlflow_experiment_id": result.mlflow_experiment_id,
        "artifacts": result.artifacts,
        "algorithm": model_config.algorithm,
        "confusion_matrix": result.confusion_matrix,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline completo (limpieza + EDA + modelado).")
    parser.add_argument("--config", default="config/params.yaml", help="Ruta al archivo params.yaml.")
    parser.add_argument("--project-root", default=None, help="Carpeta raíz del proyecto (para resolver rutas).")
    parser.add_argument("--no-report", action="store_true", help="No generar el reporte EDA del dataset limpio.")
    parser.add_argument(
        "--algorithm",
        default=None,
        help="Sobrescribe el algoritmo configurado (p.ej. logreg, rf, gb, svc, knn).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        outputs = run_full_pipeline(
            config_path=Path(args.config),
            project_root=Path(args.project_root) if args.project_root else None,
            generate_clean_report=not args.no_report,
            algorithm=args.algorithm,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("[OK] Pipeline ejecutado correctamente.")
    print(f"  • Dataset crudo: {outputs['raw_path']}")
    print(f"  • Dataset limpio: {outputs['clean_path']}")
    if outputs["report_path"]:
        print(f"  • Reporte EDA limpio: {outputs['report_path']}")
    if outputs["model_metrics"]:
        print(f"  • Métricas: {outputs['model_metrics']}")
    if outputs.get("algorithm"):
        print(f"  • Algoritmo: {outputs['algorithm']}")
    if outputs.get("mlflow_run_id"):
        print(f"  • MLflow run_id: {outputs['mlflow_run_id']}")
    artifacts = outputs.get("artifacts") or {}
    if artifacts.get("model_registry_entry"):
        print(f"  • Registro del modelo: {artifacts['model_registry_entry']}")
    if artifacts.get("model_registry_latest"):
        print(f"  • Registro actual: {artifacts['model_registry_latest']}")
    if artifacts.get("mlflow_artifact_uri"):
        print(f"  • Artefactos MLflow: {artifacts['mlflow_artifact_uri']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
