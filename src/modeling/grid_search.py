from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.modeling import ModelConfig, ModelTrainer
from src.utils.seed import set_global_seed

DEFAULT_ALGORITHMS = ["logreg", "rf", "gb", "svc", "knn"]


def _load_params(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependencia externa
        raise ModuleNotFoundError("PyYAML es requerido para ejecutar el grid search.") from exc

    return yaml.safe_load(path.read_text())


def _resolve_algorithms(cfg: Dict[str, Any], cli_algorithms: Sequence[str] | None) -> List[str]:
    if cli_algorithms:
        return list(dict.fromkeys(cli_algorithms))

    selection = cfg.get("model_selection", {})
    algos = selection.get("algorithms")
    if isinstance(algos, list) and algos:
        return [str(a) for a in algos]
    return DEFAULT_ALGORITHMS


def _get_param_grid(cfg: Dict[str, Any], algorithm: str) -> Iterable[Dict[str, Any]]:
    selection = cfg.get("model_selection", {})
    grids = selection.get("grids", {})
    grid_spec = grids.get(algorithm)
    if not grid_spec:
        existing_params = cfg.get("model", {}).get(algorithm, {})
        return [existing_params]
    return ParameterGrid(grid_spec)


def _disable_tracking(cfg: ModelConfig) -> ModelConfig:
    """Devuelve una copia de la configuración sin tracking (MLflow/registry)."""
    return replace(cfg, mlflow_enabled=False, model_registry_dir=None)


def _build_run_name(base_name: str | None, algorithm: str) -> str:
    prefix = base_name or "grid_search"
    return f"{prefix}_{algorithm}_best"


def _register_best_models(
    config: ModelConfig,
    best_records: List[Dict[str, Any]],
    run_name: str | None = None,
) -> List[Dict[str, Any]]:
    """Reentrena y registra los mejores modelos por algoritmo."""
    persisted: List[Dict[str, Any]] = []
    for record in best_records:
        algorithm = record["algorithm"]
        params = record["params"]
        cfg = replace(
            config,
            algorithm=algorithm,
            algorithm_params=params,
            mlflow_run_name=_build_run_name(run_name or config.mlflow_run_name, algorithm),
        )
        trainer = ModelTrainer(cfg)
        result = trainer.run()
        enriched = {**record}
        enriched["model_artifact"] = result.artifacts.get("model_artifact_path")
        enriched["registry_entry"] = result.artifacts.get("model_registry_entry")
        enriched["registry_latest"] = result.artifacts.get("model_registry_latest")
        enriched["mlflow_run_id"] = result.mlflow_run_id
        enriched["mlflow_experiment_id"] = result.mlflow_experiment_id
        persisted.append(enriched)
    return persisted


def run_grid_search(
    config_path: Path,
    output_dir: Path,
    algorithms: Sequence[str] | None = None,
    register_best: bool = True,
) -> Path:
    params = _load_params(config_path)
    project_root = config_path.resolve().parent.parent
    base_config = ModelConfig.from_params(params, project_root=project_root)
    scoring = base_config.scoring or "roc_auc"
    set_global_seed(base_config.random_state)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for algorithm in _resolve_algorithms(params, algorithms):
        grid = _get_param_grid(params, algorithm)
        for combo in grid:
            combo_dict = dict(combo)
            trial_cfg = replace(_disable_tracking(base_config), algorithm=algorithm, algorithm_params=combo_dict)
            trainer = ModelTrainer(trial_cfg)
            result = trainer.run()
            metric_value = result.metrics.get(scoring)
            entry = {
                "algorithm": algorithm,
                "params": combo_dict,
                "metrics": result.metrics,
                "scoring": scoring,
                "score": float(metric_value) if metric_value is not None else float("nan"),
            }
            rows.append(entry)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Grid search no produjo resultados; revisa la configuración.")
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    csv_path = output_dir / "grid_search_results.csv"
    json_path = output_dir / "grid_search_results.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2))

    best_entry = df.iloc[0].to_dict()
    summary_path = output_dir / "best_model.json"
    summary_path.write_text(json.dumps(best_entry, indent=2))

    best_by_algorithm = (
        df.sort_values("score", ascending=False)
        .groupby("algorithm", as_index=False)
        .first()
        .to_dict(orient="records")
    )
    best_by_path = output_dir / "best_by_algorithm.json"
    best_by_path.write_text(json.dumps(best_by_algorithm, indent=2))

    if register_best and best_by_algorithm:
        persisted = _register_best_models(base_config, best_by_algorithm, run_name=base_config.mlflow_run_name)
        persisted_path = output_dir / "best_by_algorithm_registered.json"
        persisted_path.write_text(json.dumps(persisted, indent=2))

    return summary_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta grid search sobre los algoritmos configurados.")
    parser.add_argument("--config", default="config/params.yaml", help="Ruta al archivo de parámetros.")
    parser.add_argument("--output-dir", default="reports/grid_search", help="Carpeta donde guardar los resultados.")
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help="Lista opcional de algoritmos a probar (logreg, rf, gb, svc, knn, ...).",
    )
    parser.add_argument(
        "--no-register-best",
        dest="register_best",
        action="store_false",
        help="Si se especifica, no reentrena ni registra los mejores modelos por algoritmo.",
    )
    parser.set_defaults(register_best=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir)
    try:
        best_path = run_grid_search(
            config_path,
            output_dir,
            algorithms=args.algorithms,
            register_best=args.register_best,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[ERROR] {exc}")
        return 1

    print(f"[OK] Grid search completado. Mejor configuración en {best_path}.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
