from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
import pandas as pd

@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    clean_dir: Path
    interim: Path
    processed: Path
    report_html: Path
    report_css: Path
    dictionary: Path | None = None
    csv_input: Path | None = None
    csv_output: Path | None = None
    report_output: Path | None = None

class ConfigLoader:
    REQUIRED_PATH_KEYS = {"data_dir", "raw_dir", "clean_dir", "interim", "processed", "report_html", "report_css"}

    def __init__(self, yaml_path: Path | None = None, project_root: Path | None = None):
        if project_root is None:
            cwd = Path.cwd()
            project_root = cwd.parent if cwd.name.lower() == "notebooks" else cwd
        self.project_root = project_root.resolve()

        if yaml_path is None:
            yaml_path = self.project_root / "config" / "params.yaml"
        self.yaml_path = yaml_path.resolve()

        self._cfg = self._load_yaml()
        self.paths = self._build_paths()

    def _load_yaml(self) -> dict:
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"No se encontró el YAML: {self.yaml_path}")
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"El YAML está vacío o mal formateado: {self.yaml_path}")
        if "paths" not in cfg:
            raise KeyError("El YAML no contiene la sección 'paths'.")
        return cfg

    def _build_paths(self) -> ProjectPaths:
        p = self._cfg["paths"]
        missing = self.REQUIRED_PATH_KEYS - set(p.keys())
        if missing:
            raise KeyError(f"Faltan claves en 'paths': {sorted(missing)}")

        data_dir   = self.project_root / p["data_dir"]
        raw_dir    = self.project_root / p["raw_dir"]
        clean_dir  = self.project_root / p["clean_dir"]
        interim    = self.project_root / p["interim"]
        processed  = self.project_root / p["processed"]
        report_html= self.project_root / p["report_html"]
        report_css = self.project_root / p["report_css"]

        f = self._cfg.get("files", {})
        dictionary    = self.project_root / f["dictionary"]     if "dictionary"    in f else None
        csv_input     = self.project_root / f["csv_input"]      if "csv_input"     in f else None
        csv_output    = self.project_root / f["csv_output"]     if "csv_output"    in f else None
        report_output = self.project_root / f["report_output"]  if "report_output" in f else None

        return ProjectPaths(
            project_root=self.project_root,
            data_dir=data_dir,
            raw_dir=raw_dir,
            clean_dir=clean_dir,
            interim=interim,
            processed=processed,
            report_html=report_html,
            report_css=report_css,
            dictionary=dictionary,
            csv_input=csv_input,
            csv_output=csv_output,
            report_output=report_output,
        )

    def ensure_dirs(self) -> None:
        self.paths.clean_dir.mkdir(parents=True, exist_ok=True)
        self.paths.interim.parent.mkdir(parents=True, exist_ok=True)
        self.paths.processed.parent.mkdir(parents=True, exist_ok=True)
        self.paths.report_html.parent.mkdir(parents=True, exist_ok=True)
        self.paths.report_css.parent.mkdir(parents=True, exist_ok=True)

    def list_raw(self, pattern: str = "*") -> list[Path]:
        return sorted(self.paths.raw_dir.glob(pattern))

    def read_csv_input(self, **read_csv_kwargs) -> pd.DataFrame:
        if self.paths.csv_input is None:
            raise KeyError("No se definió 'files.csv_input' en params.yaml.")
        if not self.paths.csv_input.exists():
            raise FileNotFoundError(f"No existe el archivo: {self.paths.csv_input}")
        return pd.read_csv(self.paths.csv_input, **read_csv_kwargs)
