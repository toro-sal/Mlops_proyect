"""
Herramientas de limpieza de datos (OOP)
======================================

Este módulo define clases reutilizables para limpiar el dataset crudo
`insurance_company_modified.csv`.  La idea es encapsular las reglas
de normalización y limpieza dentro de objetos configurables.

Ejemplo de uso básico
---------------------

```python
from pathlib import Path
import pandas as pd
from src.data_proc.data_cleaner import CleanConfig, DataCleaner

config = CleanConfig(drop_na_thresh=0.95, outlier_z=4.0)
cleaner = DataCleaner(config)

df_raw = cleaner.load(Path("data/raw/insurance_company_modified.csv"))
df_clean = cleaner.clean(df_raw)
cleaner.save(df_clean, Path("data/interim/insurance_clean.parquet"))
```
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_MISSING_TOKENS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "?",
    "unknown",
    "missing",
    "not available",
    "invalid",
    "bad",
    "--",
}


@dataclass
class CleanConfig:
    """Parámetros de limpieza que pueden cargarse desde YAML/config."""

    drop_na_thresh: float = 0.95
    outlier_z: float = 4.0
    numeric_like_ratio: float = 0.8
    extra_missing_tokens: Iterable[str] = field(default_factory=list)
    coerce_float: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.drop_na_thresh <= 1:
            raise ValueError("drop_na_thresh debe estar en [0, 1].")
        if self.outlier_z is not None and self.outlier_z <= 0:
            raise ValueError("outlier_z debe ser positivo.")
        if not 0 <= self.numeric_like_ratio <= 1:
            raise ValueError("numeric_like_ratio debe estar en [0, 1].")

    @property
    def missing_tokens(self) -> set:
        tokens = {t.strip().lower() for t in DEFAULT_MISSING_TOKENS}
        tokens.update({t.strip().lower() for t in self.extra_missing_tokens})
        return tokens


class DataCleaner:
    """Limpia un DataFrame aplicando reglas parametrizables."""

    def __init__(self, config: Optional[CleanConfig] = None) -> None:
        self.config = config or CleanConfig()
        self.report_: Dict[str, List[str]] = {}

    # -------------------- API pública -------------------- #
    def load(self, path: Path, **read_csv_kwargs) -> pd.DataFrame:
        """
        Lee el CSV original.  Se usan `low_memory=False` y `keep_default_na=False`
        para evitar que pandas realice conversiones automáticas indeseadas.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        default_kwargs = {
            "low_memory": False,
            "keep_default_na": False,
            "dtype": str,
        }
        default_kwargs.update(read_csv_kwargs)

        logger.info("Leyendo datos crudos desde %s", path)
        df = pd.read_csv(path, **default_kwargs)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ejecuta las transformaciones de limpieza sobre el DataFrame."""
        logger.info("Iniciando rutina de limpieza (filas=%d, columnas=%d)", len(df), len(df.columns))

        working_df = df.copy()
        working_df = self._normalize_column_names(working_df)
        working_df = self._strip_object_columns(working_df)
        working_df = self._standardize_missing_tokens(working_df)
        working_df = self._coerce_numeric_columns(working_df)

        if self.config.outlier_z is not None:
            working_df = self._mask_outliers(working_df)

        working_df = self._drop_sparse_columns(working_df)

        logger.info("Limpieza terminada (filas=%d, columnas=%d)", len(working_df), len(working_df.columns))
        return working_df

    def save(self, df: pd.DataFrame, path: Path) -> Path:
        """Guarda el DataFrame en CSV o Parquet según la extensión del path."""
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)

        ext = path.suffix.lower()
        logger.info("Guardando dataset limpio en %s", path)
        if ext == ".parquet":
            df.to_parquet(path, index=False)
        elif ext in {".csv", ".txt"}:
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Extensión de archivo no soportada: {ext}")
        return path

    # -------------------- helpers internos -------------------- #
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera nombres únicos estandarizados para las columnas."""
        new_cols: List[str] = []
        counts: Dict[str, int] = {}
        for raw_col in df.columns:
            norm = self._slugify(str(raw_col))
            if norm in counts:
                counts[norm] += 1
                norm = f"{norm}_{counts[norm]}"
            else:
                counts[norm] = 0
            new_cols.append(norm)

        df = df.copy()
        df.columns = new_cols
        self.report_["renamed_columns"] = new_cols
        return df

    def _strip_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina espacios extra de columnas tipo string."""
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
        return df

    def _standardize_missing_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reemplaza tokens predefinidos por NaN."""
        df = df.copy()
        tokens = self.config.missing_tokens
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                series = df[col]
                lowered = series.str.lower()
                mask = lowered.isin(tokens)
                series = series.where(~mask, other=np.nan)
                df[col] = series.replace({"": np.nan})
        return df

    def _coerce_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte columnas string a numéricas si la mayoría de los valores
        pueden interpretarse como números.
        """
        if not self.config.coerce_float:
            return df

        df = df.copy()
        numeric_cols: List[str] = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
                continue

            if not pd.api.types.is_object_dtype(df[col]):
                continue

            ser = df[col]
            coerced = pd.to_numeric(ser, errors="coerce")
            non_missing = ser.notna().sum()
            if non_missing == 0:
                continue

            numeric_ratio = coerced.notna().sum() / non_missing
            if numeric_ratio >= self.config.numeric_like_ratio:
                df[col] = coerced
                numeric_cols.append(col)

        self.report_["numeric_columns"] = numeric_cols
        return df

    def _mask_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sustituye outliers (|z| > threshold) por NaN."""
        df = df.copy()
        threshold = self.config.outlier_z
        masked_cols: List[str] = []

        for col in df.select_dtypes(include=[np.number]).columns:
            ser = df[col]
            std = ser.std(skipna=True)
            mean = ser.mean(skipna=True)

            if std in (0, None) or np.isnan(std) or std == 0:
                continue

            z_scores = (ser - mean) / std
            mask = z_scores.abs() > threshold
            if mask.any():
                masked_cols.append(col)
                df.loc[mask, col] = np.nan

        self.report_["outlier_masked_cols"] = masked_cols
        return df

    def _drop_sparse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina columnas con demasiados valores faltantes."""
        if self.config.drop_na_thresh >= 1:
            return df

        df = df.copy()
        ratios = df.isna().mean()
        to_drop = ratios[ratios > self.config.drop_na_thresh].index.tolist()
        if to_drop:
            df = df.drop(columns=to_drop)
        self.report_["dropped_columns"] = to_drop
        return df

    # -------------------- utilidades -------------------- #
    @staticmethod
    def _slugify(name: str) -> str:
        """
        Convierte cualquier string en un identificador apto como nombre de columna.
        Se reemplazan símbolos por `_` y se evita comenzar con dígitos.
        """
        name = name.strip().lower()
        name = re.sub(r"[^\w]+", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")

        if not name:
            name = "col"
        if name[0].isdigit():
            name = f"col_{name}"
        return name


class DataCleaningPipeline:
    """
    Pequeño orquestador para ejecutar la limpieza end-to-end:
    lee -> limpia -> guarda y expone un resumen.
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        cleaner: Optional[DataCleaner] = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else None
        self.cleaner = cleaner or DataCleaner()
        self.summary_: Dict[str, List[str]] = {}

    def run(self) -> pd.DataFrame:
        """Ejecuta el flujo completo y devuelve el DataFrame limpio."""
        df_raw = self.cleaner.load(self.input_path)
        df_clean = self.cleaner.clean(df_raw)

        if self.output_path:
            self.cleaner.save(df_clean, self.output_path)

        self.summary_ = self.cleaner.report_
        return df_clean

    def describe(self) -> Dict[str, List[str]]:
        """Devuelve el reporte de operaciones realizadas."""
        return self.summary_
