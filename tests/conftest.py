from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """Small but heterogeneous dataset to exercise cleaning + modeling."""
    rng = np.random.default_rng(1234)
    rows = 60

    caravan = ((np.arange(rows) % 4) == 0).astype(int)
    income_values = [str(2500 + i * 37) for i in range(rows)]
    # Inject some missing-token variants
    income_values[1] = ""
    income_values[2] = "?"
    income_values[3] = "None"

    region_values = [" North" if i % 2 == 0 else "SOUTH " for i in range(rows)]
    region_values[0] = "NA"

    mostly_missing = [np.nan] * rows
    mostly_missing[-1] = "42"

    df = pd.DataFrame(
        {
            "Customer ID": np.arange(1, rows + 1),
            "Income ": income_values,
            "pastclaims": rng.normal(1.5, 0.3, size=rows).round(2).astype(str),
            "CARAVAN": caravan,
            "Region Label": region_values,
            "campaign_label": ["promo" if i % 2 == 0 else "control" for i in range(rows)],
            "mostly_missing": mostly_missing,
            "text_numeric ": [f"{2 + (i % 4) * 0.5:.1f}" for i in range(rows)],
        }
    )
    df.loc[5, "campaign_label"] = "unknown"
    df.loc[6, "campaign_label"] = ""
    return df


@pytest.fixture
def tmp_project(tmp_path, sample_raw_dataframe: pd.DataFrame) -> Dict[str, Path]:
    """Creates an isolated project layout with params + raw data for integration tests."""
    project_root = tmp_path / "ml_project"
    raw_path = project_root / "data" / "raw" / "synthetic.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    sample_raw_dataframe.to_csv(raw_path, index=False)

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "paths": {
            "raw": str(raw_path.relative_to(project_root)),
            "interim": "data/interim/clean.csv",
            "report_html": "reports/eda_clean.html",
        },
        "eda": {"target": "caravan"},
        "clean": {"drop_na_thresh": 0.95, "outlier_z": 4.0},
        "preprocess": {
            "scale_numeric": True,
            "test_size": 0.25,
            "random_state": 7,
            "stratify": True,
        },
        "model": {
            "algorithm": "logreg",
            "target": "caravan",
            "drop_label_suffixes": ["_label"],
            "logreg": {"C": 1.0, "max_iter": 300, "solver": "lbfgs"},
        },
        "tracking": {
            "mlflow": {"enabled": False},
            "registry": {"dir": "models/registry"},
        },
    }

    config_path = config_dir / "params.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False))

    return {
        "project_root": project_root,
        "config_path": config_path,
        "raw_path": raw_path,
    }
