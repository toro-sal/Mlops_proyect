from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_proc.data_cleaner import CleanConfig, DataCleaner


def test_data_cleaner_coerces_numeric_and_drops_sparse(sample_raw_dataframe: pd.DataFrame) -> None:
    config = CleanConfig(drop_na_thresh=0.95, numeric_like_ratio=0.6)
    cleaner = DataCleaner(config)

    cleaned = cleaner.clean(sample_raw_dataframe)

    # Column names get slugified; Income -> income, Region Label -> region_label.
    assert "income" in cleaned.columns
    assert np.issubdtype(cleaned["income"].dtype, np.number)
    assert "region_label" in cleaned.columns
    assert cleaned["region_label"].isnull().sum() > 0  # Missing tokens converted to NaN

    # Column with >95% NaN should be removed.
    assert "mostly_missing" not in cleaned.columns
    assert "mostly_missing" in cleaner.report_.get("dropped_columns", [])


def test_data_cleaner_reports_numeric_features(sample_raw_dataframe: pd.DataFrame) -> None:
    cleaner = DataCleaner(CleanConfig(drop_na_thresh=0.99, numeric_like_ratio=0.5))
    cleaner.clean(sample_raw_dataframe)
    numeric_cols = cleaner.report_.get("numeric_columns", [])

    assert "income" in numeric_cols
    assert "pastclaims" in numeric_cols
