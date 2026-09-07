
from typing import Any

import pandas as pd

from app.data.keys import analyze_key_candidate
from app.data.model import ColumnProfile, DatasetProfile
from app.data.schema import classify_column


def _safe_float(value: Any) -> float | None:
    """Convert a value to float safely."""

    if pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_values(
    series: pd.Series,
    limit: int = 5,
) -> list[Any]:
    """Return a small sample of non-null values."""

    values = series.dropna().head(limit).tolist()

    result = []

    for value in values:
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass

        result.append(value)

    return result


def profile_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> DatasetProfile:
    """
    Generate deterministic metadata for a DataFrame.

    The profiler collects:
    - Dataset-level statistics
    - Column data types
    - Semantic column types
    - Null statistics
    - Unique-value statistics
    - Sample values
    - Numeric statistics
    - Key characteristics
    """

    row_count = len(dataframe)
    column_count = len(dataframe.columns)

    duplicate_row_count = int(
        dataframe.duplicated().sum()
    )

    column_profiles: list[ColumnProfile] = []

    for column in dataframe.columns:
        series = dataframe[column]

        null_count = int(
            series.isna().sum()
        )

        if row_count > 0:
            null_percentage = (
                null_count / row_count
            ) * 100

            unique_percentage = (
                series.nunique(dropna=True)
                / row_count
            ) * 100
        else:
            null_percentage = 0.0
            unique_percentage = 0.0

        semantic_type = classify_column(
            column_name=str(column),
            series=series,
        )

        key_profile = analyze_key_candidate(
            column_name=str(column),
            series=series,
        )

        profile = ColumnProfile(
            name=str(column),
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            nullable=bool(series.isna().any()),
            null_count=null_count,
            null_percentage=round(
                null_percentage,
                4,
            ),
            unique_count=int(
                series.nunique(dropna=True)
            ),
            unique_percentage=round(
                unique_percentage,
                4,
            ),
            sample_values=_sample_values(series),
            key_profile=key_profile,
        )

        if pd.api.types.is_numeric_dtype(series):
            profile.min_value = _safe_float(
                series.min()
            )

            profile.max_value = _safe_float(
                series.max()
            )

            profile.mean = _safe_float(
                series.mean()
            )

            profile.median = _safe_float(
                series.median()
            )

            profile.standard_deviation = _safe_float(
                series.std()
            )

        column_profiles.append(profile)

    return DatasetProfile(
        dataset_name=dataset_name,
        row_count=row_count,
        column_count=column_count,
        duplicate_row_count=duplicate_row_count,
        columns=column_profiles,
    )

