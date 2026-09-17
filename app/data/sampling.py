from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class SamplingMetadata(BaseModel):
    """Metadata describing how the adaptive sample was created."""

    total_rows: int = Field(ge=0)
    sample_size: int = Field(ge=0)
    random_sample_size: int = Field(ge=0)
    representative_row_count: int = Field(ge=0)
    categorical_example_count: int = Field(ge=0)
    rare_value_example_count: int = Field(ge=0)
    strategy: str


class DatasetSample(BaseModel):
    """Small, generic dataset sample used only for dataset understanding."""

    random_rows: list[dict[str, Any]] = Field(default_factory=list)
    representative_rows: list[dict[str, Any]] = Field(default_factory=list)
    categorical_examples: dict[str, list[Any]] = Field(default_factory=dict)
    numeric_examples: dict[str, list[Any]] = Field(default_factory=dict)
    datetime_examples: dict[str, list[Any]] = Field(default_factory=dict)
    rare_value_examples: dict[str, list[Any]] = Field(default_factory=dict)
    sampling_metadata: SamplingMetadata


def build_adaptive_sample(
    dataframe: pd.DataFrame,
    *,
    random_sample_size: int = 8,
    representative_row_count: int = 8,
    categorical_example_count: int = 8,
    rare_value_example_count: int = 5,
    random_state: int = 42,
) -> DatasetSample:
    """
    Build a compact, domain-agnostic sample for LLM dataset understanding.

    IMPORTANT:
    This sample is context only. It must never be used as the source of
    final numerical claims; deterministic analysis tools still operate on
    the complete dataframe.
    """
    total_rows = len(dataframe)

    if total_rows == 0:
        metadata = SamplingMetadata(
            total_rows=0,
            sample_size=0,
            random_sample_size=0,
            representative_row_count=0,
            categorical_example_count=0,
            rare_value_example_count=0,
            strategy="empty_dataset",
        )
        return DatasetSample(sampling_metadata=metadata)

    random_n = min(max(random_sample_size, 0), total_rows)
    representative_n = min(max(representative_row_count, 0), total_rows)

    random_df = (
        dataframe.sample(n=random_n, random_state=random_state)
        if random_n
        else dataframe.iloc[0:0]
    )

    representative_df = _representative_rows(
        dataframe,
        representative_n,
    )

    categorical_examples: dict[str, list[Any]] = {}
    numeric_examples: dict[str, list[Any]] = {}
    datetime_examples: dict[str, list[Any]] = {}
    rare_value_examples: dict[str, list[Any]] = {}

    for column in dataframe.columns:
        series = dataframe[column].dropna()

        if series.empty:
            continue

        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            datetime_examples[column] = _safe_values(
                series,
                categorical_example_count,
            )
            continue

        if pd.api.types.is_numeric_dtype(dataframe[column]):
            numeric_examples[column] = _safe_values(
                series,
                categorical_example_count,
            )
            continue

        # Generic categorical/text-like values. We intentionally do not
        # hard-code business/domain labels.
        unique_values = series.drop_duplicates()
        categorical_examples[column] = _safe_values(
            unique_values,
            categorical_example_count,
        )

        rare_values = _rare_values(
            series,
            rare_value_example_count,
        )
        if rare_values:
            rare_value_examples[column] = rare_values

    random_rows = _records(random_df)
    representative_rows = _records(representative_df)

    metadata = SamplingMetadata(
        total_rows=total_rows,
        sample_size=len(random_rows) + len(representative_rows),
        random_sample_size=len(random_rows),
        representative_row_count=len(representative_rows),
        categorical_example_count=sum(
            len(values) for values in categorical_examples.values()
        ),
        rare_value_example_count=sum(
            len(values) for values in rare_value_examples.values()
        ),
        strategy=(
            "random_rows + representative_rows + "
            "value_examples + rare_value_examples"
        ),
    )

    return DatasetSample(
        random_rows=random_rows,
        representative_rows=representative_rows,
        categorical_examples=categorical_examples,
        numeric_examples=numeric_examples,
        datetime_examples=datetime_examples,
        rare_value_examples=rare_value_examples,
        sampling_metadata=metadata,
    )


def _representative_rows(
    dataframe: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    """Select rows spread across the dataframe rather than only the head."""
    if count <= 0 or dataframe.empty:
        return dataframe.iloc[0:0]

    if count >= len(dataframe):
        return dataframe.copy()

    positions = [
        round(index * (len(dataframe) - 1) / (count - 1))
        if count > 1
        else 0
        for index in range(count)
    ]
    return dataframe.iloc[positions]


def _rare_values(series: pd.Series, count: int) -> list[Any]:
    """Return a few least-frequent non-null values."""
    if count <= 0:
        return []

    frequencies = series.value_counts(dropna=True)
    if frequencies.empty:
        return []

    return _safe_values(frequencies.sort_values().head(count).index, count)


def _safe_values(values: Any, count: int) -> list[Any]:
    """Convert pandas/numpy scalar values into JSON-friendly values."""
    result: list[Any] = []

    for value in list(values)[:count]:
        if pd.isna(value):
            continue

        if isinstance(value, pd.Timestamp):
            result.append(value.isoformat())
        elif hasattr(value, "item"):
            try:
                result.append(value.item())
            except (ValueError, TypeError):
                result.append(str(value))
        else:
            result.append(value)

    return result


def _records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize rows without changing the source dataframe."""
    if dataframe.empty:
        return []

    records = dataframe.to_dict(orient="records")

    return [
        {
            str(key): (
                value.isoformat()
                if isinstance(value, pd.Timestamp)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in row.items()
        }
        for row in records
    ]
