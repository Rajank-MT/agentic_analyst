from __future__ import annotations

from typing import Any

import pandas as pd


SUPPORTED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
}

SUPPORTED_DERIVED_METRICS = {
    "ratio",
    "difference",
    "percentage_of_total",
}


class AnalysisEngineError(ValueError):
    """Raised when a composed analysis request is invalid."""


def _validate_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]

    if missing:
        raise AnalysisEngineError(
            f"Columns not found in dataset: {missing}"
        )


def aggregate_by_group(
    dataframe: pd.DataFrame,
    group_by: list[str],
    metrics: dict[str, str],
) -> pd.DataFrame:
    """
    Group a dataframe and calculate deterministic aggregations.

    Example:
        group_by=["Category"]
        metrics={
            "Revenue": "sum",
            "Margin": "mean",
        }
    """

    if not group_by:
        raise AnalysisEngineError("At least one group_by column is required.")

    _validate_columns(
        dataframe,
        group_by + list(metrics.keys()),
    )

    for aggregation in metrics.values():
        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise AnalysisEngineError(
                f"Unsupported aggregation: {aggregation}"
            )

    result = (
        dataframe
        .groupby(group_by, dropna=False)
        .agg(metrics)
        .reset_index()
    )

    # Flatten MultiIndex columns if pandas creates one.
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            "_".join(str(part) for part in column if part)
            for column in result.columns
        ]

    return result


def calculate_ratio(
    dataframe: pd.DataFrame,
    numerator: str,
    denominator: str,
    output_name: str,
) -> pd.DataFrame:
    """
    Calculate numerator / denominator safely.
    """

    _validate_columns(dataframe, [numerator, denominator])

    result = dataframe.copy()

    denominator_values = pd.to_numeric(
        result[denominator],
        errors="coerce",
    )

    numerator_values = pd.to_numeric(
        result[numerator],
        errors="coerce",
    )

    result[output_name] = numerator_values.div(
        denominator_values.replace(0, pd.NA)
    )

    return result


def calculate_difference(
    dataframe: pd.DataFrame,
    left: str,
    right: str,
    output_name: str,
) -> pd.DataFrame:
    """
    Calculate left - right deterministically.
    """

    _validate_columns(dataframe, [left, right])

    result = dataframe.copy()

    result[output_name] = (
        pd.to_numeric(result[left], errors="coerce")
        - pd.to_numeric(result[right], errors="coerce")
    )

    return result


def calculate_percentage_of_total(
    dataframe: pd.DataFrame,
    column: str,
    output_name: str,
) -> pd.DataFrame:
    """
    Calculate each row's percentage contribution to the column total.
    """

    _validate_columns(dataframe, [column])

    result = dataframe.copy()

    values = pd.to_numeric(
        result[column],
        errors="coerce",
    )

    total = values.sum()

    if total == 0:
        result[output_name] = 0.0
    else:
        result[output_name] = values / total * 100.0

    return result


def rank_dataframe(
    dataframe: pd.DataFrame,
    column: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank rows using a deterministic metric.
    """

    _validate_columns(dataframe, [column])

    result = dataframe.copy()

    result["_rank"] = (
        pd.to_numeric(result[column], errors="coerce")
        .rank(
            method="min",
            ascending=ascending,
        )
    )

    result = result.sort_values(
        by="_rank",
        ascending=True,
    )

    return result


def top_n(
    dataframe: pd.DataFrame,
    column: str,
    n: int,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return the top or bottom N rows according to a metric.
    """

    if n <= 0:
        raise AnalysisEngineError("n must be greater than zero.")

    _validate_columns(dataframe, [column])

    return (
        dataframe
        .sort_values(
            by=column,
            ascending=ascending,
        )
        .head(n)
        .reset_index(drop=True)
    )