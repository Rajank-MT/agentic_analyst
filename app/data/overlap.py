
from typing import Any

import pandas as pd

from app.data.model import ValueOverlapProfile


def _normalize_value(value: Any) -> Any:
    """Normalize a value for reliable comparison."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def calculate_value_overlap(
    left_series: pd.Series,
    right_series: pd.Series,
) -> ValueOverlapProfile:
    """
    Calculate unique-value overlap between two columns.

    Null values are ignored because they do not represent
    meaningful relationships between records.
    """

    left_values = {
        _normalize_value(value)
        for value in left_series.dropna().tolist()
    }

    right_values = {
        _normalize_value(value)
        for value in right_series.dropna().tolist()
    }

    common_values = left_values.intersection(
        right_values
    )

    common_value_count = len(common_values)

    left_unique_count = len(left_values)
    right_unique_count = len(right_values)

    if left_unique_count > 0:
        left_overlap_ratio = (
            common_value_count / left_unique_count
        )
    else:
        left_overlap_ratio = 0.0

    if right_unique_count > 0:
        right_overlap_ratio = (
            common_value_count / right_unique_count
        )
    else:
        right_overlap_ratio = 0.0

    return ValueOverlapProfile(
        common_value_count=common_value_count,
        left_unique_count=left_unique_count,
        right_unique_count=right_unique_count,
        left_overlap_ratio=round(
            left_overlap_ratio,
            6,
        ),
        right_overlap_ratio=round(
            right_overlap_ratio,
            6,
        ),
    )

