import pandas as pd

from app.data.model import (
    CardinalityProfile,
    CardinalityType,
)


def infer_cardinality(
    left_series: pd.Series,
    right_series: pd.Series,
) -> CardinalityProfile:
    """
    Infer cardinality from uniqueness of the two columns.

    The function compares the uniqueness of each column
    independently.

    Null values are ignored.
    """

    left_values = left_series.dropna()
    right_values = right_series.dropna()

    left_unique_count = int(
        left_values.nunique()
    )

    right_unique_count = int(
        right_values.nunique()
    )

    left_row_count = len(left_values)
    right_row_count = len(right_values)

    left_unique = (
        left_row_count > 0
        and left_unique_count == left_row_count
    )

    right_unique = (
        right_row_count > 0
        and right_unique_count == right_row_count
    )

    if left_unique and right_unique:
        cardinality = CardinalityType.ONE_TO_ONE

    elif left_unique and not right_unique:
        cardinality = CardinalityType.ONE_TO_MANY

    elif not left_unique and right_unique:
        cardinality = CardinalityType.MANY_TO_ONE

    elif left_row_count > 0 and right_row_count > 0:
        cardinality = CardinalityType.MANY_TO_MANY

    else:
        cardinality = CardinalityType.UNKNOWN

    return CardinalityProfile(
        cardinality=cardinality,
        left_unique=left_unique,
        right_unique=right_unique,
        left_unique_count=left_unique_count,
        right_unique_count=right_unique_count,
    )