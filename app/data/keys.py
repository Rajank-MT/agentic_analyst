import pandas as pd

from app.data.model import KeyProfile


def analyze_key_candidate(
    column_name: str,
    series: pd.Series,
) -> KeyProfile:
    """
    Analyze whether a column is a candidate primary key.

    A candidate primary key must:
        - contain no null values
        - contain unique values
        - contain at least one value
    """

    row_count = len(series)

    if row_count == 0:
        return KeyProfile(
            column_name=column_name,
            is_unique=False,
            uniqueness_ratio=0.0,
            is_candidate_primary_key=False,
        )

    non_null_count = int(series.notna().sum())

    unique_count = int(
        series.nunique(dropna=True)
    )

    uniqueness_ratio = (
        unique_count / row_count
    )

    is_unique = (
        unique_count == row_count
        and non_null_count == row_count
    )

    is_candidate_primary_key = is_unique

    return KeyProfile(
        column_name=column_name,
        is_unique=is_unique,
        uniqueness_ratio=round(
            uniqueness_ratio,
            6,
        ),
        is_candidate_primary_key=is_candidate_primary_key,
    )