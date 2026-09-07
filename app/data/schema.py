import re

import pandas as pd

from app.data.model import ColumnSemanticType


_IDENTIFIER_NAME_PATTERNS = (
    r"(^id$)",
    r"(_id$)",
    r"(^id_)",
    r"(^.*identifier$)",
    r"(^.*_key$)",
)

_DATETIME_NAME_HINTS = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "created_at",
    "updated_at",
    "modified_at",
    "deleted_at",
    "dob",
    "birth_date",
)

_CATEGORICAL_NAME_HINTS = (
    "category",
    "type",
    "status",
    "city",
    "country",
    "state",
    "region",
    "gender",
    "department",
)

_TEXT_NAME_HINTS = (
    "description",
    "comment",
    "comments",
    "remark",
    "remarks",
    "message",
    "review",
    "notes",
    "note",
    "address",
)


def _looks_like_identifier(
    column_name: str,
    series: pd.Series,
) -> bool:
    """Determine whether a column is likely an identifier."""

    normalized_name = column_name.strip().lower()

    return any(
        re.search(pattern, normalized_name)
        for pattern in _IDENTIFIER_NAME_PATTERNS
    )


def _looks_like_datetime(
    column_name: str,
    series: pd.Series,
) -> bool:
    """
    Determine whether a column contains datetime-like values.

    The function uses column-name hints and actual value parsing.
    """

    normalized_name = column_name.strip().lower()

    non_null = series.dropna()

    if non_null.empty:
        return False

    # Already recognized by pandas as datetime.
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Numeric columns should not be interpreted as dates here.
    if pd.api.types.is_numeric_dtype(series):
        return False

    # First use explicit column-name hints.
    has_datetime_hint = any(
        hint in normalized_name
        for hint in _DATETIME_NAME_HINTS
    )

    # Convert values to strings for safe parsing.
    values = non_null.astype(str)

    parsed = pd.to_datetime(
        values,
        errors="coerce",
        format="mixed",
    )

    parse_ratio = parsed.notna().mean()

    # Strong evidence from the column name.
    if has_datetime_hint and parse_ratio >= 0.80:
        return True

    # For columns without a date-related name, require very strong
    # evidence before classifying them as datetime.
    if parse_ratio >= 0.95:
        return True

    return False


def _looks_like_categorical(
    column_name: str,
    series: pd.Series,
) -> bool:
    """Determine whether a string column is likely categorical."""

    normalized_name = column_name.strip().lower()

    if any(
        hint in normalized_name
        for hint in _CATEGORICAL_NAME_HINTS
    ):
        return True

    if any(
        hint in normalized_name
        for hint in _TEXT_NAME_HINTS
    ):
        return False

    non_null = series.dropna()

    if non_null.empty:
        return False

    unique_count = non_null.nunique()
    value_count = len(non_null)
    unique_ratio = unique_count / value_count

    if unique_count <= 10 and unique_ratio <= 0.5:
        return True

    if unique_ratio <= 0.05:
        return True

    return False


def classify_column(
    column_name: str,
    series: pd.Series,
) -> ColumnSemanticType:
    """
    Classify a column into a high-level semantic type.

    Classification priority:

        Identifier
        Boolean
        Datetime
        Numeric
        Categorical
        Text
        Unknown
    """

    if _looks_like_identifier(
        column_name,
        series,
    ):
        return ColumnSemanticType.IDENTIFIER

    if pd.api.types.is_bool_dtype(series):
        return ColumnSemanticType.BOOLEAN

    if _looks_like_datetime(
        column_name,
        series,
    ):
        return ColumnSemanticType.DATETIME

    if pd.api.types.is_numeric_dtype(series):
        return ColumnSemanticType.NUMERIC

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        non_null = series.dropna()

        if non_null.empty:
            return ColumnSemanticType.UNKNOWN

        if _looks_like_categorical(
            column_name,
            series,
        ):
            return ColumnSemanticType.CATEGORICAL

        return ColumnSemanticType.TEXT

    return ColumnSemanticType.UNKNOWN