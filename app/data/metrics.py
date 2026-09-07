import re

from app.data.model import (
    ColumnProfile,
    ColumnSemanticType,
    MetricProfile,
    MetricRole,
)


TIME_KEYWORDS = {
    "date",
    "time",
    "timestamp",
    "year",
    "month",
    "day",
}

ENTITY_KEYWORDS = {
    "id",
    "code",
    "identifier",
    "key",
}

MEASURE_KEYWORDS = {
    "amount",
    "revenue",
    "sales",
    "price",
    "cost",
    "profit",
    "income",
    "salary",
    "quantity",
    "count",
    "value",
    "score",
    "rate",
    "total",
}


def _contains_keyword(column_name: str, keywords: set[str]) -> bool:
    normalized_name = column_name.lower()

    tokens = set(
        re.split(
            r"[^a-zA-Z0-9]+",
            normalized_name,
        )
    )

    return bool(tokens.intersection(keywords))


def discover_metric_role(
    column: ColumnProfile,
) -> MetricProfile:

    signals: list[str] = []

    # Time dimension
    if column.semantic_type == ColumnSemanticType.DATETIME:
        signals.append("datetime_column")

        return MetricProfile(
            column_name=column.name,
            role=MetricRole.TIME,
            confidence=1.0,
            signals=signals,
        )

    # Entity / identifier
    if column.semantic_type == ColumnSemanticType.IDENTIFIER:
        signals.append("identifier_column")

        return MetricProfile(
            column_name=column.name,
            role=MetricRole.ENTITY,
            confidence=1.0,
            signals=signals,
        )

    # Numeric measures
    if column.semantic_type == ColumnSemanticType.NUMERIC:

        confidence = 0.60
        signals.append("numeric_column")

        if _contains_keyword(column.name, MEASURE_KEYWORDS):
            confidence += 0.30
            signals.append("business_measure_name")

        if column.unique_percentage < 95:
            confidence += 0.05
            signals.append("repeated_values")

        confidence = min(confidence, 1.0)

        return MetricProfile(
            column_name=column.name,
            role=MetricRole.MEASURE,
            confidence=round(confidence, 6),
            signals=signals,
        )

    # Categorical dimensions
    if column.semantic_type == ColumnSemanticType.CATEGORICAL:
        signals.append("categorical_column")

        return MetricProfile(
            column_name=column.name,
            role=MetricRole.DIMENSION,
            confidence=0.90,
            signals=signals,
        )

    return MetricProfile(
        column_name=column.name,
        role=MetricRole.UNKNOWN,
        confidence=0.0,
        signals=[],
    )