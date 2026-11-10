from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from app.data.model import (
    CardinalityType,
    ColumnProfile,
    RelationshipCandidate,
)


def _normalize_column_name(column_name: str) -> str:
    """Normalize a column name before comparing it with another name."""
    return (
        column_name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _name_similarity(left_column: str, right_column: str) -> float:
    """Return a similarity score between two normalized column names."""
    left = _normalize_column_name(left_column)
    right = _normalize_column_name(right_column)
    return SequenceMatcher(None, left, right).ratio()


def _semantic_compatibility(
    left: ColumnProfile,
    right: ColumnProfile,
) -> float:
    """Score whether two columns have compatible semantic types."""
    if left.semantic_type == right.semantic_type:
        return 1.0

    compatible_pairs = {
        ("identifier", "numeric"),
        ("numeric", "identifier"),
    }
    pair = (
        left.semantic_type.value,
        right.semantic_type.value,
    )
    return 0.8 if pair in compatible_pairs else 0.0


def _dtype_compatibility(
    left: ColumnProfile,
    right: ColumnProfile,
) -> float:
    """Score compatibility of the pandas dtype names stored in profiles."""
    left_dtype = str(left.dtype).lower()
    right_dtype = str(right.dtype).lower()

    if left_dtype == right_dtype:
        return 1.0

    numeric_types = {
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
    }
    if left_dtype in numeric_types and right_dtype in numeric_types:
        return 1.0

    return 0.0


def generate_relationship_candidate(
    left_dataset: str,
    left_column: ColumnProfile,
    right_dataset: str,
    right_column: ColumnProfile,
) -> RelationshipCandidate:
    """
    Generate a relationship candidate from two column profiles.

    The score ranks a possible relationship; it does not confirm that the
    columns are related. Use validate_relationship_candidate with the
    underlying Series for data-level validation.
    """
    name_score = _name_similarity(left_column.name, right_column.name)
    semantic_score = _semantic_compatibility(left_column, right_column)
    dtype_score = _dtype_compatibility(left_column, right_column)

    candidate_score = (
        name_score * 0.4
        + semantic_score * 0.4
        + dtype_score * 0.2
    )

    return RelationshipCandidate(
        left_dataset=left_dataset,
        left_column=left_column.name,
        right_dataset=right_dataset,
        right_column=right_column.name,
        name_similarity=round(name_score, 6),
        semantic_compatibility=round(semantic_score, 6),
        dtype_compatibility=round(dtype_score, 6),
        candidate_score=round(candidate_score, 6),
    )


def _clean_values(series: pd.Series) -> pd.Series:
    """Drop nulls and normalize scalar values for overlap comparison."""
    return series.dropna().map(lambda value: str(value).strip())


def calculate_value_overlap(
    left_series: pd.Series,
    right_series: pd.Series,
) -> dict[str, Any]:
    """
    Calculate distinct-value overlap between two columns.

    left_overlap_ratio is the fraction of distinct left values also present
    on the right. right_overlap_ratio is the reverse. Keeping both ratios is
    important because a parent table can contain unused parent records.
    """
    left_values = set(_clean_values(left_series))
    right_values = set(_clean_values(right_series))
    common_values = left_values.intersection(right_values)

    left_ratio = len(common_values) / len(left_values) if left_values else 0.0
    right_ratio = len(common_values) / len(right_values) if right_values else 0.0

    return {
        "left_unique_count": len(left_values),
        "right_unique_count": len(right_values),
        "common_count": len(common_values),
        "left_overlap_ratio": round(left_ratio, 6),
        "right_overlap_ratio": round(right_ratio, 6),
        "common_values": common_values,
    }


def infer_cardinality(
    left_series: pd.Series,
    right_series: pd.Series,
) -> CardinalityType:
    """
    Estimate relationship cardinality from column uniqueness.

    This is a sample-based estimate, not proof of a database PK/FK constraint.
    """
    left = _clean_values(left_series)
    right = _clean_values(right_series)

    if left.empty or right.empty:
        return CardinalityType.UNKNOWN

    left_unique = left.nunique(dropna=True) == len(left)
    right_unique = right.nunique(dropna=True) == len(right)

    if left_unique and right_unique:
        return CardinalityType.ONE_TO_ONE
    if left_unique and not right_unique:
        return CardinalityType.ONE_TO_MANY
    if not left_unique and right_unique:
        return CardinalityType.MANY_TO_ONE
    return CardinalityType.MANY_TO_MANY


def validate_relationship_candidate(
    candidate: RelationshipCandidate,
    left_series: pd.Series,
    right_series: pd.Series,
    *,
    minimum_overlap: float = 0.80,
    minimum_candidate_score: float = 0.60,
) -> dict[str, Any]:
    """
    Validate a candidate using overlap and estimated cardinality.

    Returns a dictionary with status CONFIRMED, POSSIBLE, or REJECTED.
    CONFIRMED is a heuristic result, not a declared database constraint.
    """
    if not 0.0 <= minimum_overlap <= 1.0:
        raise ValueError("minimum_overlap must be between 0 and 1.")
    if not 0.0 <= minimum_candidate_score <= 1.0:
        raise ValueError("minimum_candidate_score must be between 0 and 1.")

    overlap = calculate_value_overlap(left_series, right_series)
    cardinality = infer_cardinality(left_series, right_series)

    left_ratio = overlap["left_overlap_ratio"]
    right_ratio = overlap["right_overlap_ratio"]
    candidate_score = float(candidate.candidate_score)

    # For left-unique/right-repeated (one-to-many), right values should map
    # into the left parent values. The inverse applies to many-to-one.
    if cardinality == CardinalityType.ONE_TO_MANY:
        directional_coverage = right_ratio
    elif cardinality == CardinalityType.MANY_TO_ONE:
        directional_coverage = left_ratio
    elif cardinality == CardinalityType.ONE_TO_ONE:
        directional_coverage = min(left_ratio, right_ratio)
    else:
        directional_coverage = max(left_ratio, right_ratio)

    if overlap["common_count"] == 0 or candidate_score < minimum_candidate_score:
        status = "REJECTED"
    elif (
        cardinality in {
            CardinalityType.ONE_TO_ONE,
            CardinalityType.ONE_TO_MANY,
            CardinalityType.MANY_TO_ONE,
        }
        and directional_coverage >= minimum_overlap
    ):
        status = "CONFIRMED"
    else:
        status = "POSSIBLE"

    serializable_overlap = {
        key: value
        for key, value in overlap.items()
        if key != "common_values"
    }
    confidence_score = (
        candidate_score * 0.5
        + directional_coverage * 0.5
    )

    return {
        "left_dataset": candidate.left_dataset,
        "left_column": candidate.left_column,
        "right_dataset": candidate.right_dataset,
        "right_column": candidate.right_column,
        "status": status,
        "confidence_score": round(confidence_score, 6),
        "candidate_score": round(candidate_score, 6),
        "directional_coverage": round(directional_coverage, 6),
        "overlap": serializable_overlap,
        "cardinality": cardinality.value,
    }
