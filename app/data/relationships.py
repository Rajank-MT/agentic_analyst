from difflib import SequenceMatcher

from app.data.model import (
    ColumnProfile,
    RelationshipCandidate,
)


def _normalize_column_name(
    column_name: str,
) -> str:
    """Normalize a column name for comparison."""

    return (
        column_name
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _name_similarity(
    left_column: str,
    right_column: str,
) -> float:
    """Calculate normalized column-name similarity."""

    left = _normalize_column_name(left_column)
    right = _normalize_column_name(right_column)

    return SequenceMatcher(
        None,
        left,
        right,
    ).ratio()


def _semantic_compatibility(
    left: ColumnProfile,
    right: ColumnProfile,
) -> float:
    """Calculate semantic-type compatibility."""

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

    if pair in compatible_pairs:
        return 0.8

    return 0.0


def _dtype_compatibility(
    left: ColumnProfile,
    right: ColumnProfile,
) -> float:
    """Calculate basic pandas dtype compatibility."""

    if left.dtype == right.dtype:
        return 1.0

    numeric_types = {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    }

    if (
        left.dtype in numeric_types
        and right.dtype in numeric_types
    ):
        return 1.0

    return 0.0


def generate_relationship_candidate(
    left_dataset: str,
    left_column: ColumnProfile,
    right_dataset: str,
    right_column: ColumnProfile,
) -> RelationshipCandidate:
    """Generate a relationship candidate from two columns."""

    name_score = _name_similarity(
        left_column.name,
        right_column.name,
    )

    semantic_score = _semantic_compatibility(
        left_column,
        right_column,
    )

    dtype_score = _dtype_compatibility(
        left_column,
        right_column,
    )

    candidate_score = (
        (name_score * 0.4)
        + (semantic_score * 0.4)
        + (dtype_score * 0.2)
    )

    return RelationshipCandidate(
        left_dataset=left_dataset,
        left_column=left_column.name,
        right_dataset=right_dataset,
        right_column=right_column.name,
        name_similarity=round(
            name_score,
            6,
        ),
        semantic_compatibility=round(
            semantic_score,
            6,
        ),
        dtype_compatibility=round(
            dtype_score,
            6,
        ),
        candidate_score=round(
            candidate_score,
            6,
        ),
    )