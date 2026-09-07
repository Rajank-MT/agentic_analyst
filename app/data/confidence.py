from app.data.model import (
    CardinalityProfile,
    CardinalityType,
    KeyProfile,
    RelationshipCandidate,
    RelationshipConfidence,
    ValueOverlapProfile,
)


def _cardinality_score(
    cardinality: CardinalityType,
) -> float:
    """Convert cardinality evidence into a score."""

    if cardinality in {
        CardinalityType.ONE_TO_ONE,
        CardinalityType.ONE_TO_MANY,
        CardinalityType.MANY_TO_ONE,
    }:
        return 1.0

    if cardinality == CardinalityType.MANY_TO_MANY:
        return 0.5

    return 0.0


def _key_score(
    left_key: KeyProfile | None,
    right_key: KeyProfile | None,
) -> float:
    """Evaluate key evidence for a relationship."""

    if left_key is None and right_key is None:
        return 0.0

    left_is_key = (
        left_key is not None
        and left_key.is_candidate_primary_key
    )

    right_is_key = (
        right_key is not None
        and right_key.is_candidate_primary_key
    )

    if left_is_key or right_is_key:
        return 1.0

    return 0.0


def calculate_relationship_confidence(
    candidate: RelationshipCandidate,
    overlap: ValueOverlapProfile,
    cardinality: CardinalityProfile,
    left_key: KeyProfile | None = None,
    right_key: KeyProfile | None = None,
) -> RelationshipConfidence:
    """
    Calculate an explainable relationship confidence score.

    Evidence weights:

    Name similarity       15%
    Semantic compatibility 20%
    dtype compatibility   10%
    Value overlap         30%
    Cardinality            10%
    Key evidence           15%
    """

    overlap_score = max(
        overlap.left_overlap_ratio,
        overlap.right_overlap_ratio,
    )

    cardinality_score = _cardinality_score(
        cardinality.cardinality
    )

    key_score = _key_score(
        left_key,
        right_key,
    )

    score = (
        candidate.name_similarity * 0.15
        + candidate.semantic_compatibility * 0.20
        + candidate.dtype_compatibility * 0.10
        + overlap_score * 0.30
        + cardinality_score * 0.10
        + key_score * 0.15
    )

    return RelationshipConfidence(
        score=round(score, 6),
        name_score=candidate.name_similarity,
        semantic_score=candidate.semantic_compatibility,
        dtype_score=candidate.dtype_compatibility,
        overlap_score=round(
            overlap_score,
            6,
        ),
        cardinality_score=cardinality_score,
        key_score=key_score,
    )