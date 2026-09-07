from app.data.cardinality import infer_cardinality
from app.data.confidence import (
    calculate_relationship_confidence,
)
from app.data.model import (
    ColumnProfile,
    DiscoveredRelationship,
    RelationshipCandidate,
    RelationshipStatus,
)
from app.data.overlap import calculate_value_overlap
from app.data.relationships import (
    generate_relationship_candidate,
)


CONFIRMED_THRESHOLD = 0.80
POSSIBLE_THRESHOLD = 0.60


def _determine_status(
    confidence_score: float,
) -> RelationshipStatus:
    """Convert confidence score into a relationship status."""

    if confidence_score >= CONFIRMED_THRESHOLD:
        return RelationshipStatus.CONFIRMED

    if confidence_score >= POSSIBLE_THRESHOLD:
        return RelationshipStatus.POSSIBLE

    return RelationshipStatus.REJECTED


def discover_relationship(
    left_dataset: str,
    left_column: ColumnProfile,
    left_series,
    right_dataset: str,
    right_column: ColumnProfile,
    right_series,
) -> DiscoveredRelationship:
    """
    Discover and score a relationship between two columns.

    This function combines:

    1. Structural candidate matching
    2. Value overlap
    3. Cardinality
    4. Confidence scoring
    """

    candidate = generate_relationship_candidate(
        left_dataset=left_dataset,
        left_column=left_column,
        right_dataset=right_dataset,
        right_column=right_column,
    )

    overlap = calculate_value_overlap(
        left_series,
        right_series,
    )

    cardinality = infer_cardinality(
        left_series,
        right_series,
    )

    confidence = calculate_relationship_confidence(
        candidate=candidate,
        overlap=overlap,
        cardinality=cardinality,
        left_key=left_column.key_profile,
        right_key=right_column.key_profile,
    )

    status = _determine_status(
        confidence.score
    )

    return DiscoveredRelationship(
        left_dataset=left_dataset,
        left_column=left_column.name,
        right_dataset=right_dataset,
        right_column=right_column.name,
        status=status,
        confidence=confidence,
        overlap=overlap,
        cardinality=cardinality,
    )