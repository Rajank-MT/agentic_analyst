from __future__ import annotations

import pandas as pd

from app.data.model import (
    CardinalityProfile,
    CardinalityType,
    DiscoveredRelationship,
    RelationshipConfidence,
    RelationshipStatus,
    ValueOverlapProfile,
)
from app.data.profiling import profile_dataset
from app.data.relationships import (
    generate_relationship_candidate,
    validate_relationship_candidate,
)


def discover_relationships_across_tables(
    tables: dict[str, pd.DataFrame],
    *,
    minimum_candidate_score: float = 0.60,
    minimum_overlap: float = 0.80,
    max_candidate_pairs: int = 5000,
) -> list[DiscoveredRelationship]:
    """Discover and validate likely column relationships across table samples."""
    if not 0 <= minimum_candidate_score <= 1:
        raise ValueError("minimum_candidate_score must be between 0 and 1.")
    if not 0 <= minimum_overlap <= 1:
        raise ValueError("minimum_overlap must be between 0 and 1.")
    if max_candidate_pairs < 1:
        raise ValueError("max_candidate_pairs must be at least 1.")

    table_names = sorted(tables)
    if len(table_names) < 2:
        return []

    profiles = {
        name: profile_dataset(tables[name], name)
        for name in table_names
    }
    results: list[DiscoveredRelationship] = []
    checked = 0

    for i, left_name in enumerate(table_names):
        left_df = tables[left_name]
        for right_name in table_names[i + 1:]:
            right_df = tables[right_name]
            for left_col in profiles[left_name].columns:
                if left_col.name not in left_df.columns:
                    continue
                for right_col in profiles[right_name].columns:
                    if right_col.name not in right_df.columns:
                        continue

                    candidate = generate_relationship_candidate(
                        left_name, left_col, right_name, right_col
                    )
                    if candidate.candidate_score < minimum_candidate_score:
                        continue
                    checked += 1
                    if checked > max_candidate_pairs:
                        return results

                    validation = validate_relationship_candidate(
                        candidate,
                        left_df[left_col.name],
                        right_df[right_col.name],
                        minimum_overlap=minimum_overlap,
                        minimum_candidate_score=minimum_candidate_score,
                    )
                    ov = validation["overlap"]
                    cardinality = CardinalityType(validation["cardinality"])
                    left_values = left_df[left_col.name].dropna().map(lambda x: str(x).strip())
                    right_values = right_df[right_col.name].dropna().map(lambda x: str(x).strip())
                    left_unique = bool(not left_values.empty and left_values.nunique() == len(left_values))
                    right_unique = bool(not right_values.empty and right_values.nunique() == len(right_values))

                    left_key = left_col.key_profile
                    right_key = right_col.key_profile
                    key_score = max(
                        left_key.uniqueness_ratio if left_key else 0.0,
                        right_key.uniqueness_ratio if right_key else 0.0,
                    )

                    confidence = RelationshipConfidence(
                        score=validation["confidence_score"],
                        name_score=candidate.name_similarity,
                        semantic_score=candidate.semantic_compatibility,
                        dtype_score=candidate.dtype_compatibility,
                        overlap_score=validation["directional_coverage"],
                        cardinality_score=0.0 if cardinality == CardinalityType.UNKNOWN else 1.0,
                        key_score=key_score,
                    )
                    overlap = ValueOverlapProfile(
                        common_value_count=ov["common_count"],
                        left_unique_count=ov["left_unique_count"],
                        right_unique_count=ov["right_unique_count"],
                        left_overlap_ratio=ov["left_overlap_ratio"],
                        right_overlap_ratio=ov["right_overlap_ratio"],
                    )
                    cardinality_profile = CardinalityProfile(
                        cardinality=cardinality,
                        left_unique=left_unique,
                        right_unique=right_unique,
                        left_unique_count=ov["left_unique_count"],
                        right_unique_count=ov["right_unique_count"],
                    )
                    results.append(DiscoveredRelationship(
                        left_dataset=left_name,
                        left_column=left_col.name,
                        right_dataset=right_name,
                        right_column=right_col.name,
                        status=RelationshipStatus(validation["status"].lower()),
                        confidence=confidence,
                        overlap=overlap,
                        cardinality=cardinality_profile,
                    ))
    return results


def summarize_relationships(
    relationships: list[DiscoveredRelationship],
) -> dict[str, int]:
    """Count discovered relationships by status."""
    return {
        "total": len(relationships),
        "confirmed": sum(r.status == RelationshipStatus.CONFIRMED for r in relationships),
        "possible": sum(r.status == RelationshipStatus.POSSIBLE for r in relationships),
        "rejected": sum(r.status == RelationshipStatus.REJECTED for r in relationships),
    }
