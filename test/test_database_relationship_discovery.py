
import pandas as pd
import pytest

from app.data.database_relationship_discovery import (
    discover_relationships_across_tables,
    summarize_relationships,
)

from app.data.model import (
    CardinalityType,
    RelationshipStatus,
)


def test_discovers_one_to_many_relationship():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "customer_name": ["Amit", "Rahul", "Priya", "Neha"],
    })

    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "customer_id": [1, 1, 2, 3, 3],
    })

    relationships = discover_relationships_across_tables({
        "customers": customers,
        "orders": orders,
    })

    matching = [
        r for r in relationships
        if r.left_column == "customer_id"
        and r.right_column == "customer_id"
    ]

    assert len(matching) == 1

    relationship = matching[0]

    assert relationship.left_dataset == "customers"
    assert relationship.right_dataset == "orders"
    assert relationship.status == RelationshipStatus.CONFIRMED
    assert (
        relationship.cardinality.cardinality
        == CardinalityType.ONE_TO_MANY
    )


def test_discovers_relationship_with_null_values():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
    })

    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "customer_id": [1, 2, None, 3],
    })

    relationships = discover_relationships_across_tables({
        "customers": customers,
        "orders": orders,
    })

    assert any(
        r.left_column == "customer_id"
        and r.right_column == "customer_id"
        for r in relationships
    )


def test_does_not_discover_relationship_for_unrelated_columns():
    employees = pd.DataFrame({
        "employee_name": ["Amit", "Rahul", "Priya"],
        "salary": [30000, 40000, 50000],
    })

    products = pd.DataFrame({
        "product_name": ["Laptop", "Phone", "Tablet"],
        "price": [70000, 20000, 30000],
    })

    relationships = discover_relationships_across_tables({
        "employees": employees,
        "products": products,
    })

    # Candidates may be retained for diagnostic purposes,
    # but unrelated columns must not be confirmed or possible.
    assert all(
        r.status == RelationshipStatus.REJECTED
        for r in relationships
    )

    assert not any(
        r.status in (
            RelationshipStatus.CONFIRMED,
            RelationshipStatus.POSSIBLE,
        )
        for r in relationships
    )


def test_returns_empty_list_for_single_table():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
    })

    relationships = discover_relationships_across_tables({
        "customers": customers,
    })

    assert relationships == []


def test_returns_empty_list_for_no_tables():
    relationships = discover_relationships_across_tables({})

    assert relationships == []


def test_handles_empty_dataframes():
    customers = pd.DataFrame({
        "customer_id": pd.Series(dtype="int64"),
    })

    orders = pd.DataFrame({
        "customer_id": pd.Series(dtype="int64"),
    })

    relationships = discover_relationships_across_tables({
        "customers": customers,
        "orders": orders,
    })

    assert isinstance(relationships, list)


def test_relationship_summary_counts_statuses():
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
    })

    orders = pd.DataFrame({
        "customer_id": [1, 1, 2],
    })

    relationships = discover_relationships_across_tables({
        "customers": customers,
        "orders": orders,
    })

    summary = summarize_relationships(relationships)

    assert summary["total"] == len(relationships)

    assert summary["confirmed"] == sum(
        r.status == RelationshipStatus.CONFIRMED
        for r in relationships
    )

    assert summary["possible"] == sum(
        r.status == RelationshipStatus.POSSIBLE
        for r in relationships
    )

    assert summary["rejected"] == sum(
        r.status == RelationshipStatus.REJECTED
        for r in relationships
    )


def test_rejects_invalid_candidate_score():
    with pytest.raises(ValueError):
        discover_relationships_across_tables(
            {},
            minimum_candidate_score=1.5,
        )


def test_rejects_invalid_overlap_threshold():
    with pytest.raises(ValueError):
        discover_relationships_across_tables(
            {},
            minimum_overlap=-0.1,
        )


def test_rejects_invalid_candidate_pair_limit():
    with pytest.raises(ValueError):
        discover_relationships_across_tables(
            {},
            max_candidate_pairs=0,
        )
