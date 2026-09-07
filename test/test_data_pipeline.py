
from pathlib import Path

import pandas as pd
import pytest

from app.data.ingestion import (
    DataIngestionError,
    load_dataset,
)
from app.data.model import ColumnSemanticType
from app.data.profiling import profile_dataset
from app.data.schema import classify_column
from app.data.keys import analyze_key_candidate

from app.data.relationships import (
    generate_relationship_candidate,
)

from app.data.overlap import calculate_value_overlap
from app.data.cardinality import infer_cardinality
from app.data.model import CardinalityType

from app.data.confidence import (
    calculate_relationship_confidence,
)

from app.data.model import (
    CardinalityProfile,
    CardinalityType,
    ColumnSemanticType,
    KeyProfile,
    RelationshipCandidate,
    ValueOverlapProfile,
)

from app.data.discovery import (
    POSSIBLE_THRESHOLD,
    discover_relationship,
)
from app.data.model import (
    DiscoveredRelationship,
    RelationshipStatus,
)

from app.data.statistics import (
    calculate_numeric_statistics,
)

from app.data.correlation import calculate_correlation


from app.data.categorical import calculate_categorical_analysis

from app.data.anomaly import detect_iqr_anomalies


from app.data.metrics import discover_metric_role
from app.data.model import (
    ColumnProfile,
    ColumnSemanticType,
    MetricRole,
)


from app.data.planner import build_analysis_plan

from app.data.model import (
    AnalysisType,
    MetricProfile,
    MetricRole,
)

from app.data.model import (
    BusinessInsight,
    BusinessInsightReport,
    InsightEvidence,
    InsightPriority,
)

def test_load_csv():
    dataframe = load_dataset(
        Path("datasets/customers.csv")
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 10
    assert len(dataframe.columns) == 6


def test_missing_file():
    with pytest.raises(DataIngestionError):
        load_dataset(
            "datasets/does_not_exist.csv"
        )


def test_unsupported_file():
    path = Path("datasets/test.txt")
    path.write_text(
        "hello",
        encoding="utf-8",
    )

    try:
        with pytest.raises(DataIngestionError):
            load_dataset(path)
    finally:
        path.unlink(missing_ok=True)


def test_dataset_profile():
    dataframe = load_dataset(
        Path("datasets/customers.csv")
    )

    profile = profile_dataset(
        dataframe,
        "customers.csv",
    )

    assert profile.dataset_name == "customers.csv"
    assert profile.row_count == 10
    assert profile.column_count == 6
    assert profile.duplicate_row_count == 0

    column_names = {
        column.name
        for column in profile.columns
    }

    assert "customer_id" in column_names
    assert "income" in column_names
    assert "city" in column_names


def test_numeric_profile():
    dataframe = pd.DataFrame(
        {
            "value": [10, 20, 30, 40],
        }
    )

    profile = profile_dataset(
        dataframe,
        "test.csv",
    )

    column = profile.columns[0]

    assert column.mean == 25.0
    assert column.median == 25.0
    assert column.min_value == 10.0
    assert column.max_value == 40.0


def test_identifier_classification():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
        }
    )

    result = classify_column(
        "customer_id",
        dataframe["customer_id"],
    )

    assert result == ColumnSemanticType.IDENTIFIER


def test_numeric_classification():
    dataframe = pd.DataFrame(
        {
            "income": [50000, 60000, 70000, 80000],
        }
    )

    result = classify_column(
        "income",
        dataframe["income"],
    )

    assert result == ColumnSemanticType.NUMERIC


def test_categorical_classification():
    dataframe = pd.DataFrame(
        {
            "city": [
                "Hyderabad",
                "Hyderabad",
                "Delhi",
                "Hyderabad",
            ],
        }
    )

    result = classify_column(
        "city",
        dataframe["city"],
    )

    assert result == ColumnSemanticType.CATEGORICAL


def test_boolean_classification():
    dataframe = pd.DataFrame(
        {
            "is_active": [
                True,
                False,
                True,
                True,
            ],
        }
    )

    result = classify_column(
        "is_active",
        dataframe["is_active"],
    )

    assert result == ColumnSemanticType.BOOLEAN


def test_text_classification():
    dataframe = pd.DataFrame(
        {
            "description": [
                "First customer",
                "Second customer",
                "Third customer",
                "Fourth customer",
            ],
        }
    )

    result = classify_column(
        "description",
        dataframe["description"],
    )

    assert result == ColumnSemanticType.TEXT

def test_datetime_classification():
    dataframe = pd.DataFrame(
        {
            "signup_date": [
                "2025-01-15",
                "2025-02-10",
                "2025-03-05",
                "2025-04-11",
            ],
        }
    )

    result = classify_column(
        "signup_date",
        dataframe["signup_date"],
    )

    assert result == ColumnSemanticType.DATETIME


def test_datetime_timestamp_classification():
    dataframe = pd.DataFrame(
        {
            "created_at": [
                "2025-01-15 10:30:00",
                "2025-02-10 12:45:00",
                "2025-03-05 09:15:00",
                "2025-04-11 18:20:00",
            ],
        }
    )

    result = classify_column(
        "created_at",
        dataframe["created_at"],
    )

    assert result == ColumnSemanticType.DATETIME


def test_non_datetime_text():
    dataframe = pd.DataFrame(
        {
            "description": [
                "Customer joined the platform",
                "Customer purchased a product",
                "Customer contacted support",
                "Customer cancelled an order",
            ],
        }
    )

    result = classify_column(
        "description",
        dataframe["description"],
    )

    assert result == ColumnSemanticType.TEXT


def test_numeric_not_datetime():
    dataframe = pd.DataFrame(
        {
            "year": [
                2021,
                2022,
                2023,
                2024,
            ],
        }
    )

    result = classify_column(
        "year",
        dataframe["year"],
    )

    assert result == ColumnSemanticType.NUMERIC

def test_unique_column_is_candidate_primary_key():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
        }
    )

    result = analyze_key_candidate(
        "customer_id",
        dataframe["customer_id"],
    )

    assert result.is_unique is True
    assert result.uniqueness_ratio == 1.0
    assert result.is_candidate_primary_key is True


def test_duplicate_column_is_not_primary_key():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
        }
    )

    result = analyze_key_candidate(
        "customer_id",
        dataframe["customer_id"],
    )

    assert result.is_unique is False
    assert result.uniqueness_ratio == 0.75
    assert result.is_candidate_primary_key is False


def test_null_column_is_not_primary_key():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, None, 4],
        }
    )

    result = analyze_key_candidate(
        "customer_id",
        dataframe["customer_id"],
    )

    assert result.is_unique is False
    assert result.is_candidate_primary_key is False


def test_key_profile_is_added_to_dataset_profile():
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "name": [
                "Rahul",
                "Priya",
                "Amit",
                "Neha",
            ],
        }
    )

    profile = profile_dataset(
        dataframe,
        "customers.csv",
    )

    customer_id_profile = next(
        column
        for column in profile.columns
        if column.name == "customer_id"
    )

    assert customer_id_profile.key_profile is not None
    assert (
        customer_id_profile.key_profile
        .is_candidate_primary_key
        is True
    )

def test_relationship_candidate_for_matching_columns():
    left_dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
        }
    )

    right_dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
        }
    )

    left_profile = profile_dataset(
        left_dataframe,
        "customers.csv",
    )

    right_profile = profile_dataset(
        right_dataframe,
        "orders.csv",
    )

    left_column = left_profile.columns[0]
    right_column = right_profile.columns[0]

    result = generate_relationship_candidate(
        left_dataset="customers.csv",
        left_column=left_column,
        right_dataset="orders.csv",
        right_column=right_column,
    )

    assert result.left_dataset == "customers.csv"
    assert result.left_column == "customer_id"

    assert result.right_dataset == "orders.csv"
    assert result.right_column == "customer_id"

    assert result.name_similarity == 1.0
    assert result.semantic_compatibility == 1.0
    assert result.dtype_compatibility == 1.0
    assert result.candidate_score == 1.0


def test_unrelated_columns_have_lower_candidate_score():
    left_dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
        }
    )

    right_dataframe = pd.DataFrame(
        {
            "amount": [100, 200, 300, 400],
        }
    )

    left_profile = profile_dataset(
        left_dataframe,
        "customers.csv",
    )

    right_profile = profile_dataset(
        right_dataframe,
        "orders.csv",
    )

    result = generate_relationship_candidate(
        left_dataset="customers.csv",
        left_column=left_profile.columns[0],
        right_dataset="orders.csv",
        right_column=right_profile.columns[0],
    )

    assert result.candidate_score < 1.0
    assert result.name_similarity < 1.0
def test_value_overlap():
    left_series = pd.Series(
        [1, 2, 3, 4, 5]
    )

    right_series = pd.Series(
        [1, 1, 2, 3]
    )

    result = calculate_value_overlap(
        left_series,
        right_series,
    )

    assert result.common_value_count == 3

    assert result.left_unique_count == 5
    assert result.right_unique_count == 3

    assert result.left_overlap_ratio == 0.6
    assert result.right_overlap_ratio == 1.0


def test_no_value_overlap():
    left_series = pd.Series(
        [1, 2, 3]
    )

    right_series = pd.Series(
        [10, 11, 12]
    )

    result = calculate_value_overlap(
        left_series,
        right_series,
    )

    assert result.common_value_count == 0
    assert result.left_overlap_ratio == 0.0
    assert result.right_overlap_ratio == 0.0


def test_complete_value_overlap():
    left_series = pd.Series(
        [1, 2, 3]
    )

    right_series = pd.Series(
        [1, 2, 3]
    )

    result = calculate_value_overlap(
        left_series,
        right_series,
    )

    assert result.common_value_count == 3
    assert result.left_overlap_ratio == 1.0
    assert result.right_overlap_ratio == 1.0


def test_null_values_are_ignored():
    left_series = pd.Series(
        [1, 2, None, 3]
    )

    right_series = pd.Series(
        [2, 3, None, 4]
    )

    result = calculate_value_overlap(
        left_series,
        right_series,
    )

    assert result.common_value_count == 2
    assert result.left_unique_count == 3
    assert result.right_unique_count == 3

    assert result.left_overlap_ratio == round(
        2 / 3,
        6,
    )

    assert result.right_overlap_ratio == round(
        2 / 3,
        6,
    )

def test_one_to_one_cardinality():
    left_series = pd.Series(
        [1, 2, 3, 4]
    )

    right_series = pd.Series(
        [10, 20, 30, 40]
    )

    result = infer_cardinality(
        left_series,
        right_series,
    )

    assert result.cardinality == CardinalityType.ONE_TO_ONE
    assert result.left_unique is True
    assert result.right_unique is True


def test_one_to_many_cardinality():
    left_series = pd.Series(
        [1, 2, 3]
    )

    right_series = pd.Series(
        [1, 1, 2, 2, 3]
    )

    result = infer_cardinality(
        left_series,
        right_series,
    )

    assert result.cardinality == CardinalityType.ONE_TO_MANY
    assert result.left_unique is True
    assert result.right_unique is False


def test_many_to_one_cardinality():
    left_series = pd.Series(
        [1, 1, 2, 2, 3]
    )

    right_series = pd.Series(
        [1, 2, 3]
    )

    result = infer_cardinality(
        left_series,
        right_series,
    )

    assert result.cardinality == CardinalityType.MANY_TO_ONE
    assert result.left_unique is False
    assert result.right_unique is True


def test_many_to_many_cardinality():
    left_series = pd.Series(
        [1, 1, 2, 2]
    )

    right_series = pd.Series(
        [1, 1, 2, 2]
    )

    result = infer_cardinality(
        left_series,
        right_series,
    )

    assert result.cardinality == CardinalityType.MANY_TO_MANY
    assert result.left_unique is False
    assert result.right_unique is False


def test_null_values_do_not_break_cardinality():
    left_series = pd.Series(
        [1, 2, 3, None]
    )

    right_series = pd.Series(
        [10, 20, 30, None]
    )

    result = infer_cardinality(
        left_series,
        right_series,
    )

    assert result.cardinality == CardinalityType.ONE_TO_ONE

def test_high_confidence_relationship():
    candidate = RelationshipCandidate(
        left_dataset="customers.csv",
        left_column="customer_id",
        right_dataset="orders.csv",
        right_column="customer_id",
        name_similarity=1.0,
        semantic_compatibility=1.0,
        dtype_compatibility=1.0,
        candidate_score=1.0,
    )

    overlap = ValueOverlapProfile(
        common_value_count=3,
        left_unique_count=5,
        right_unique_count=3,
        left_overlap_ratio=0.6,
        right_overlap_ratio=1.0,
    )

    cardinality = CardinalityProfile(
        cardinality=CardinalityType.ONE_TO_MANY,
        left_unique=True,
        right_unique=False,
        left_unique_count=5,
        right_unique_count=3,
    )

    left_key = KeyProfile(
        column_name="customer_id",
        is_unique=True,
        uniqueness_ratio=1.0,
        is_candidate_primary_key=True,
    )

    result = calculate_relationship_confidence(
        candidate=candidate,
        overlap=overlap,
        cardinality=cardinality,
        left_key=left_key,
    )

    assert result.score == 1.0
    assert result.overlap_score == 1.0
    assert result.cardinality_score == 1.0
    assert result.key_score == 1.0


def test_zero_overlap_reduces_confidence():
    candidate = RelationshipCandidate(
        left_dataset="customers.csv",
        left_column="customer_id",
        right_dataset="orders.csv",
        right_column="customer_id",
        name_similarity=1.0,
        semantic_compatibility=1.0,
        dtype_compatibility=1.0,
        candidate_score=1.0,
    )

    overlap = ValueOverlapProfile(
        common_value_count=0,
        left_unique_count=5,
        right_unique_count=5,
        left_overlap_ratio=0.0,
        right_overlap_ratio=0.0,
    )

    cardinality = CardinalityProfile(
        cardinality=CardinalityType.ONE_TO_ONE,
        left_unique=True,
        right_unique=True,
        left_unique_count=5,
        right_unique_count=5,
    )

    result = calculate_relationship_confidence(
        candidate=candidate,
        overlap=overlap,
        cardinality=cardinality,
    )

    assert result.score == 0.55
    assert result.overlap_score == 0.0


def test_unrelated_columns_have_low_confidence():
    candidate = RelationshipCandidate(
        left_dataset="customers.csv",
        left_column="age",
        right_dataset="orders.csv",
        right_column="amount",
        name_similarity=0.2,
        semantic_compatibility=1.0,
        dtype_compatibility=1.0,
        candidate_score=0.58,
    )

    overlap = ValueOverlapProfile(
        common_value_count=0,
        left_unique_count=10,
        right_unique_count=10,
        left_overlap_ratio=0.0,
        right_overlap_ratio=0.0,
    )

    cardinality = CardinalityProfile(
        cardinality=CardinalityType.MANY_TO_MANY,
        left_unique=False,
        right_unique=False,
        left_unique_count=5,
        right_unique_count=5,
    )

    result = calculate_relationship_confidence(
        candidate=candidate,
        overlap=overlap,
        cardinality=cardinality,
    )

    assert result.score < 0.5


def test_discover_confirmed_relationship():
    left_dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
        }
    )

    right_dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
        }
    )

    left_profile = profile_dataset(
        left_dataframe,
        "customers.csv",
    )

    right_profile = profile_dataset(
        right_dataframe,
        "orders.csv",
    )

    result = discover_relationship(
        left_dataset="customers.csv",
        left_column=left_profile.columns[0],
        left_series=left_dataframe["customer_id"],
        right_dataset="orders.csv",
        right_column=right_profile.columns[0],
        right_series=right_dataframe["customer_id"],
    )

    assert isinstance(
        result,
        DiscoveredRelationship,
    )

    assert result.status == RelationshipStatus.CONFIRMED

    assert result.confidence.score == 1.0

    assert result.overlap.common_value_count == 3

    assert (
        result.cardinality.cardinality.value
        == "1:N"
    )


def test_discover_rejected_relationship():
    left_dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
        }
    )

    right_dataframe = pd.DataFrame(
        {
            "amount": [100, 200, 300, 400],
        }
    )

    left_profile = profile_dataset(
        left_dataframe,
        "customers.csv",
    )

    right_profile = profile_dataset(
        right_dataframe,
        "orders.csv",
    )

    result = discover_relationship(
        left_dataset="customers.csv",
        left_column=left_profile.columns[0],
        left_series=left_dataframe["customer_id"],
        right_dataset="orders.csv",
        right_column=right_profile.columns[0],
        right_series=right_dataframe["amount"],
    )

    assert result.status == RelationshipStatus.REJECTED

    assert result.confidence.score < POSSIBLE_THRESHOLD

    assert result.overlap.common_value_count == 0

def test_numeric_statistics():
    dataframe = pd.DataFrame(
        {
            "income": [
                100,
                200,
                300,
                400,
                500,
            ],
        }
    )

    result = calculate_numeric_statistics(
        "income",
        dataframe["income"],
    )

    assert result.column_name == "income"
    assert result.count == 5
    assert result.mean == 300.0
    assert result.median == 300.0
    assert result.minimum == 100.0
    assert result.maximum == 500.0

def test_numeric_statistics_handles_invalid_values():
    dataframe = pd.DataFrame(
        {
            "income": [
                100,
                200,
                "N/A",
                300,
                None,
            ],
        }
    )

    result = calculate_numeric_statistics(
        "income",
        dataframe["income"],
    )

    assert result.count == 3
    assert result.mean == 200.0
    assert result.minimum == 100.0
    assert result.maximum == 300.0


def test_positive_correlation():
    dataframe = pd.DataFrame(
        {
            "age": [10, 20, 30, 40, 50],
            "income": [100, 200, 300, 400, 500],
        }
    )

    result = calculate_correlation(
        "age",
        dataframe["age"],
        "income",
        dataframe["income"],
    )

    assert result.left_column == "age"
    assert result.right_column == "income"
    assert result.coefficient == 1.0
    assert result.method == "pearson"
    assert result.strength == "very_strong"

def test_negative_correlation():
    dataframe = pd.DataFrame(
        {
            "age": [10, 20, 30, 40, 50],
            "score": [50, 40, 30, 20, 10],
        }
    )

    result = calculate_correlation(
        "age",
        dataframe["age"],
        "score",
        dataframe["score"],
    )

    assert result.coefficient == -1.0
    assert result.strength == "very_strong"

def test_correlation_handles_invalid_values():
    dataframe = pd.DataFrame(
        {
            "age": [10, 20, "N/A", 40, None],
            "income": [100, 200, 300, "invalid", 500],
        }
    )

    result = calculate_correlation(
        "age",
        dataframe["age"],
        "income",
        dataframe["income"],
    )

    assert result.coefficient == 1.0

def test_categorical_analysis():
    dataframe = pd.DataFrame(
        {
            "city": [
                "Mumbai",
                "Mumbai",
                "Delhi",
                "Delhi",
                "Pune",
            ],
            "income": [
                50000,
                60000,
                40000,
                50000,
                70000,
            ],
        }
    )

    result = calculate_categorical_analysis(
        "city",
        dataframe["city"],
        "income",
        dataframe["income"],
    )

    assert result.categorical_column == "city"
    assert result.numeric_column == "income"
    assert len(result.categories) == 3

    mumbai = next(
        item
        for item in result.categories
        if item.category == "Mumbai"
    )

    assert mumbai.count == 2
    assert mumbai.mean == 55000.0
    assert mumbai.median == 55000.0


def test_categorical_analysis_handles_invalid_numeric_values():
    dataframe = pd.DataFrame(
        {
            "city": [
                "Mumbai",
                "Mumbai",
                "Delhi",
            ],
            "income": [
                50000,
                "N/A",
                40000,
            ],
        }
    )

    result = calculate_categorical_analysis(
        "city",
        dataframe["city"],
        "income",
        dataframe["income"],
    )

    mumbai = next(
        item
        for item in result.categories
        if item.category == "Mumbai"
    )

    assert mumbai.count == 1
    assert mumbai.mean == 50000.0

def test_iqr_anomaly_detection():
    dataframe = pd.DataFrame(
        {
            "income": [
                100,
                110,
                105,
                120,
                115,
                108,
                112,
                500,
            ]
        }
    )

    result = detect_iqr_anomalies(
        "income",
        dataframe["income"],
    )

    assert result.column_name == "income"
    assert result.method == "iqr"
    assert result.anomaly_count == 1
    assert result.anomaly_values == [500.0]


def test_iqr_anomaly_detection_without_anomalies():
    dataframe = pd.DataFrame(
        {
            "income": [
                100,
                110,
                105,
                120,
                115,
            ]
        }
    )

    result = detect_iqr_anomalies(
        "income",
        dataframe["income"],
    )

    assert result.anomaly_count == 0
    assert result.anomaly_values == []
    assert result.anomaly_percentage == 0.0

def test_iqr_anomaly_detection_handles_invalid_values():
    dataframe = pd.DataFrame(
        {
            "income": [
                100,
                110,
                "N/A",
                105,
                120,
                None,
                500,
            ]
        }
    )

    result = detect_iqr_anomalies(
        "income",
        dataframe["income"],
    )

    assert result.anomaly_count == 1
    assert result.anomaly_values == [500.0]

def test_metric_role_for_numeric_measure():
    column = ColumnProfile(
        name="revenue",
        dtype="int64",
        semantic_type=ColumnSemanticType.NUMERIC,
        nullable=False,
        null_count=0,
        null_percentage=0.0,
        unique_count=8,
        unique_percentage=80.0,
    )

    result = discover_metric_role(column)

    assert result.role == MetricRole.MEASURE
    assert result.confidence > 0.8
    assert "numeric_column" in result.signals
    assert "business_measure_name" in result.signals

def test_metric_role_for_datetime():
    column = ColumnProfile(
        name="order_date",
        dtype="datetime64[ns]",
        semantic_type=ColumnSemanticType.DATETIME,
        nullable=False,
        null_count=0,
        null_percentage=0.0,
        unique_count=10,
        unique_percentage=100.0,
    )

    result = discover_metric_role(column)

    assert result.role == MetricRole.TIME
    assert result.confidence == 1.0

def test_metric_role_for_identifier():
    column = ColumnProfile(
        name="customer_id",
        dtype="int64",
        semantic_type=ColumnSemanticType.IDENTIFIER,
        nullable=False,
        null_count=0,
        null_percentage=0.0,
        unique_count=10,
        unique_percentage=100.0,
    )

    result = discover_metric_role(column)

    assert result.role == MetricRole.ENTITY
    assert result.confidence == 1.0

def test_metric_role_for_categorical_column():
    column = ColumnProfile(
        name="city",
        dtype="object",
        semantic_type=ColumnSemanticType.CATEGORICAL,
        nullable=False,
        null_count=0,
        null_percentage=0.0,
        unique_count=5,
        unique_percentage=50.0,
    )

    result = discover_metric_role(column)

    assert result.role == MetricRole.DIMENSION
    assert result.confidence == 0.9

def test_analysis_planner():
    profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=0.95,
            signals=["numeric_column", "business_measure_name"],
        ),
        MetricProfile(
            column_name="quantity",
            role=MetricRole.MEASURE,
            confidence=0.90,
            signals=["numeric_column"],
        ),
        MetricProfile(
            column_name="city",
            role=MetricRole.DIMENSION,
            confidence=0.90,
            signals=["categorical_column"],
        ),
        MetricProfile(
            column_name="order_date",
            role=MetricRole.TIME,
            confidence=1.0,
            signals=["datetime_column"],
        ),
    ]

    result = build_analysis_plan(profiles)

    analysis_types = [
        item.analysis_type
        for item in result.analyses
    ]

    assert AnalysisType.NUMERIC_STATISTICS in analysis_types
    assert AnalysisType.ANOMALY_DETECTION in analysis_types
    assert AnalysisType.DISTRIBUTION_ANALYSIS in analysis_types
    assert AnalysisType.CORRELATION in analysis_types
    assert AnalysisType.CATEGORICAL_ANALYSIS in analysis_types
    assert AnalysisType.TREND_ANALYSIS in analysis_types


def test_analysis_planner_creates_measure_correlation():
    profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=0.95,
            signals=["numeric_column"],
        ),
        MetricProfile(
            column_name="quantity",
            role=MetricRole.MEASURE,
            confidence=0.90,
            signals=["numeric_column"],
        ),
    ]

    result = build_analysis_plan(profiles)

    correlations = [
        item
        for item in result.analyses
        if item.analysis_type == AnalysisType.CORRELATION
    ]

    assert len(correlations) == 1

    assert set(correlations[0].target_columns) == {
        "revenue",
        "quantity",
    }

def test_analysis_planner_creates_trend_analysis():
    profiles = [
        MetricProfile(
            column_name="order_date",
            role=MetricRole.TIME,
            confidence=1.0,
            signals=["datetime_column"],
        ),
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=0.95,
            signals=["numeric_column"],
        ),
    ]

    result = build_analysis_plan(profiles)

    trends = [
        item
        for item in result.analyses
        if item.analysis_type == AnalysisType.TREND_ANALYSIS
    ]

    assert len(trends) == 1

    assert trends[0].target_columns == [
        "order_date",
        "revenue",
    ]

def test_analysis_planner_creates_segment_analysis():
    profiles = [
        MetricProfile(
            column_name="city",
            role=MetricRole.DIMENSION,
            confidence=0.9,
            signals=["categorical_column"],
        ),
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=0.95,
            signals=["numeric_column"],
        ),
    ]

    result = build_analysis_plan(profiles)

    categorical_analyses = [
        item
        for item in result.analyses
        if item.analysis_type
        == AnalysisType.CATEGORICAL_ANALYSIS
    ]

    assert len(categorical_analyses) == 1

    assert categorical_analyses[0].target_columns == [
        "city",
        "revenue",
    ]   

def test_trend_analysis_calculates_monthly_growth():
    from app.data.trend import calculate_trend

    dates = pd.Series([
        "2026-01-01",
        "2026-01-15",
        "2026-02-01",
        "2026-03-01",
    ])

    values = pd.Series([100, 200, 400, 500])

    result = calculate_trend(
        "date",
        dates,
        "revenue",
        values,
    )

    assert len(result.points) == 3
    assert result.points[0].value == 300
    assert result.points[1].value == 400
    assert result.points[1].change == 100
    assert result.points[1].change_percentage == 33.333333
    assert result.overall_direction.value == "growth"


def test_trend_analysis_sorts_periods():
    from app.data.trend import calculate_trend

    dates = pd.Series([
        "2026-03-01",
        "2026-01-01",
        "2026-02-01",
    ])

    values = pd.Series([300, 100, 200])

    result = calculate_trend(
        "date",
        dates,
        "sales",
        values,
    )

    assert [point.period for point in result.points] == [
        "2026-01",
        "2026-02",
        "2026-03",
    ]


def test_trend_analysis_ignores_invalid_values():
    from app.data.trend import calculate_trend

    dates = pd.Series([
        "2026-01-01",
        "invalid",
        "2026-02-01",
    ])

    values = pd.Series([
        100,
        500,
        "invalid",
    ])

    result = calculate_trend(
        "date",
        dates,
        "sales",
        values,
    )

    assert len(result.points) == 1
    assert result.points[0].value == 100
    assert result.overall_direction.value == "insufficient_data"


def test_trend_analysis_handles_zero_previous_value():
    from app.data.trend import calculate_trend

    dates = pd.Series([
        "2026-01-01",
        "2026-02-01",
    ])

    values = pd.Series([0, 100])

    result = calculate_trend(
        "date",
        dates,
        "sales",
        values,
    )

    assert result.points[1].change == 100
    assert result.points[1].change_percentage is None
    assert result.overall_direction.value == "growth"


def test_trend_analysis_detects_decline():
    from app.data.trend import calculate_trend

    dates = pd.Series([
        "2026-01-01",
        "2026-02-01",
        "2026-03-01",
    ])

    values = pd.Series([500, 400, 300])

    result = calculate_trend(
        "date",
        dates,
        "sales",
        values,
    )

    assert result.overall_direction.value == "decline"
    assert result.total_change == -200
    assert result.total_change_percentage == -40


def test_contribution_analysis_calculates_category_contribution():
    from app.data.contribution import calculate_contribution_analysis

    categories = pd.Series(["A", "A", "B", "C"])
    values = pd.Series([100, 100, 50, 50])

    result = calculate_contribution_analysis(
        "category",
        categories,
        "revenue",
        values,
    )

    assert result.total_value == 300
    assert result.categories[0].category == "A"
    assert result.categories[0].value == 200
    assert result.categories[0].contribution_percentage == 66.666667
    assert result.categories[0].rank == 1


def test_contribution_analysis_ranks_categories():
    from app.data.contribution import calculate_contribution_analysis

    categories = pd.Series(["A", "B", "C"])
    values = pd.Series([10, 30, 20])

    result = calculate_contribution_analysis(
        "category",
        categories,
        "sales",
        values,
    )

    assert [item.category for item in result.categories] == [
        "B",
        "C",
        "A",
    ]

    assert [item.rank for item in result.categories] == [1, 2, 3]


def test_contribution_analysis_ignores_invalid_values():
    from app.data.contribution import calculate_contribution_analysis

    categories = pd.Series(["A", "B", "C"])
    values = pd.Series([100, "invalid", None])

    result = calculate_contribution_analysis(
        "category",
        categories,
        "revenue",
        values,
    )

    assert result.total_value == 100
    assert len(result.categories) == 1


def test_contribution_analysis_supports_mean():
    from app.data.contribution import calculate_contribution_analysis

    categories = pd.Series(["A", "A", "B", "B"])
    values = pd.Series([100, 200, 50, 150])

    result = calculate_contribution_analysis(
        "category",
        categories,
        "score",
        values,
        aggregation="mean",
    )

    assert result.categories[0].category == "A"
    assert result.categories[0].value == 150
    assert result.categories[1].value == 100


def test_contribution_analysis_handles_zero_total():
    from app.data.contribution import calculate_contribution_analysis

    categories = pd.Series(["A", "B"])
    values = pd.Series([0, 0])

    result = calculate_contribution_analysis(
        "category",
        categories,
        "revenue",
        values,
    )

    assert result.total_value == 0
    assert all(
        item.contribution_percentage == 0
        for item in result.categories
    )

def test_distribution_analysis_calculates_statistics():
    from app.data.distribution import calculate_distribution_analysis

    values = pd.Series([10, 20, 30, 40, 50])

    result = calculate_distribution_analysis(
        "revenue",
        values,
    )

    assert result.count == 5
    assert result.minimum == 10
    assert result.maximum == 50
    assert result.mean == 30
    assert result.median == 30
    assert result.q1 == 20
    assert result.q3 == 40
    assert result.iqr == 20


def test_distribution_analysis_detects_right_skew():
    from app.data.distribution import calculate_distribution_analysis

    values = pd.Series([
        10, 10, 10, 10, 20, 100
    ])

    result = calculate_distribution_analysis(
        "income",
        values,
    )

    assert result.skewness > 0
    assert result.distribution_shape == "right_skewed"


def test_distribution_analysis_detects_left_skew():
    from app.data.distribution import calculate_distribution_analysis

    values = pd.Series([
        1, 80, 90, 90, 90, 90
    ])

    result = calculate_distribution_analysis(
        "score",
        values,
    )

    assert result.skewness < 0
    assert result.distribution_shape == "left_skewed"


def test_distribution_analysis_ignores_invalid_values():
    from app.data.distribution import calculate_distribution_analysis

    values = pd.Series([
        10,
        20,
        "invalid",
        None,
        30,
    ])

    result = calculate_distribution_analysis(
        "sales",
        values,
    )

    assert result.count == 3
    assert result.minimum == 10
    assert result.maximum == 30


def test_distribution_analysis_handles_empty_data():
    from app.data.distribution import calculate_distribution_analysis

    values = pd.Series([
        "invalid",
        None,
    ])

    result = calculate_distribution_analysis(
        "revenue",
        values,
    )

    assert result.count == 0
    assert result.distribution_shape == "insufficient_data"


def test_segment_relationship_calculates_overall_statistics():
    from app.data.segment_relationship import calculate_segment_relationship

    categories = pd.Series(["A", "A", "B", "B"])
    values = pd.Series([100, 200, 300, 400])

    result = calculate_segment_relationship(
        "category",
        categories,
        "revenue",
        values,
    )

    assert result.overall_mean == 250
    assert result.overall_median == 250


def test_segment_relationship_calculates_segment_difference():
    from app.data.segment_relationship import calculate_segment_relationship

    categories = pd.Series(["A", "A", "B", "B"])
    values = pd.Series([100, 200, 300, 400])

    result = calculate_segment_relationship(
        "category",
        categories,
        "revenue",
        values,
    )

    segment_a = next(
        segment for segment in result.segments
        if segment.category == "A"
    )

    assert segment_a.mean == 150
    assert segment_a.difference_from_overall == -100
    assert segment_a.difference_percentage == -40


def test_segment_relationship_identifies_highest_and_lowest():
    from app.data.segment_relationship import calculate_segment_relationship

    categories = pd.Series(["A", "A", "B", "B", "C", "C"])
    values = pd.Series([10, 20, 50, 60, 30, 40])

    result = calculate_segment_relationship(
        "category",
        categories,
        "sales",
        values,
    )

    assert result.highest_segment == "B"
    assert result.lowest_segment == "A"


def test_segment_relationship_ignores_invalid_values():
    from app.data.segment_relationship import calculate_segment_relationship

    categories = pd.Series(["A", "B", "C"])
    values = pd.Series([100, "invalid", None])

    result = calculate_segment_relationship(
        "category",
        categories,
        "revenue",
        values,
    )

    assert len(result.segments) == 1
    assert result.segments[0].category == "A"
    assert result.segments[0].mean == 100


def test_segment_relationship_handles_empty_data():
    from app.data.segment_relationship import calculate_segment_relationship

    categories = pd.Series(["A", "B"])
    values = pd.Series(["invalid", None])

    result = calculate_segment_relationship(
        "category",
        categories,
        "revenue",
        values,
    )

    assert result.overall_mean is None
    assert result.overall_median is None
    assert result.segments == []
    assert result.highest_segment is None
    assert result.lowest_segment is None

def test_insight_signals_detect_strong_correlation():
    from app.data.model import CorrelationProfile
    from app.data.signals import detect_insight_signals

    correlation = CorrelationProfile(
        left_column="revenue",
        right_column="profit",
        coefficient=0.85,
        method="pearson",
        strength="very_strong",
    )

    result = detect_insight_signals(
        correlations=[correlation]
    )

    assert len(result.signals) == 1
    assert result.signals[0].signal_type.value == "strong_correlation"
    assert result.signals[0].strength == 0.85


def test_insight_signals_detect_anomaly():
    from app.data.model import AnomalyProfile
    from app.data.signals import detect_insight_signals

    anomaly = AnomalyProfile(
        column_name="revenue",
        method="iqr",
        lower_bound=10,
        upper_bound=100,
        anomaly_count=2,
        anomaly_percentage=10,
        anomaly_values=[150, 200],
    )

    result = detect_insight_signals(
        anomalies=[anomaly]
    )

    assert len(result.signals) == 1
    assert result.signals[0].signal_type.value == "anomaly"
    assert result.signals[0].evidence["anomaly_count"] == 2


def test_insight_signals_detect_trend():
    from app.data.model import (
        TrendDirection,
        TrendPoint,
        TrendProfile,
    )
    from app.data.signals import detect_insight_signals

    trend = TrendProfile(
        time_column="date",
        measure_column="revenue",
        frequency="M",
        aggregation="sum",
        points=[
            TrendPoint(period="2026-01", value=100),
            TrendPoint(
                period="2026-02",
                value=150,
                change=50,
                change_percentage=50,
            ),
        ],
        overall_direction=TrendDirection.GROWTH,
        total_change=50,
        total_change_percentage=50,
    )

    result = detect_insight_signals(
        trends=[trend]
    )

    assert len(result.signals) == 1
    assert result.signals[0].signal_type.value == "trend"
    assert result.signals[0].evidence["direction"] == "growth"


def test_insight_signals_detect_dominant_segment():
    from app.data.model import (
        ContributionAnalysisProfile,
        ContributionStatistics,
    )
    from app.data.signals import detect_insight_signals

    contribution = ContributionAnalysisProfile(
        categorical_column="city",
        measure_column="revenue",
        aggregation="sum",
        total_value=1000,
        categories=[
            ContributionStatistics(
                category="Pune",
                value=700,
                contribution_percentage=70,
                rank=1,
            )
        ],
    )

    result = detect_insight_signals(
        contributions=[contribution]
    )

    assert len(result.signals) == 1
    assert result.signals[0].signal_type.value == "dominant_segment"
    assert result.signals[0].evidence["category"] == "Pune"


def test_insight_signals_detect_segment_difference():
    from app.data.model import (
        SegmentComparison,
        SegmentRelationshipProfile,
    )
    from app.data.signals import detect_insight_signals

    relationship = SegmentRelationshipProfile(
        categorical_column="city",
        measure_column="income",
        overall_mean=100,
        overall_median=100,
        segments=[
            SegmentComparison(
                category="Pune",
                count=10,
                mean=130,
                median=125,
                difference_from_overall=30,
                difference_percentage=30,
            )
        ],
        highest_segment="Pune",
        lowest_segment="Mumbai",
    )

    result = detect_insight_signals(
        segment_relationships=[relationship]
    )

    assert len(result.signals) == 1
    assert result.signals[0].signal_type.value == "segment_difference"
    assert result.signals[0].evidence["category"] == "Pune"


def test_tool_registry_contains_expected_tools():
    from app.data.tool_registry import build_analysis_tool_registry

    registry = build_analysis_tool_registry()

    names = {tool.name for tool in registry.tools}

    assert "numeric_statistics" in names
    assert "correlation" in names
    assert "categorical_analysis" in names
    assert "anomaly_detection" in names
    assert "distribution_analysis" in names
    assert "trend_analysis" in names


def test_tool_registry_contains_six_tools():
    from app.data.tool_registry import build_analysis_tool_registry

    registry = build_analysis_tool_registry()

    assert len(registry.tools) == 9


def test_tool_registry_has_correct_roles_for_trend():
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.model import MetricRole

    registry = build_analysis_tool_registry()

    tool = next(
        tool for tool in registry.tools
        if tool.name == "trend_analysis"
    )

    assert tool.required_roles == [
        MetricRole.TIME,
        MetricRole.MEASURE,
    ]


def test_tool_registry_has_correct_roles_for_categorical_analysis():
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.model import MetricRole

    registry = build_analysis_tool_registry()

    tool = next(
        tool for tool in registry.tools
        if tool.name == "categorical_analysis"
    )

    assert tool.required_roles == [
        MetricRole.DIMENSION,
        MetricRole.MEASURE,
    ]


def test_tool_registry_tool_names_are_unique():
    from app.data.tool_registry import build_analysis_tool_registry

    registry = build_analysis_tool_registry()

    names = [tool.name for tool in registry.tools]

    assert len(names) == len(set(names))

def test_tool_registry_contains_advanced_tools():
    from app.data.tool_registry import build_analysis_tool_registry

    registry = build_analysis_tool_registry()

    names = {tool.name for tool in registry.tools}

    assert "contribution_analysis" in names
    assert "segment_relationship" in names
    assert "insight_signals" in names

def test_tool_executor_runs_numeric_statistics():
    from app.data.tool_executor import execute_tool

    dataframe = pd.DataFrame({
        "revenue": [100, 200, 300],
    })

    result = execute_tool(
        "numeric_statistics",
        dataframe,
        ["revenue"],
    )

    assert result.column_name == "revenue"
    assert result.mean == 200


def test_tool_executor_runs_correlation():
    from app.data.tool_executor import execute_tool

    dataframe = pd.DataFrame({
        "revenue": [1, 2, 3],
        "profit": [2, 4, 6],
    })

    result = execute_tool(
        "correlation",
        dataframe,
        ["revenue", "profit"],
    )

    assert result.coefficient == 1.0


def test_tool_executor_runs_trend_analysis():
    from app.data.tool_executor import execute_tool

    dataframe = pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-02-01",
        ],
        "revenue": [
            100,
            200,
        ],
    })

    result = execute_tool(
        "trend_analysis",
        dataframe,
        ["date", "revenue"],
    )

    assert len(result.points) == 2
    assert result.total_change == 100


def test_tool_executor_rejects_invalid_column_count():
    from app.data.tool_executor import execute_tool

    dataframe = pd.DataFrame({
        "revenue": [100, 200],
    })

    with pytest.raises(ValueError):
        execute_tool(
            "numeric_statistics",
            dataframe,
            ["revenue", "extra"],
        )


def test_tool_executor_rejects_unknown_tool():
    from app.data.tool_executor import execute_tool

    dataframe = pd.DataFrame({
        "revenue": [100, 200],
    })

    with pytest.raises(ValueError):
        execute_tool(
            "unknown_tool",
            dataframe,
            ["revenue"],
        )

def test_tool_validator_accepts_valid_request():
    from app.data.model import ToolRequest
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "revenue": [100, 200, 300],
    })

    request = ToolRequest(
        tool_name="numeric_statistics",
        target_columns=["revenue"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
    )

    assert result.valid is True
    assert result.errors == []


def test_tool_validator_rejects_unknown_tool():
    from app.data.model import ToolRequest
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "revenue": [100, 200, 300],
    })

    request = ToolRequest(
        tool_name="fake_tool",
        target_columns=["revenue"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
    )

    assert result.valid is False
    assert "Unknown tool" in result.errors[0]


def test_tool_validator_rejects_missing_column():
    from app.data.model import ToolRequest
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "revenue": [100, 200, 300],
    })

    request = ToolRequest(
        tool_name="numeric_statistics",
        target_columns=["profit"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
    )

    assert result.valid is False
    assert "Columns not found" in result.errors[0]


def test_tool_validator_rejects_wrong_column_count():
    from app.data.model import ToolRequest
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "revenue": [100, 200, 300],
        "profit": [20, 40, 60],
    })

    request = ToolRequest(
        tool_name="numeric_statistics",
        target_columns=["revenue", "profit"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
    )

    assert result.valid is False
    assert any(
        "requires 1 target column" in error
        for error in result.errors
    )


def test_tool_validator_validates_trend_columns():
    from app.data.model import ToolRequest
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "date": ["2026-01-01", "2026-02-01"],
        "revenue": [100, 200],
    })

    request = ToolRequest(
        tool_name="trend_analysis",
        target_columns=["date", "revenue"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
    )

    assert result.valid is True


def test_schema_validator_accepts_matching_measure_role():
    from app.data.model import (
        MetricProfile,
        MetricRole,
        ToolRequest,
    )
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "revenue": [100, 200, 300],
    })

    profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=0.9,
            signals=["numeric_column"],
        )
    ]

    request = ToolRequest(
        tool_name="numeric_statistics",
        target_columns=["revenue"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
        profiles,
    )

    assert result.valid is True
    assert result.errors == []


def test_schema_validator_rejects_wrong_measure_role():
    from app.data.model import (
        MetricProfile,
        MetricRole,
        ToolRequest,
    )
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "city": ["Pune", "Mumbai", "Delhi"],
    })

    profiles = [
        MetricProfile(
            column_name="city",
            role=MetricRole.DIMENSION,
            confidence=0.9,
            signals=["categorical_column"],
        )
    ]

    request = ToolRequest(
        tool_name="numeric_statistics",
        target_columns=["city"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
        profiles,
    )

    assert result.valid is False
    assert any("requires 'measure'" in error for error in result.errors)


def test_schema_validator_accepts_valid_trend_roles():
    from app.data.model import (
        MetricProfile,
        MetricRole,
        ToolRequest,
    )
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "date": ["2026-01-01", "2026-02-01"],
        "revenue": [100, 200],
    })

    profiles = [
        MetricProfile(
            column_name="date",
            role=MetricRole.TIME,
            confidence=1.0,
            signals=["datetime_column"],
        ),
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=0.9,
            signals=["numeric_column"],
        ),
    ]

    request = ToolRequest(
        tool_name="trend_analysis",
        target_columns=["date", "revenue"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
        profiles,
    )

    assert result.valid is True


def test_schema_validator_rejects_invalid_trend_roles():
    from app.data.model import (
        MetricProfile,
        MetricRole,
        ToolRequest,
    )
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "city": ["Pune", "Mumbai"],
        "revenue": [100, 200],
    })

    profiles = [
        MetricProfile(
            column_name="city",
            role=MetricRole.DIMENSION,
            confidence=0.9,
            signals=["categorical_column"],
        ),
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=0.9,
            signals=["numeric_column"],
        ),
    ]

    request = ToolRequest(
        tool_name="trend_analysis",
        target_columns=["city", "revenue"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
        profiles,
    )

    assert result.valid is False
    assert any("requires 'time'" in error for error in result.errors)


def test_schema_validator_handles_missing_metric_profile():
    from app.data.model import (
        MetricProfile,
        MetricRole,
        ToolRequest,
    )
    from app.data.tool_registry import build_analysis_tool_registry
    from app.data.tool_validator import validate_tool_request

    dataframe = pd.DataFrame({
        "revenue": [100, 200, 300],
    })

    profiles = []

    request = ToolRequest(
        tool_name="numeric_statistics",
        target_columns=["revenue"],
    )

    result = validate_tool_request(
        request,
        dataframe,
        build_analysis_tool_registry(),
        profiles,
    )

    assert result.valid is False
    assert any(
        "No metric profile found" in error
        for error in result.errors
    )

def test_agent_state_is_created():
    from app.agent.state import create_agent_state
    from app.data.model import MetricProfile, MetricRole

    profile = MetricProfile(
        column_name="revenue",
        role=MetricRole.MEASURE,
        confidence=0.9,
        signals=["numeric_column"],
    )

    state = create_agent_state(
        user_goal="Find important revenue insights",
        dataset_names=["customers.csv"],
        metric_profiles=[profile],
        available_tools=[],
    )

    assert state.user_goal == "Find important revenue insights"
    assert state.dataset_names == ["customers.csv"]
    assert len(state.metric_profiles) == 1
    assert state.status.value == "initialized"


def test_agent_state_updates_status():
    from app.agent.state import (
        create_agent_state,
        update_agent_status,
    )
    from app.data.model import AgentStatus

    state = create_agent_state(
        user_goal="Analyze revenue",
        dataset_names=["customers.csv"],
        metric_profiles=[],
        available_tools=[],
    )

    update_agent_status(
        state,
        AgentStatus.EXECUTING,
    )

    assert state.status == AgentStatus.EXECUTING


def test_agent_state_attaches_analysis_plan():
    from app.agent.state import (
        attach_analysis_plan,
        create_agent_state,
    )
    from app.data.model import AnalysisPlan, AnalysisPlanItem, AnalysisType

    state = create_agent_state(
        user_goal="Analyze revenue",
        dataset_names=["customers.csv"],
        metric_profiles=[],
        available_tools=[],
    )

    plan = AnalysisPlan(
        analyses=[
            AnalysisPlanItem(
                analysis_type=AnalysisType.NUMERIC_STATISTICS,
                target_columns=["revenue"],
                reason="Analyze business measure.",
            )
        ]
    )

    attach_analysis_plan(state, plan)

    assert state.analysis_plan is not None
    assert len(state.analysis_plan.analyses) == 1
    assert state.status.value == "planning"


def test_agent_state_records_tool_execution():
    from app.agent.state import (
        add_tool_execution,
        create_agent_state,
    )
    from app.data.model import ToolExecution

    state = create_agent_state(
        user_goal="Analyze revenue",
        dataset_names=["customers.csv"],
        metric_profiles=[],
        available_tools=[],
    )

    execution = ToolExecution(
        tool_name="numeric_statistics",
        target_columns=["revenue"],
        success=True,
    )

    add_tool_execution(state, execution)

    assert len(state.executions) == 1
    assert state.executions[0].tool_name == "numeric_statistics"
    assert state.executions[0].success is True


def test_agent_state_records_insight_signal():
    from app.agent.state import (
        add_insight_signal,
        create_agent_state,
    )
    from app.data.model import (
        InsightSignal,
        InsightSignalType,
    )

    state = create_agent_state(
        user_goal="Find important insights",
        dataset_names=["customers.csv"],
        metric_profiles=[],
        available_tools=[],
    )

    signal = InsightSignal(
        signal_type=InsightSignalType.TREND,
        columns=["date", "revenue"],
        strength=0.8,
        evidence={
            "direction": "growth",
        },
        reason="Revenue changed over time.",
    )

    add_insight_signal(state, signal)

    assert len(state.insight_signals) == 1
    assert (
        state.insight_signals[0].signal_type
        == InsightSignalType.TREND
    )

def test_agent_planner_identifies_measure_analysis():
    from app.agent.planner import build_agent_decisions
    from app.data.model import (
        AgentState,
        AnalysisTool,
        MetricProfile,
        MetricRole,
    )

    state = AgentState(
        user_goal="Find important business insights",
        dataset_names=["customers.csv"],
        metric_profiles=[
            MetricProfile(
                column_name="revenue",
                role=MetricRole.MEASURE,
                confidence=0.9,
            )
        ],
        available_tools=[
            AnalysisTool(
                name="numeric_statistics",
                description="Calculate statistics",
                analysis_type="numeric_statistics",
                required_roles=[MetricRole.MEASURE],
            )
        ],
    )

    decisions = build_agent_decisions(state)

    assert len(decisions) == 1
    assert decisions[0].tool_name == "numeric_statistics"
    assert decisions[0].target_columns == ["revenue"]

def test_agent_planner_requests_anomaly_analysis():
    from app.agent.planner import build_agent_decisions
    from app.data.model import (
        AgentState,
        AnalysisTool,
        MetricProfile,
        MetricRole,
    )

    state = AgentState(
        user_goal="Find unusual business behavior",
        dataset_names=["customers.csv"],
        metric_profiles=[
            MetricProfile(
                column_name="revenue",
                role=MetricRole.MEASURE,
                confidence=0.9,
            )
        ],
        available_tools=[
            AnalysisTool(
                name="anomaly_detection",
                description="Detect anomalies",
                analysis_type="anomaly_detection",
                required_roles=[MetricRole.MEASURE],
            )
        ],
    )

    decisions = build_agent_decisions(state)

    assert len(decisions) == 1
    assert decisions[0].decision_type == "investigate_anomalies"
    assert decisions[0].tool_name == "anomaly_detection"

def test_agent_planner_requests_segment_analysis():
    from app.agent.planner import build_agent_decisions
    from app.data.model import (
        AgentState,
        AnalysisTool,
        MetricProfile,
        MetricRole,
    )

    state = AgentState(
        user_goal="Understand revenue across customer segments",
        dataset_names=["customers.csv"],
        metric_profiles=[
            MetricProfile(
                column_name="region",
                role=MetricRole.DIMENSION,
                confidence=0.9,
            ),
            MetricProfile(
                column_name="revenue",
                role=MetricRole.MEASURE,
                confidence=0.9,
            ),
        ],
        available_tools=[
            AnalysisTool(
                name="categorical_analysis",
                description="Compare measure across segments",
                analysis_type="categorical_analysis",
                required_roles=[
                    MetricRole.DIMENSION,
                    MetricRole.MEASURE,
                ],
            )
        ],
    )

    decisions = build_agent_decisions(state)

    assert len(decisions) == 1
    assert decisions[0].decision_type == "compare_segments"
    assert decisions[0].target_columns == ["region", "revenue"]

def test_agent_planner_requests_trend_analysis():
    from app.agent.planner import build_agent_decisions
    from app.data.model import (
        AgentState,
        AnalysisTool,
        MetricProfile,
        MetricRole,
    )

    state = AgentState(
        user_goal="Understand revenue growth",
        dataset_names=["customers.csv"],
        metric_profiles=[
            MetricProfile(
                column_name="order_date",
                role=MetricRole.TIME,
                confidence=1.0,
            ),
            MetricProfile(
                column_name="revenue",
                role=MetricRole.MEASURE,
                confidence=0.9,
            ),
        ],
        available_tools=[
            AnalysisTool(
                name="trend_analysis",
                description="Analyze trends",
                analysis_type="trend_analysis",
                required_roles=[
                    MetricRole.TIME,
                    MetricRole.MEASURE,
                ],
            )
        ],
    )

    decisions = build_agent_decisions(state)

    assert len(decisions) == 1
    assert decisions[0].decision_type == "investigate_trend"
    assert decisions[0].target_columns == [
        "order_date",
        "revenue",
    ]

def test_agent_planner_ignores_unavailable_tools():
    from app.agent.planner import build_agent_decisions
    from app.data.model import (
        AgentState,
        MetricProfile,
        MetricRole,
    )

    state = AgentState(
        user_goal="Analyze revenue",
        dataset_names=["customers.csv"],
        metric_profiles=[
            MetricProfile(
                column_name="revenue",
                role=MetricRole.MEASURE,
                confidence=0.9,
            )
        ],
        available_tools=[],
    )

    decisions = build_agent_decisions(state)

    assert decisions == []

def test_agent_tool_call_contract():
    from app.data.model import AgentToolCall

    call = AgentToolCall(
        tool_name="correlation",
        target_columns=["revenue", "orders"],
        reasoning="Investigate whether order volume is related to revenue.",
    )

    assert call.tool_name == "correlation"
    assert call.target_columns == ["revenue", "orders"]
    assert call.reasoning != ""

def test_agent_observation_contract():
    from app.data.model import AgentObservation

    observation = AgentObservation(
        tool_name="numeric_statistics",
        target_columns=["revenue"],
        success=True,
        result={"mean": 500},
    )

    assert observation.success is True
    assert observation.result["mean"] == 500

def test_agent_action_can_request_tool():
    from app.data.model import (
        AgentAction,
        AgentActionType,
        AgentToolCall,
    )

    action = AgentAction(
        action_type=AgentActionType.RUN_TOOL,
        tool_call=AgentToolCall(
            tool_name="anomaly_detection",
            target_columns=["revenue"],
            reasoning="Investigate unusual revenue values.",
        ),
        reason="Potential unusual observations should be investigated.",
    )

    assert action.action_type == AgentActionType.RUN_TOOL
    assert action.tool_call is not None
    assert action.tool_call.tool_name == "anomaly_detection"

def test_agent_action_can_finish():
    from app.data.model import (
        AgentAction,
        AgentActionType,
    )

    action = AgentAction(
        action_type=AgentActionType.FINISH,
        reason="Enough evidence has been collected.",
    )

    assert action.action_type == AgentActionType.FINISH
    assert action.tool_call is None

def test_mock_llm_returns_agent_action():
    from app.agent.mock_llm import MockAgentLLM
    from app.data.model import (
        AgentAction,
        AgentActionType,
        AgentState,
    )

    action = AgentAction(
        action_type=AgentActionType.FINISH,
        reason="Testing agent completion.",
    )

    llm = MockAgentLLM(action)

    state = AgentState(
        user_goal="Analyze dataset",
        dataset_names=["customers.csv"],
        metric_profiles=[],
        available_tools=[],
    )

    result = llm.decide(state, [])

    assert result.action_type == AgentActionType.FINISH


def test_agent_loop_can_execute_tool_and_finish():
    import pandas as pd

    from app.agent.loop import AgentLoop
    from app.agent.mock_llm import MockAgentLLM
    from app.agent.state import create_agent_state
    from app.data.model import (
        AgentAction,
        AgentActionType,
        AgentToolCall,
        AgentStatus,
        MetricProfile,
        MetricRole,
    )
    from app.data.tool_registry import build_analysis_tool_registry

    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300, 400],
        }
    )

    metric_profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=1.0,
            signals=["numeric_column"],
        )
    ]

    registry = build_analysis_tool_registry()

    state = create_agent_state(
        user_goal="Understand business performance.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    actions = [
        AgentAction(
            action_type=AgentActionType.RUN_TOOL,
            tool_call=AgentToolCall(
                tool_name="numeric_statistics",
                target_columns=["revenue"],
                reasoning="Analyze the main business measure.",
            ),
            reason="Start with revenue statistics.",
        ),
        AgentAction(
            action_type=AgentActionType.FINISH,
            reason="Sufficient analysis completed.",
        ),
    ]

    class SequentialMockLLM:
        def __init__(self, actions):
            self.actions = actions
            self.index = 0

        def decide(self, state, observations):
            action = self.actions[self.index]
            self.index += 1
            return action

    llm = SequentialMockLLM(actions)

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        max_steps=5,
    )

    result = loop.run(
        state=state,
        dataframe=dataframe,
    )

    assert result.status == AgentStatus.COMPLETED
    assert len(result.executions) == 1
    assert result.executions[0].success is True
    assert result.executions[0].tool_name == "numeric_statistics"

def test_agent_loop_respects_max_steps():
    import pandas as pd

    from app.agent.loop import AgentLoop
    from app.agent.state import create_agent_state
    from app.data.model import (
        AgentAction,
        AgentActionType,
        AgentStatus,
        AgentToolCall,
        MetricProfile,
        MetricRole,
    )
    from app.data.tool_registry import build_analysis_tool_registry

    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    metric_profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=1.0,
            signals=["numeric_column"],
        )
    ]

    registry = build_analysis_tool_registry()

    state = create_agent_state(
        user_goal="Analyze the dataset.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    class EndlessMockLLM:
        def decide(self, state, observations):
            return AgentAction(
                action_type=AgentActionType.RUN_TOOL,
                tool_call=AgentToolCall(
                    tool_name="numeric_statistics",
                    target_columns=["revenue"],
                    reasoning="Continue analysis.",
                ),
                reason="Continue.",
            )

    loop = AgentLoop(
        llm=EndlessMockLLM(),
        registry=registry,
        max_steps=3,
    )

    result = loop.run(
        state=state,
        dataframe=dataframe,
    )

    assert result.status == AgentStatus.MAX_STEPS_REACHED
    assert len(result.executions) == 4
    assert result.executions[-1].tool_name == "agent_loop"
    assert result.executions[-1].success is False
    assert "Maximum reasoning steps" in result.executions[-1].error
  

def test_agent_loop_rejects_invalid_max_steps():
    from app.agent.loop import AgentLoop
    from app.data.tool_registry import build_analysis_tool_registry

    try:
        AgentLoop(
            llm=None,
            registry=build_analysis_tool_registry(),
            max_steps=0,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "max_steps" in str(exc)

def test_agent_loop_prevents_duplicate_tool_calls():
    import pandas as pd

    from app.agent.loop import AgentLoop
    from app.agent.state import create_agent_state
    from app.data.model import (
        AgentAction,
        AgentActionType,
        AgentStatus,
        AgentToolCall,
        MetricProfile,
        MetricRole,
    )
    from app.data.tool_registry import build_analysis_tool_registry

    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    metric_profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=1.0,
            signals=["numeric_column"],
        )
    ]

    registry = build_analysis_tool_registry()

    state = create_agent_state(
        user_goal="Analyze revenue.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    actions = [
        AgentAction(
            action_type=AgentActionType.RUN_TOOL,
            tool_call=AgentToolCall(
                tool_name="numeric_statistics",
                target_columns=["revenue"],
                reasoning="Analyze revenue.",
            ),
            reason="Analyze revenue.",
        ),
        AgentAction(
            action_type=AgentActionType.RUN_TOOL,
            tool_call=AgentToolCall(
                tool_name="numeric_statistics",
                target_columns=["revenue"],
                reasoning="Repeat analysis.",
            ),
            reason="Repeat analysis.",
        ),
        AgentAction(
            action_type=AgentActionType.FINISH,
            reason="Finish analysis.",
        ),
    ]

    class SequentialMockLLM:
        def __init__(self, actions):
            self.actions = actions
            self.index = 0

        def decide(self, state, observations):
            action = self.actions[self.index]
            self.index += 1
            return action

    llm = SequentialMockLLM(actions)

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        max_steps=5,
    )

    result = loop.run(
        state=state,
        dataframe=dataframe,
    )

    assert result.status == AgentStatus.COMPLETED

    assert len(result.executions) == 2

    assert result.executions[0].success is True
    assert result.executions[0].tool_name == "numeric_statistics"

    assert result.executions[1].success is False
    assert "Duplicate tool call" in result.executions[1].error

def test_agent_loop_recovers_from_invalid_tool_call():
    import pandas as pd

    from app.agent.loop import AgentLoop
    from app.agent.state import create_agent_state
    from app.data.model import (
        AgentAction,
        AgentActionType,
        AgentStatus,
        AgentToolCall,
        MetricProfile,
        MetricRole,
    )
    from app.data.tool_registry import build_analysis_tool_registry

    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    metric_profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=1.0,
            signals=["numeric_column"],
        )
    ]

    registry = build_analysis_tool_registry()

    state = create_agent_state(
        user_goal="Analyze revenue.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    actions = [
        AgentAction(
            action_type=AgentActionType.RUN_TOOL,
            tool_call=AgentToolCall(
                tool_name="invalid_tool",
                target_columns=["revenue"],
                reasoning="Try an invalid tool.",
            ),
            reason="Invalid request.",
        ),
        AgentAction(
            action_type=AgentActionType.RUN_TOOL,
            tool_call=AgentToolCall(
                tool_name="numeric_statistics",
                target_columns=["revenue"],
                reasoning="Use a valid tool.",
            ),
            reason="Recover from the previous error.",
        ),
        AgentAction(
            action_type=AgentActionType.FINISH,
            reason="Analysis complete.",
        ),
    ]

    class SequentialMockLLM:
        def __init__(self, actions):
            self.actions = actions
            self.index = 0

        def decide(self, state, observations):
            action = self.actions[self.index]
            self.index += 1
            return action

    llm = SequentialMockLLM(actions)

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        max_steps=5,
    )

    result = loop.run(
        state=state,
        dataframe=dataframe,
    )

    assert result.status == AgentStatus.COMPLETED

    assert len(result.executions) == 2

    assert result.executions[0].success is False
    assert "Unknown tool" in result.executions[0].error

    assert result.executions[1].success is True
    assert result.executions[1].tool_name == "numeric_statistics"


def test_build_agent_context_contains_core_information():
    from app.agent.context import build_agent_context
    from app.agent.state import create_agent_state
    from app.data.model import (
        MetricProfile,
        MetricRole,
    )
    from app.data.tool_registry import build_analysis_tool_registry

    registry = build_analysis_tool_registry()

    metric_profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=1.0,
            signals=["numeric_column"],
        ),
        MetricProfile(
            column_name="region",
            role=MetricRole.DIMENSION,
            confidence=0.9,
            signals=["categorical_column"],
        ),
    ]

    state = create_agent_state(
        user_goal="Understand business performance.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    context = build_agent_context(
        state=state,
        observations=[],
    )

    assert context["user_goal"] == (
        "Understand business performance."
    )

    assert context["datasets"] == ["customers"]

    assert len(context["metrics"]) == 2

    assert context["metrics"][0]["column_name"] == "revenue"
    assert context["metrics"][0]["role"] == "measure"

    assert context["metrics"][1]["column_name"] == "region"
    assert context["metrics"][1]["role"] == "dimension"

    assert len(context["available_tools"]) > 0

def test_build_agent_context_includes_previous_observations():
    from app.agent.context import build_agent_context
    from app.agent.state import create_agent_state
    from app.data.model import (
        AgentObservation,
        MetricProfile,
        MetricRole,
        NumericStatistics,
    )
    from app.data.tool_registry import build_analysis_tool_registry

    registry = build_analysis_tool_registry()

    metric_profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=1.0,
            signals=["numeric_column"],
        )
    ]

    state = create_agent_state(
        user_goal="Analyze revenue.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    statistics = NumericStatistics(
        column_name="revenue",
        count=4,
        mean=250.0,
        median=250.0,
        minimum=100.0,
        maximum=400.0,
        standard_deviation=129.099,
        variance=16666.667,
        q1=175.0,
        q3=325.0,
        skewness=0.0,
    )

    observations = [
        AgentObservation(
            tool_name="numeric_statistics",
            target_columns=["revenue"],
            success=True,
            result=statistics,
        )
    ]

    context = build_agent_context(
        state=state,
        observations=observations,
    )

    assert len(context["previous_observations"]) == 1

    observation = context["previous_observations"][0]

    assert observation["tool_name"] == (
        "numeric_statistics"
    )

    assert observation["target_columns"] == [
        "revenue"
    ]

    assert observation["success"] is True

    assert isinstance(
        observation["result"],
        dict,
    )

    assert observation["result"]["column_name"] == (
        "revenue"
    )

    assert observation["result"]["mean"] == 250.0

def test_build_agent_context_preserves_failed_observation():
    from app.agent.context import build_agent_context
    from app.agent.state import create_agent_state
    from app.data.model import (
        AgentObservation,
        MetricProfile,
        MetricRole,
    )
    from app.data.tool_registry import build_analysis_tool_registry

    registry = build_analysis_tool_registry()

    metric_profiles = [
        MetricProfile(
            column_name="revenue",
            role=MetricRole.MEASURE,
            confidence=1.0,
            signals=["numeric_column"],
        )
    ]

    state = create_agent_state(
        user_goal="Analyze revenue.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    observations = [
        AgentObservation(
            tool_name="invalid_tool",
            target_columns=["revenue"],
            success=False,
            error="Unknown tool: invalid_tool",
        )
    ]

    context = build_agent_context(
        state=state,
        observations=observations,
    )

    assert len(context["previous_observations"]) == 1

    observation = context["previous_observations"][0]

    assert observation["success"] is False

    assert observation["error"] == (
        "Unknown tool: invalid_tool"
    )

    assert observation["result"] is None

def test_parse_agent_action_accepts_agent_action():
    from app.agent.decision import parse_agent_action
    from app.data.model import (
        AgentAction,
        AgentActionType,
    )

    action = AgentAction(
        action_type=AgentActionType.FINISH,
        reason="Analysis is complete.",
    )

    result = parse_agent_action(action)

    assert result == action

def test_parse_agent_action_accepts_dictionary():
    from app.agent.decision import parse_agent_action

    response = {
        "action_type": "finish",
        "tool_call": None,
        "reason": "Analysis is complete.",
    }

    result = parse_agent_action(response)

    assert result.action_type.value == "finish"
    assert result.reason == "Analysis is complete."

def test_parse_agent_action_accepts_json():
    from app.agent.decision import parse_agent_action

    response = """
    {
        "action_type": "run_tool",
        "tool_call": {
            "tool_name": "numeric_statistics",
            "target_columns": ["revenue"],
            "reasoning": "Analyze revenue statistics."
        },
        "reason": "Revenue is an important business measure."
    }
    """

    result = parse_agent_action(response)

    assert result.action_type.value == "run_tool"

    assert result.tool_call is not None

    assert result.tool_call.tool_name == (
        "numeric_statistics"
    )

    assert result.tool_call.target_columns == [
        "revenue"
    ]

def test_parse_agent_action_rejects_invalid_json():
    from app.agent.decision import parse_agent_action

    try:
        parse_agent_action(
            '{"action_type": "finish"'
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "valid JSON" in str(exc)

def test_parse_agent_action_rejects_invalid_action():
    from app.agent.decision import parse_agent_action

    response = {
        "action_type": "invalid_action",
        "tool_call": None,
        "reason": "Invalid.",
    }

    try:
        parse_agent_action(response)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Invalid agent action" in str(exc)

def test_business_insight_model():
    insight = BusinessInsight(
        title="Revenue and cost move together",
        finding="Revenue and cost have a very strong positive relationship.",
        evidence=[
            InsightEvidence(
                execution_index=2,
                fact="Pearson correlation = 0.996675",
            )
        ],
        business_impact="Revenue growth may be accompanied by increasing costs.",
        recommendation="Investigate cost efficiency as revenue scales.",
        priority=InsightPriority.HIGH,
        caveats=[
            "The dataset contains only 5 observations."
        ],
    )

    assert insight.title == "Revenue and cost move together"
    assert insight.priority == InsightPriority.HIGH
    assert len(insight.evidence) == 1
    assert insight.evidence[0].execution_index == 2


def test_business_insight_report_model():
    report = BusinessInsightReport(
        insights=[
            BusinessInsight(
                title="Test insight",
                finding="Test finding",
                evidence=[
                    InsightEvidence(
                        execution_index=0,
                        fact="Test evidence",
                    )
                ],
                business_impact="Test impact",
                recommendation="Test recommendation",
                priority=InsightPriority.MEDIUM,
            )
        ]
    )

    assert len(report.insights) == 1
    assert report.insights[0].title == "Test insight"


def test_business_insight_rejects_negative_execution_index():
    import pytest

    with pytest.raises(ValueError):
        InsightEvidence(
            execution_index=-1,
            fact="Invalid evidence",
        )