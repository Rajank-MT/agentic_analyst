import pandas as pd
import pytest

from app.data.analysis_engine import (
    AnalysisEngineError,
    aggregate_by_group,
    calculate_difference,
    calculate_percentage_of_total,
    calculate_ratio,
    rank_dataframe,
    top_n,
)


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "Category": ["A", "A", "B", "B"],
            "Revenue": [100, 200, 300, 400],
            "Cost": [60, 120, 180, 250],
            "Units": [10, 20, 30, 40],
        }
    )


def test_aggregate_by_group(dataframe):
    result = aggregate_by_group(
        dataframe,
        group_by=["Category"],
        metrics={
            "Revenue": "sum",
            "Cost": "sum",
        },
    )

    category_a = result[result["Category"] == "A"].iloc[0]
    category_b = result[result["Category"] == "B"].iloc[0]

    assert category_a["Revenue"] == 300
    assert category_a["Cost"] == 180

    assert category_b["Revenue"] == 700
    assert category_b["Cost"] == 430


def test_calculate_ratio(dataframe):
    result = calculate_ratio(
        dataframe,
        numerator="Revenue",
        denominator="Units",
        output_name="Revenue_per_Unit",
    )

    assert result["Revenue_per_Unit"].tolist() == [
        10.0,
        10.0,
        10.0,
        10.0,
    ]


def test_calculate_difference(dataframe):
    result = calculate_difference(
        dataframe,
        left="Revenue",
        right="Cost",
        output_name="Margin_Calculated",
    )

    assert result["Margin_Calculated"].tolist() == [
        40,
        80,
        120,
        150,
    ]


def test_percentage_of_total(dataframe):
    result = calculate_percentage_of_total(
        dataframe,
        column="Revenue",
        output_name="Revenue_Share",
    )

    assert round(result["Revenue_Share"].sum(), 6) == 100.0


def test_rank_dataframe(dataframe):
    result = rank_dataframe(
        dataframe,
        column="Revenue",
    )

    assert result.iloc[0]["Revenue"] == 400
    assert result.iloc[-1]["Revenue"] == 100


def test_top_n(dataframe):
    result = top_n(
        dataframe,
        column="Revenue",
        n=2,
    )

    assert result["Revenue"].tolist() == [400, 300]


def test_bottom_n(dataframe):
    result = top_n(
        dataframe,
        column="Revenue",
        n=2,
        ascending=True,
    )

    assert result["Revenue"].tolist() == [100, 200]


def test_missing_column_is_rejected(dataframe):
    with pytest.raises(AnalysisEngineError):
        aggregate_by_group(
            dataframe,
            group_by=["Unknown"],
            metrics={"Revenue": "sum"},
        )


def test_invalid_aggregation_is_rejected(dataframe):
    with pytest.raises(AnalysisEngineError):
        aggregate_by_group(
            dataframe,
            group_by=["Category"],
            metrics={"Revenue": "invalid"},
        )


def test_invalid_top_n_is_rejected(dataframe):
    with pytest.raises(AnalysisEngineError):
        top_n(
            dataframe,
            column="Revenue",
            n=0,
        )