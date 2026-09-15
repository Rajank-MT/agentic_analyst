import pandas as pd

from app.data.analysis_executor import AnalysisExecutor
from app.data.analysis_request import (
    AggregationRequest,
    AnalysisRequest,
    DerivedMetricOperation,
    DerivedMetricRequest,
)


def get_dataframe():
    return pd.DataFrame(
        {
            "Category": ["A", "A", "B", "B"],
            "Revenue": [100, 200, 300, 400],
            "Cost": [60, 120, 180, 250],
            "Units": [10, 20, 30, 40],
        }
    )


def test_group_and_aggregate():

    request = AnalysisRequest(
        analysis_name="category_revenue",
        purpose="Compare category revenue",
        group_by=["Category"],
        metrics=[
            AggregationRequest(
                column="Revenue",
                aggregation="sum",
            )
        ],
    )

    result = AnalysisExecutor().execute(
        request,
        get_dataframe(),
    )

    values = dict(
        zip(
            result["Category"],
            result["Revenue"],
        )
    )

    assert values["A"] == 300
    assert values["B"] == 700


def test_derived_ratio():

    request = AnalysisRequest(
        analysis_name="revenue_per_unit",
        purpose="Calculate revenue per unit",
        group_by=["Category"],
        metrics=[
            AggregationRequest(
                column="Revenue",
                aggregation="sum",
            ),
            AggregationRequest(
                column="Units",
                aggregation="sum",
            ),
        ],
        derived_metrics=[
            DerivedMetricRequest(
                name="Revenue_per_Unit",
                operation=DerivedMetricOperation.RATIO,
                numerator="Revenue",
                denominator="Units",
            )
        ],
    )

    result = AnalysisExecutor().execute(
        request,
        get_dataframe(),
    )

    assert result["Revenue_per_Unit"].tolist() == [
        10.0,
        10.0,
    ]


def test_derived_difference():

    request = AnalysisRequest(
        analysis_name="margin",
        purpose="Calculate margin",
        group_by=["Category"],
        metrics=[
            AggregationRequest(
                column="Revenue",
                aggregation="sum",
            ),
            AggregationRequest(
                column="Cost",
                aggregation="sum",
            ),
        ],
        derived_metrics=[
            DerivedMetricRequest(
                name="Margin",
                operation=DerivedMetricOperation.DIFFERENCE,
                left="Revenue",
                right="Cost",
            )
        ],
    )

    result = AnalysisExecutor().execute(
        request,
        get_dataframe(),
    )

    assert result["Margin"].tolist() == [
        120,
        270,
    ]


def test_rank_and_top_n():

    request = AnalysisRequest(
        analysis_name="top_categories",
        purpose="Find top categories",
        group_by=["Category"],
        metrics=[
            AggregationRequest(
                column="Revenue",
                aggregation="sum",
            )
        ],
        rank_by="Revenue",
        top_n=1,
    )

    result = AnalysisExecutor().execute(
        request,
        get_dataframe(),
    )

    assert len(result) == 1
    assert result.iloc[0]["Category"] == "B"
    assert result.iloc[0]["Revenue"] == 700


def test_percentage_of_total():

    request = AnalysisRequest(
        analysis_name="revenue_share",
        purpose="Calculate revenue contribution",
        group_by=["Category"],
        metrics=[
            AggregationRequest(
                column="Revenue",
                aggregation="sum",
            )
        ],
        derived_metrics=[
            DerivedMetricRequest(
                name="Revenue_Share",
                operation=DerivedMetricOperation.PERCENTAGE_OF_TOTAL,
                source_column="Revenue",
            )
        ],
    )

    result = AnalysisExecutor().execute(
        request,
        get_dataframe(),
    )

    assert round(result["Revenue_Share"].sum(), 6) == 100.0