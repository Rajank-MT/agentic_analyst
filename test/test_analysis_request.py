from app.data.analysis_request import (
    AggregationRequest,
    AnalysisRequest,
    DerivedMetricOperation,
    DerivedMetricRequest,
)


def test_analysis_request():

    request = AnalysisRequest(
        analysis_name="category_unit_economics",
        purpose="Compare category economics",
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
        rank_by="Revenue",
        top_n=5,
    )

    assert request.group_by == ["Category"]
    assert len(request.metrics) == 2
    assert len(request.derived_metrics) == 1
    assert request.rank_by == "Revenue"
    assert request.top_n == 5


def test_ratio_request():

    request = DerivedMetricRequest(
        name="Revenue_per_Unit",
        operation=DerivedMetricOperation.RATIO,
        numerator="Revenue",
        denominator="Units",
    )

    assert request.operation == DerivedMetricOperation.RATIO
    assert request.numerator == "Revenue"
    assert request.denominator == "Units"


def test_difference_request():

    request = DerivedMetricRequest(
        name="Profit",
        operation=DerivedMetricOperation.DIFFERENCE,
        left="Revenue",
        right="Cost",
    )

    assert request.operation == DerivedMetricOperation.DIFFERENCE
    assert request.left == "Revenue"
    assert request.right == "Cost"