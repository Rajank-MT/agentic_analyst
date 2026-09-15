import pandas as pd

from app.data.analysis_request import (
    AggregationRequest,
    AnalysisRequest,
    DerivedMetricOperation,
    DerivedMetricRequest,
)

from app.data.analysis_validator import AnalysisRequestValidator


def get_dataframe():
    return pd.DataFrame(
        {
            "Category": ["A", "B"],
            "Revenue": [100, 200],
            "Cost": [60, 120],
            "Units": [10, 20],
        }
    )


def test_valid_request():

    request = AnalysisRequest(
        analysis_name="category_analysis",
        purpose="Compare categories",
        group_by=["Category"],
        metrics=[
            AggregationRequest(
                column="Revenue",
                aggregation="sum",
            )
        ],
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert errors == []


def test_missing_group_column():

    request = AnalysisRequest(
        analysis_name="test",
        purpose="test",
        group_by=["Unknown"],
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert any("Group-by column not found" in error for error in errors)


def test_missing_metric_column():

    request = AnalysisRequest(
        analysis_name="test",
        purpose="test",
        metrics=[
            AggregationRequest(
                column="Unknown",
                aggregation="sum",
            )
        ],
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert any("Metric column not found" in error for error in errors)


def test_invalid_aggregation():

    request = AnalysisRequest(
        analysis_name="test",
        purpose="test",
        metrics=[
            AggregationRequest(
                column="Revenue",
                aggregation="invalid",
            )
        ],
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert any("Unsupported aggregation" in error for error in errors)


def test_invalid_ratio_column():

    request = AnalysisRequest(
        analysis_name="test",
        purpose="test",
        derived_metrics=[
            DerivedMetricRequest(
                name="Revenue_per_Unit",
                operation=DerivedMetricOperation.RATIO,
                numerator="Revenue",
                denominator="Unknown",
            )
        ],
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert any("column not found" in error for error in errors)


def test_invalid_difference():

    request = AnalysisRequest(
        analysis_name="test",
        purpose="test",
        derived_metrics=[
            DerivedMetricRequest(
                name="Profit",
                operation=DerivedMetricOperation.DIFFERENCE,
                left="Revenue",
                right="Unknown",
            )
        ],
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert any("column not found" in error for error in errors)


def test_invalid_rank_column():

    request = AnalysisRequest(
        analysis_name="test",
        purpose="test",
        rank_by="Unknown",
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert any("Rank column not found" in error for error in errors)


def test_top_n_requires_rank():

    request = AnalysisRequest(
        analysis_name="test",
        purpose="test",
        top_n=5,
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert "top_n requires rank_by." in errors


def test_empty_request():

    request = AnalysisRequest(
        analysis_name="test",
        purpose="test",
    )

    errors = AnalysisRequestValidator().validate(
        request,
        get_dataframe(),
    )

    assert "Analysis request contains no operations." in errors