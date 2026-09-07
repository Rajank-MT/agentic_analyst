from typing import Any

import pandas as pd

from app.data.anomaly import detect_iqr_anomalies
from app.data.categorical import calculate_categorical_analysis
from app.data.contribution import calculate_contribution_analysis
from app.data.correlation import calculate_correlation
from app.data.distribution import calculate_distribution_analysis
from app.data.segment_relationship import calculate_segment_relationship
from app.data.statistics import calculate_numeric_statistics
from app.data.trend import calculate_trend


def execute_tool(
    tool_name: str,
    dataframe: pd.DataFrame,
    target_columns: list[str],
) -> Any:

    if tool_name == "numeric_statistics":
        if len(target_columns) != 1:
            raise ValueError(
                "numeric_statistics requires exactly one column."
            )

        column = target_columns[0]

        return calculate_numeric_statistics(
            column,
            dataframe[column],
        )

    if tool_name == "correlation":
        if len(target_columns) != 2:
            raise ValueError(
                "correlation requires exactly two columns."
            )

        left_column, right_column = target_columns

        return calculate_correlation(
            left_column,
            dataframe[left_column],
            right_column,
            dataframe[right_column],
        )

    if tool_name == "categorical_analysis":
        if len(target_columns) != 2:
            raise ValueError(
                "categorical_analysis requires two columns."
            )

        categorical_column, numeric_column = target_columns

        return calculate_categorical_analysis(
            categorical_column,
            dataframe[categorical_column],
            numeric_column,
            dataframe[numeric_column],
        )

    if tool_name == "anomaly_detection":
        if len(target_columns) != 1:
            raise ValueError(
                "anomaly_detection requires exactly one column."
            )

        column = target_columns[0]

        return detect_iqr_anomalies(
            column,
            dataframe[column],
        )

    if tool_name == "distribution_analysis":
        if len(target_columns) != 1:
            raise ValueError(
                "distribution_analysis requires exactly one column."
            )

        column = target_columns[0]

        return calculate_distribution_analysis(
            column,
            dataframe[column],
        )

    if tool_name == "trend_analysis":
        if len(target_columns) != 2:
            raise ValueError(
                "trend_analysis requires time and measure columns."
            )

        time_column, measure_column = target_columns

        return calculate_trend(
            time_column,
            dataframe[time_column],
            measure_column,
            dataframe[measure_column],
        )

    if tool_name == "contribution_analysis":
        if len(target_columns) != 2:
            raise ValueError(
                "contribution_analysis requires two columns."
            )

        categorical_column, measure_column = target_columns

        return calculate_contribution_analysis(
            categorical_column,
            dataframe[categorical_column],
            measure_column,
            dataframe[measure_column],
        )

    if tool_name == "segment_relationship":
        if len(target_columns) != 2:
            raise ValueError(
                "segment_relationship requires two columns."
            )

        categorical_column, measure_column = target_columns

        return calculate_segment_relationship(
            categorical_column,
            dataframe[categorical_column],
            measure_column,
            dataframe[measure_column],
        )

    raise ValueError(f"Unknown analysis tool: {tool_name}")