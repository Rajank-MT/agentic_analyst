from __future__ import annotations

import pandas as pd

from app.data.analysis_engine import (
    aggregate_by_group,
    calculate_difference,
    calculate_percentage_of_total,
    calculate_ratio,
    rank_dataframe,
    top_n,
)
from app.data.analysis_request import (
    AnalysisRequest,
    DerivedMetricOperation,
)


class AnalysisExecutionError(ValueError):
    """Raised when a composed analysis cannot be executed."""


class AnalysisExecutor:
    """Executes validated analysis requests deterministically."""

    def execute(
        self,
        request: AnalysisRequest,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        result = dataframe.copy()

        # Group + aggregation
        if request.group_by and request.metrics:

            metrics = {
                metric.column: metric.aggregation
                for metric in request.metrics
            }

            result = aggregate_by_group(
                result,
                group_by=request.group_by,
                metrics=metrics,
            )

        # Derived metrics
        for metric in request.derived_metrics:

            if metric.operation == DerivedMetricOperation.RATIO:

                if not metric.numerator or not metric.denominator:
                    raise AnalysisExecutionError(
                        f"{metric.name}: ratio requires numerator and denominator."
                    )

                result = calculate_ratio(
                    result,
                    numerator=metric.numerator,
                    denominator=metric.denominator,
                    output_name=metric.name,
                )

            elif metric.operation == DerivedMetricOperation.DIFFERENCE:

                if not metric.left or not metric.right:
                    raise AnalysisExecutionError(
                        f"{metric.name}: difference requires left and right."
                    )

                result = calculate_difference(
                    result,
                    left=metric.left,
                    right=metric.right,
                    output_name=metric.name,
                )

            elif (
                metric.operation
                == DerivedMetricOperation.PERCENTAGE_OF_TOTAL
            ):

                if not metric.source_column:
                    raise AnalysisExecutionError(
                        f"{metric.name}: source_column is required."
                    )

                result = calculate_percentage_of_total(
                    result,
                    column=metric.source_column,
                    output_name=metric.name,
                )

        # Ranking
        if request.rank_by:

            result = rank_dataframe(
                result,
                column=request.rank_by,
                ascending=request.rank_ascending,
            )

        # Top / bottom N
        if request.top_n is not None:

            if not request.rank_by:
                raise AnalysisExecutionError(
                    "top_n requires rank_by."
                )

            result = top_n(
                result,
                column=request.rank_by,
                n=request.top_n,
                ascending=request.rank_ascending,
            )

        return result.reset_index(drop=True)