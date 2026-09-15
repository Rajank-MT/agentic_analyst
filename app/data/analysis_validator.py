from __future__ import annotations

import pandas as pd

from app.data.analysis_engine import SUPPORTED_AGGREGATIONS
from app.data.analysis_request import (
    AnalysisRequest,
    DerivedMetricOperation,
)


class AnalysisRequestValidator:
    """Validates LLM-generated composed analysis requests."""

    def validate(
        self,
        request: AnalysisRequest,
        dataframe: pd.DataFrame,
    ) -> list[str]:

        errors: list[str] = []

        columns = set(dataframe.columns)

        # Group-by columns
        for column in request.group_by:
            if column not in columns:
                errors.append(
                    f"Group-by column not found: {column}"
                )

        # Aggregations
        for metric in request.metrics:
            if metric.column not in columns:
                errors.append(
                    f"Metric column not found: {metric.column}"
                )

            if metric.aggregation not in SUPPORTED_AGGREGATIONS:
                errors.append(
                    f"Unsupported aggregation: {metric.aggregation}"
                )

        # ---------------------------------------------------------
        # Derived metrics
        # ---------------------------------------------------------
        #
        # A derived metric may use:
        #
        #   1. Original dataframe columns
        #   2. Previously defined derived metrics
        #
        # Example:
        #
        # Revenue_per_Unit = Revenue / Units
        # Cost_per_Unit    = Cost / Units
        # Margin_per_Unit  = Revenue_per_Unit - Cost_per_Unit
        #
        # But a metric cannot reference a derived metric that
        # appears later in the list.
        # ---------------------------------------------------------

        available_columns = set(columns)

        for metric in request.derived_metrics:

            references: list[str] = []

            if metric.operation == DerivedMetricOperation.RATIO:

                if not metric.numerator:
                    errors.append(
                        f"{metric.name}: numerator is required."
                    )

                if not metric.denominator:
                    errors.append(
                        f"{metric.name}: denominator is required."
                    )

                references = [
                    metric.numerator,
                    metric.denominator,
                ]

            elif metric.operation == DerivedMetricOperation.DIFFERENCE:

                if not metric.left:
                    errors.append(
                        f"{metric.name}: left column is required."
                    )

                if not metric.right:
                    errors.append(
                        f"{metric.name}: right column is required."
                    )

                references = [
                    metric.left,
                    metric.right,
                ]

            elif (
                metric.operation
                == DerivedMetricOperation.PERCENTAGE_OF_TOTAL
            ):

                if not metric.source_column:
                    errors.append(
                        f"{metric.name}: source_column is required."
                    )

                references = [
                    metric.source_column,
                ]

            # Validate against original + previously created
            # derived columns.
            for column in references:

                if (
                    column
                    and column not in available_columns
                ):
                    errors.append(
                        f"{metric.name}: column not found: "
                        f"{column}"
                    )

            # Make this derived metric available to the
            # following derived metrics.
            available_columns.add(metric.name)

        # ---------------------------------------------------------
        # Rank
        # ---------------------------------------------------------

        if request.rank_by:

            valid_rank_columns = (
                columns
                | {
                    metric.name
                    for metric in request.derived_metrics
                }
            )

            if request.rank_by not in valid_rank_columns:
                errors.append(
                    f"Rank column not found: {request.rank_by}"
                )

        # top_n requires ranking
        if request.top_n is not None and not request.rank_by:
            errors.append(
                "top_n requires rank_by."
            )

        # At least one operation
        if not (
            request.group_by
            or request.metrics
            or request.derived_metrics
            or request.rank_by
            or request.top_n
        ):
            errors.append(
                "Analysis request contains no operations."
            )

        return errors