from app.data.model import (
    AnalysisPlan,
    AnalysisPlanItem,
    AnalysisType,
    MetricProfile,
    MetricRole,
)


def build_analysis_plan(
    metric_profiles: list[MetricProfile],
) -> AnalysisPlan:
    """
    Build candidate business analyses from discovered
    metric roles.
    """

    analyses: list[AnalysisPlanItem] = []

    measures = [
        profile
        for profile in metric_profiles
        if profile.role == MetricRole.MEASURE
    ]

    dimensions = [
        profile
        for profile in metric_profiles
        if profile.role == MetricRole.DIMENSION
    ]

    time_columns = [
        profile
        for profile in metric_profiles
        if profile.role == MetricRole.TIME
    ]

    # --------------------------------------------------
    # Individual numeric measure analysis
    # --------------------------------------------------

    for measure in measures:

        analyses.append(
            AnalysisPlanItem(
                analysis_type=AnalysisType.NUMERIC_STATISTICS,
                target_columns=[measure.column_name],
                reason=(
                    "Numeric business measure requires "
                    "descriptive statistical analysis."
                ),
            )
        )

        analyses.append(
            AnalysisPlanItem(
                analysis_type=AnalysisType.ANOMALY_DETECTION,
                target_columns=[measure.column_name],
                reason=(
                    "Numeric business measure should be "
                    "checked for unusual observations."
                ),
            )
        )

        analyses.append(
            AnalysisPlanItem(
                analysis_type=AnalysisType.DISTRIBUTION_ANALYSIS,
                target_columns=[measure.column_name],
                reason=(
                    "Distribution analysis can reveal "
                    "concentration and unusual data behavior."
                ),
            )
        )

    # --------------------------------------------------
    # Measure-to-measure relationships
    # --------------------------------------------------

    for index, left_measure in enumerate(measures):

        for right_measure in measures[index + 1:]:

            analyses.append(
                AnalysisPlanItem(
                    analysis_type=AnalysisType.CORRELATION,
                    target_columns=[
                        left_measure.column_name,
                        right_measure.column_name,
                    ],
                    reason=(
                        "Both columns are numeric business "
                        "measures and may have a meaningful "
                        "relationship."
                    ),
                )
            )

    # --------------------------------------------------
    # Measure-by-dimension analysis
    # --------------------------------------------------

    for measure in measures:

        for dimension in dimensions:

            analyses.append(
                AnalysisPlanItem(
                    analysis_type=AnalysisType.CATEGORICAL_ANALYSIS,
                    target_columns=[
                        dimension.column_name,
                        measure.column_name,
                    ],
                    reason=(
                        "Comparing a business measure across "
                        "categories can reveal segment-level "
                        "performance differences."
                    ),
                )
            )

    # --------------------------------------------------
    # Time-based analysis
    # --------------------------------------------------

    for time_column in time_columns:

        for measure in measures:

            analyses.append(
                AnalysisPlanItem(
                    analysis_type=AnalysisType.TREND_ANALYSIS,
                    target_columns=[
                        time_column.column_name,
                        measure.column_name,
                    ],
                    reason=(
                        "A time dimension and business measure "
                        "enable trend and growth analysis."
                    ),
                )
            )

    return AnalysisPlan(analyses=analyses)