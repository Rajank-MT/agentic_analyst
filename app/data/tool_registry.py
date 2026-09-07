from app.data.model import (
    AnalysisTool,
    AnalysisToolRegistry,
    AnalysisType,
    MetricRole,
)


def build_analysis_tool_registry() -> AnalysisToolRegistry:
    tools = [
        AnalysisTool(
            name="numeric_statistics",
            description="Calculate descriptive statistics for a numeric business measure.",
            analysis_type=AnalysisType.NUMERIC_STATISTICS,
            required_roles=[MetricRole.MEASURE],
            min_targets=1,
            max_targets=1,
        ),
        AnalysisTool(
            name="correlation",
            description="Measure the statistical relationship between two numeric measures.",
            analysis_type=AnalysisType.CORRELATION,
            required_roles=[MetricRole.MEASURE],
            min_targets=2,
            max_targets=2,
        ),
        AnalysisTool(
            name="categorical_analysis",
            description="Compare a numeric measure across categorical segments.",
            analysis_type=AnalysisType.CATEGORICAL_ANALYSIS,
            required_roles=[
                MetricRole.DIMENSION,
                MetricRole.MEASURE,
            ],
            min_targets=2,
            max_targets=2,
        ),
        AnalysisTool(
            name="anomaly_detection",
            description="Detect statistically unusual values in a numeric measure.",
            analysis_type=AnalysisType.ANOMALY_DETECTION,
            required_roles=[MetricRole.MEASURE],
            min_targets=1,
            max_targets=1,
        ),
        AnalysisTool(
            name="distribution_analysis",
            description="Analyze the distribution and shape of a numeric measure.",
            analysis_type=AnalysisType.DISTRIBUTION_ANALYSIS,
            required_roles=[MetricRole.MEASURE],
            min_targets=1,
            max_targets=1,
        ),
        AnalysisTool(
            name="trend_analysis",
            description="Analyze changes in a business measure over time.",
            analysis_type=AnalysisType.TREND_ANALYSIS,
            required_roles=[
                MetricRole.TIME,
                MetricRole.MEASURE,
            ],
            min_targets=2,
            max_targets=2,
        ),
        AnalysisTool(
            name="contribution_analysis",
            description="Measure how much each business segment contributes to a total measure.",
            analysis_type=AnalysisType.CATEGORICAL_ANALYSIS,
            required_roles=[
                MetricRole.DIMENSION,
                MetricRole.MEASURE,
            ],
            min_targets=2,
            max_targets=2,
        ),
        AnalysisTool(
            name="segment_relationship",
            description="Compare business measures across segments against the overall population.",
            analysis_type=AnalysisType.CATEGORICAL_ANALYSIS,
            required_roles=[
                MetricRole.DIMENSION,
                MetricRole.MEASURE,
            ],
            min_targets=2,
            max_targets=2,
        ),
        AnalysisTool(
            name="insight_signals",
            description="Identify potentially important analytical signals from deterministic results.",
            analysis_type=AnalysisType.NUMERIC_STATISTICS,
            required_roles=[],
            min_targets=0,
            max_targets=0,
        ),
    ]

    return AnalysisToolRegistry(tools=tools)