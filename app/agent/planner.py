from app.data.model import (
    AgentDecision,
    AgentState,
    AnalysisType,
    MetricRole,
)


def build_agent_decisions(state: AgentState) -> list[AgentDecision]:
    decisions: list[AgentDecision] = []

    measures = [
        profile
        for profile in state.metric_profiles
        if profile.role == MetricRole.MEASURE
    ]

    dimensions = [
        profile
        for profile in state.metric_profiles
        if profile.role == MetricRole.DIMENSION
    ]

    time_columns = [
        profile
        for profile in state.metric_profiles
        if profile.role == MetricRole.TIME
    ]

    available_tool_names = {
        tool.name
        for tool in state.available_tools
    }

    executed_tools = {
        execution.tool_name
        for execution in state.executions
        if execution.success
    }

    # ---------------------------------------------------------
    # 1. Analyze important numeric business measures
    # ---------------------------------------------------------

    if "numeric_statistics" in available_tool_names:
        for measure in measures:
            if "numeric_statistics" not in executed_tools:
                decisions.append(
                    AgentDecision(
                        decision_type="analyze_measure",
                        reason=(
                            f"'{measure.column_name}' is identified as "
                            "a business measure and requires descriptive analysis."
                        ),
                        tool_name="numeric_statistics",
                        target_columns=[measure.column_name],
                    )
                )

    # ---------------------------------------------------------
    # 2. Detect anomalies
    # ---------------------------------------------------------

    if "anomaly_detection" in available_tool_names:
        for measure in measures:
            decisions.append(
                AgentDecision(
                    decision_type="investigate_anomalies",
                    reason=(
                        f"'{measure.column_name}' is a business measure, "
                        "so unusual observations should be investigated."
                    ),
                    tool_name="anomaly_detection",
                    target_columns=[measure.column_name],
                )
            )

    # ---------------------------------------------------------
    # 3. Analyze distributions
    # ---------------------------------------------------------

    if "distribution_analysis" in available_tool_names:
        for measure in measures:
            decisions.append(
                AgentDecision(
                    decision_type="analyze_distribution",
                    reason=(
                        f"'{measure.column_name}' should be checked for "
                        "skewness, concentration, and unusual distribution behavior."
                    ),
                    tool_name="distribution_analysis",
                    target_columns=[measure.column_name],
                )
            )

    # ---------------------------------------------------------
    # 4. Compare measures
    # ---------------------------------------------------------

    if "correlation" in available_tool_names:
        for index, left_measure in enumerate(measures):
            for right_measure in measures[index + 1:]:
                decisions.append(
                    AgentDecision(
                        decision_type="investigate_relationship",
                        reason=(
                            f"'{left_measure.column_name}' and "
                            f"'{right_measure.column_name}' are both business "
                            "measures and may have a meaningful relationship."
                        ),
                        tool_name="correlation",
                        target_columns=[
                            left_measure.column_name,
                            right_measure.column_name,
                        ],
                    )
                )

    # ---------------------------------------------------------
    # 5. Segment analysis
    # ---------------------------------------------------------

    if "categorical_analysis" in available_tool_names:
        for dimension in dimensions:
            for measure in measures:
                decisions.append(
                    AgentDecision(
                        decision_type="compare_segments",
                        reason=(
                            f"'{measure.column_name}' should be compared "
                            f"across '{dimension.column_name}' segments."
                        ),
                        tool_name="categorical_analysis",
                        target_columns=[
                            dimension.column_name,
                            measure.column_name,
                        ],
                    )
                )

    # ---------------------------------------------------------
    # 6. Trend analysis
    # ---------------------------------------------------------

    if "trend_analysis" in available_tool_names:
        for time_column in time_columns:
            for measure in measures:
                decisions.append(
                    AgentDecision(
                        decision_type="investigate_trend",
                        reason=(
                            f"'{measure.column_name}' can be analyzed over "
                            f"'{time_column.column_name}' to identify growth or decline."
                        ),
                        tool_name="trend_analysis",
                        target_columns=[
                            time_column.column_name,
                            measure.column_name,
                        ],
                    )
                )

    return decisions