from app.data.model import (
    AgentState,
    AgentStatus,
    AnalysisPlan,
    AnalysisTool,
    InsightSignal,
    MetricProfile,
    ToolExecution,
)


def create_agent_state(
    user_goal: str,
    dataset_names: list[str],
    metric_profiles: list[MetricProfile],
    available_tools: list[AnalysisTool],
) -> AgentState:

    return AgentState(
        user_goal=user_goal,
        dataset_names=dataset_names,
        metric_profiles=metric_profiles,
        available_tools=available_tools,
        status=AgentStatus.INITIALIZED,
    )


def update_agent_status(
    state: AgentState,
    status: AgentStatus,
) -> AgentState:

    state.status = status
    return state


def attach_analysis_plan(
    state: AgentState,
    analysis_plan: AnalysisPlan,
) -> AgentState:

    state.analysis_plan = analysis_plan
    state.status = AgentStatus.PLANNING
    return state


def add_tool_execution(
    state: AgentState,
    execution: ToolExecution,
) -> AgentState:

    state.executions.append(execution)
    return state


def add_insight_signal(
    state: AgentState,
    signal: InsightSignal,
) -> AgentState:

    state.insight_signals.append(signal)
    return state