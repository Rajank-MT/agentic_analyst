from app.agent.context import build_agent_context
from app.agent.state import create_agent_state
from app.data.metrics import discover_metric_role
from app.data.model import (
    AgentObservation,
    NumericStatistics,
)
from app.data.profiling import profile_dataset
from app.data.tool_registry import build_analysis_tool_registry

import pandas as pd


def test_agent_context_contains_investigation_history_and_results():
    dataframe = pd.DataFrame(
        {
            "revenue": [1000, 1500, 700, 2200, 1800],
        }
    )

    dataset_profile = profile_dataset(
        dataframe=dataframe,
        dataset_name="customers",
    )

    metric_profiles = [
        discover_metric_role(column)
        for column in dataset_profile.columns
    ]

    registry = build_analysis_tool_registry()

    state = create_agent_state(
        user_goal="Find important business insights.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    observation = AgentObservation(
        tool_name="numeric_statistics",
        target_columns=["revenue"],
        success=True,
        result=NumericStatistics(
            column_name="revenue",
            count=5,
            mean=1440.0,
            median=1500.0,
            minimum=700.0,
            maximum=2200.0,
            standard_deviation=602.4948,
            variance=363000.0,
            q1=1000.0,
            q3=1800.0,
            skewness=-0.008687,
        ),
    )

    context = build_agent_context(
        state=state,
        observations=[observation],
    )

    assert "investigation_history" in context
    assert "observation_results" in context

    assert len(context["investigation_history"]) == 1
    assert len(context["observation_results"]) == 1

    history = context["investigation_history"][0]

    assert history["execution_index"] == 0
    assert history["tool_name"] == "numeric_statistics"
    assert history["target_columns"] == ["revenue"]
    assert history["success"] is True

    result = context["observation_results"][0]

    assert result["execution_index"] == 0
    assert result["tool_name"] == "numeric_statistics"
    assert result["status"] == "success"

    assert result["result"]["column_name"] == "revenue"
    assert result["result"]["mean"] == 1440.0

def test_agent_context_preserves_failed_observation():
    dataframe = pd.DataFrame(
        {
            "revenue": [1000, 1500, 700],
        }
    )

    dataset_profile = profile_dataset(
        dataframe=dataframe,
        dataset_name="customers",
    )

    metric_profiles = [
        discover_metric_role(column)
        for column in dataset_profile.columns
    ]

    registry = build_analysis_tool_registry()

    state = create_agent_state(
        user_goal="Find important business insights.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    observation = AgentObservation(
        tool_name="correlation",
        target_columns=["revenue", "missing_column"],
        success=False,
        error="Columns not found.",
    )

    context = build_agent_context(
        state=state,
        observations=[observation],
    )

    assert len(context["investigation_history"]) == 1
    assert len(context["observation_results"]) == 1

    result = context["observation_results"][0]

    assert result["status"] == "failed"
    assert result["error"] == "Columns not found."