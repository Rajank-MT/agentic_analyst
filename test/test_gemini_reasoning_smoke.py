import os

import pandas as pd
import pytest

from app.agent.providers.gemini import GeminiAgentLLM
from app.agent.state import create_agent_state
from app.data.metrics import discover_metric_role
from app.data.model import AgentActionType
from app.data.profiling import profile_dataset
from app.data.tool_registry import build_analysis_tool_registry


@pytest.mark.integration
def test_gemini_reasoning_single_decision():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured.")

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "revenue": [1000, 1500, 700, 2200, 1800],
            "cost": [600, 900, 500, 1200, 1000],
            "orders": [5, 7, 3, 10, 8],
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
        user_goal="Find important business insights in this dataset.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    llm = GeminiAgentLLM(
        model="gemini-2.5-flash",
    )

    # Exactly ONE Gemini API request.
    action = llm.decide(
        state=state,
        observations=[],
    )

    print("\n" + "=" * 70)
    print("GEMINI SINGLE DECISION")
    print("=" * 70)
    print(action.model_dump())

    assert action is not None
    assert action.action_type in {
        AgentActionType.RUN_TOOL,
        AgentActionType.FINISH,
    }

    if action.action_type == AgentActionType.RUN_TOOL:
        assert action.tool_call is not None

        available_tool_names = {
            tool.name
            for tool in registry.tools
        }

        available_columns = set(dataframe.columns)

        assert action.tool_call.tool_name in available_tool_names

        for column in action.tool_call.target_columns:
            assert column in available_columns

        assert action.tool_call.reasoning.strip()