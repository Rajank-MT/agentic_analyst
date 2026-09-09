import os

import pandas as pd
import pytest

from app.agent.loop import AgentLoop
from app.agent.providers.gemini import GeminiAgentLLM
from app.agent.state import create_agent_state
from app.data.metrics import discover_metric_role
from app.data.model import AgentStatus
from app.data.profiling import profile_dataset
from app.data.tool_registry import build_analysis_tool_registry


@pytest.mark.integration
def test_gemini_agent_loop_end_to_end():
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

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        max_steps=5,
    )

    final_state = loop.run(
        state=state,
        dataframe=dataframe,
    )

    print("\n" + "=" * 70)
    print("FINAL AGENT STATUS")
    print("=" * 70)
    print(final_state.status)

    print("\n" + "=" * 70)
    print("AGENT EXECUTIONS")
    print("=" * 70)

    for index, execution in enumerate(
        final_state.executions,
        start=1,
    ):
        print(f"\nExecution #{index}")
        print(execution.model_dump())

    print("\n" + "=" * 70)
    print("INSIGHT SIGNALS")
    print("=" * 70)

    for signal in final_state.insight_signals:
        print(signal.model_dump())

    assert final_state.status in {
        AgentStatus.COMPLETED,
        AgentStatus.MAX_STEPS_REACHED,
        AgentStatus.FAILED,
    }

    assert len(final_state.executions) > 0