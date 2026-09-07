import os

import pandas as pd
import pytest

from app.agent.loop import AgentLoop
from app.agent.providers.gemini import GeminiAgentLLM
from app.agent.providers.gemini_insight import GeminiInsightGenerator
from app.agent.state import create_agent_state
from app.data.metrics import discover_metric_role
from app.data.model import AgentStatus
from app.data.profiling import profile_dataset
from app.data.tool_registry import build_analysis_tool_registry


@pytest.mark.integration
def test_gemini_agent_loop_end_to_end():

    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured.")

    # ---------------------------------------------------------
    # 1. Create test dataset
    # ---------------------------------------------------------

    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "revenue": [1000, 1500, 700, 2200, 1800],
            "cost": [600, 900, 500, 1200, 1000],
            "orders": [5, 7, 3, 10, 8],
        }
    )

    # ---------------------------------------------------------
    # 2. Profile dataset
    # ---------------------------------------------------------

    dataset_profile = profile_dataset(
        dataframe=dataframe,
        dataset_name="customers",
    )

    metric_profiles = [
        discover_metric_role(column)
        for column in dataset_profile.columns
    ]

    # ---------------------------------------------------------
    # 3. Build tool registry
    # ---------------------------------------------------------

    registry = build_analysis_tool_registry()

    # ---------------------------------------------------------
    # 4. Create agent state
    # ---------------------------------------------------------

    state = create_agent_state(
        user_goal="Find important business insights in this dataset.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    # ---------------------------------------------------------
    # 5. Create Gemini reasoning engine
    # ---------------------------------------------------------

    llm = GeminiAgentLLM(
        model="gemini-2.5-flash"
    )

    # ---------------------------------------------------------
    # 6. Create Gemini insight generator
    # ---------------------------------------------------------

    insight_generator = GeminiInsightGenerator(
        model="gemini-2.5-flash"
    )

    # ---------------------------------------------------------
    # 7. Create autonomous agent loop
    # ---------------------------------------------------------

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        max_steps=8,
        insight_generator=insight_generator,
    )

    # ---------------------------------------------------------
    # 8. Run autonomous analysis
    # ---------------------------------------------------------

    final_state = loop.run(
        state=state,
        dataframe=dataframe,
    )

    # ---------------------------------------------------------
    # 9. Print agent execution history
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("AGENT EXECUTION HISTORY")
    print("=" * 70)

    for index, execution in enumerate(final_state.executions):

        print(f"\nExecution #{index}")
        print("Tool:", execution.tool_name)
        print("Targets:", execution.target_columns)
        print("Success:", execution.success)

        if execution.result is not None:
            print("Result:", execution.result)

        if execution.error is not None:
            print("Error:", execution.error)

    # ---------------------------------------------------------
    # 10. Print final business insights
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL BUSINESS INSIGHTS")
    print("=" * 70)

    for index, insight in enumerate(final_state.insights):

        print(f"\nInsight #{index + 1}")

        print("Title:", insight.title)
        print("Finding:", insight.finding)
        print("Business Impact:", insight.business_impact)
        print("Recommendation:", insight.recommendation)
        print("Priority:", insight.priority)

        print("Evidence:")

        for evidence in insight.evidence:
            print(
                f"  Execution #{evidence.execution_index}: "
                f"{evidence.fact}"
            )

        print("Caveats:", insight.caveats)

    # ---------------------------------------------------------
    # 11. Validate final agent state
    # ---------------------------------------------------------

    assert final_state.status == AgentStatus.COMPLETED

    assert len(final_state.executions) > 0

    assert len(final_state.insights) > 0

    # ---------------------------------------------------------
    # 12. Validate business insights
    # ---------------------------------------------------------

    for insight in final_state.insights:

        assert insight.title
        assert insight.finding
        assert insight.business_impact
        assert insight.recommendation
        assert insight.evidence

        for evidence in insight.evidence:

            assert evidence.execution_index >= 0

            assert (
                evidence.execution_index
                < len(final_state.executions)
            )