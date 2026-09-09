import os

import pandas as pd
import pytest

from app.agent.insight_generator import InsightGenerator
from app.agent.loop import AgentLoop
from app.agent.providers.gemini import GeminiAgentLLM
from app.agent.state import create_agent_state
from app.data.metrics import discover_metric_role
from app.data.model import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    BusinessInsight,
    BusinessInsightReport,
    InsightEvidence,
    InsightPriority,
)
from app.data.profiling import profile_dataset
from app.data.tool_registry import build_analysis_tool_registry


class SingleCallInsightGenerator(InsightGenerator):
    """Fake insight generator so this test uses only one Gemini request."""

    def generate(
        self,
        state,
        observations: list[AgentObservation],
    ) -> BusinessInsightReport:

        successful_observations = [
            observation
            for observation in observations
            if observation.success
        ]

        return BusinessInsightReport(
            insights=[
                BusinessInsight(
                    title="Deterministic analysis completed",
                    finding=(
                        "The agent successfully selected and executed "
                        "a deterministic analytical tool."
                    ),
                    evidence=[
                        InsightEvidence(
                            execution_index=0,
                            fact=(
                                f"Tool '{successful_observations[0].tool_name}' "
                                "executed successfully."
                            ),
                        )
                    ],
                    business_impact=(
                        "The analytical pipeline successfully converted "
                        "an LLM decision into deterministic analysis."
                    ),
                    recommendation=(
                        "Continue with further analysis when additional "
                        "reasoning steps are available."
                    ),
                    priority=InsightPriority.MEDIUM,
                )
            ]
        )


class FinishAfterToolLLM:
    """
    Not used for Gemini.
    Kept here only as documentation of the desired stopping behavior.
    """


@pytest.mark.integration
def test_gemini_agent_single_decision_end_to_end():
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

    insight_generator = SingleCallInsightGenerator()

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        max_steps=1,
        insight_generator=insight_generator,
    )

    # ONE Gemini API request happens here.
    final_state = loop.run(
        state=state,
        dataframe=dataframe,
    )

    print("\n" + "=" * 70)
    print("GEMINI AGENT SINGLE-STEP END-TO-END")
    print("=" * 70)

    print("\nFinal status:")
    print(final_state.status)

    print("\nExecutions:")

    for index, execution in enumerate(final_state.executions):
        print(f"\nExecution #{index}")
        print(f"Tool: {execution.tool_name}")
        print(f"Targets: {execution.target_columns}")
        print(f"Success: {execution.success}")
        print(f"Error: {execution.error}")

    # The agent must have executed at least one deterministic tool.
    assert len(final_state.executions) >= 1

    # Because max_steps=1 and Gemini is expected to select a tool,
    # the final state should indicate that the reasoning budget was reached.
    assert final_state.status.value == "max_steps_reached"

    # The first execution should be the tool selected by Gemini.
    first_execution = final_state.executions[0]

    assert first_execution.success is True
    assert first_execution.tool_name in {
        tool.name for tool in registry.tools
    }

    # Gemini-selected columns must exist in the dataset.
    for column in first_execution.target_columns:
        assert column in dataframe.columns

    # The deterministic tool must have produced a result.
    assert first_execution.result is not None

    # Partial insight generation must have occurred locally.
    assert len(final_state.insights) == 1