import pandas as pd

from app.agent.loop import AgentLoop
from app.agent.providers.ollama import OllamaAgentLLM
from app.agent.state import create_agent_state
from app.data.metrics import discover_metric_role
from app.data.profiling import profile_dataset
from app.data.tool_registry import build_analysis_tool_registry
import pytest

@pytest.mark.integration
def test_ollama_agent_loop_end_to_end():
    # ---------------------------------------------------------
    # 1. Create a small business dataset
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
    # 2. Build dataset profile
    # ---------------------------------------------------------
    dataset_profile = profile_dataset(
        dataframe=dataframe,
        dataset_name="customers",
    )

    # ---------------------------------------------------------
    # 3. Infer semantic / metric roles
    # ---------------------------------------------------------
    metric_profiles = [
        discover_metric_role(column)
        for column in dataset_profile.columns
    ]

    # ---------------------------------------------------------
    # 4. Build analytical tool registry
    # ---------------------------------------------------------
    registry = build_analysis_tool_registry()

    # ---------------------------------------------------------
    # 5. Create Agent State
    # ---------------------------------------------------------
    state = create_agent_state(
        user_goal="Find important business insights in this dataset.",
        dataset_names=["customers"],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    # ---------------------------------------------------------
    # 6. Create real Ollama LLM
    # ---------------------------------------------------------
    llm = OllamaAgentLLM(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        timeout=120,
    )

    # ---------------------------------------------------------
    # 7. Create Agent Loop
    # ---------------------------------------------------------
    agent = AgentLoop(
        llm=llm,
        registry=registry,
        max_steps=5,
    )

    # ---------------------------------------------------------
    # 8. Run autonomous agent
    # ---------------------------------------------------------
    final_state = agent.run(
        state=state,
        dataframe=dataframe,
    )

    # ---------------------------------------------------------
    # 9. Diagnostic output
    # ---------------------------------------------------------
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

    for index, signal in enumerate(
        final_state.insight_signals,
        start=1,
    ):
        print(f"\nSignal #{index}")
        print(signal.model_dump())

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)

    successful_executions = [
        execution
        for execution in final_state.executions
        if execution.success
    ]

    failed_executions = [
        execution
        for execution in final_state.executions
        if not execution.success
    ]

    print(
        f"Total executions: {len(final_state.executions)}"
    )

    print(
        f"Successful executions: {len(successful_executions)}"
    )

    print(
        f"Failed executions: {len(failed_executions)}"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # 10. Basic integration assertions
    # ---------------------------------------------------------

    # The agent must record at least one execution.
    assert len(final_state.executions) > 0

    # The agent must end in a known terminal state.
    assert final_state.status.value in {
        "completed",
        "failed",
    }