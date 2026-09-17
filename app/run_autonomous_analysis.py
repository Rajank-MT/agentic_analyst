from pathlib import Path

import pandas as pd

from app.agent.loop import AgentLoop
#from app.agent.providers.gemini import GeminiAgentLLM
#from app.agent.providers.gemini_insight import GeminiInsightGenerator
from app.agent.state import create_agent_state
from app.agent.providers.groq import GroqAgentLLM
from app.agent.providers.groq_insight import GroqInsightGenerator
from app.data.profiling import profile_dataset
from app.data.metrics import discover_metric_role
from app.data.sampling import build_adaptive_sample
from app.data.tool_registry import build_analysis_tool_registry


DATASET_PATH = Path(
    "datasets/Indian FMCG Retail Sales  Customer Inventory (2024).csv"
)


def main():
    print("=" * 80)
    print("AGENTIC DATA ANALYST - AUTONOMOUS ANALYSIS")
    print("=" * 80)

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------
    print("\n[1] Loading dataset...")

    dataframe = pd.read_csv(DATASET_PATH)

    print(f"Rows    : {len(dataframe):,}")
    print(f"Columns : {len(dataframe.columns)}")

    # ---------------------------------------------------------------
    # 2. Profile dataset using existing pipeline
    # ---------------------------------------------------------------
    print("\n[2] Profiling dataset...")

    dataset_profile = profile_dataset(
        dataframe=dataframe,
        dataset_name=DATASET_PATH.stem,
    )

    print(
        f"Profiled: "
        f"{dataset_profile.row_count:,} rows, "
        f"{dataset_profile.column_count} columns"
    )

    # ---------------------------------------------------------------
    # 3. Discover metric roles
    # ---------------------------------------------------------------
    print("\n[3] Discovering metric roles...")

    metric_profiles = [
        discover_metric_role(column)
        for column in dataset_profile.columns
    ]

    # ---------------------------------------------------------------
    # 3b. Build adaptive dataset-understanding sample
    # ---------------------------------------------------------------
    print("\n[3b] Building adaptive dataset sample...")

    dataset_sample = build_adaptive_sample(
        dataframe=dataframe,
        random_sample_size=8,
        representative_row_count=8,
        categorical_example_count=8,
        rare_value_example_count=5,
    )

    print(
        f"Sampled {dataset_sample.sampling_metadata.sample_size} "
        "context rows plus compact value examples"
    )

    for metric in metric_profiles:
        print(
            f"  {metric.column_name:<25} "
            f"{metric.role.value:<10} "
            f"confidence={metric.confidence:.2f}"
        )

    # ---------------------------------------------------------------
    # 4. Build tool registry
    # ---------------------------------------------------------------
    print("\n[4] Building analysis tool registry...")

    registry = build_analysis_tool_registry()

    print(f"Available tools: {len(registry.tools)}")

    # ---------------------------------------------------------------
    # 5. Create agent state
    # ---------------------------------------------------------------
    state = create_agent_state(
        user_goal=(
            "Automatically analyze this FMCG retail business dataset "
            "and identify important, non-obvious, statistically "
            "supported business insights. Investigate sales, "
            "profitability, customers, channels, geography, and "
            "inventory where the data provides evidence. Do not "
            "follow a fixed analysis sequence. Decide what to "
            "investigate next based on previous findings. Use "
            "deterministic analysis results as evidence and do not "
            "invent facts or numbers."
        ),
        dataset_names=[DATASET_PATH.stem],
        metric_profiles=metric_profiles,
        available_tools=registry.tools,
    )

    # Store only the compact understanding sample in agent state.
    # Deterministic investigations continue to use the full dataframe.
    state.dataset_sample = dataset_sample.model_dump()

    # ---------------------------------------------------------------
    # 6. Initialize Gemini
    # ---------------------------------------------------------------
    print("\n[5] Initializing Gemini...")

    llm = GroqAgentLLM()

    # ---------------------------------------------------------------
    # 7. Start autonomous agent
    # ---------------------------------------------------------------
    print("\n[6] Starting autonomous investigation...")
    print("-" * 80)

    insight_generator = GroqInsightGenerator()

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        max_steps=10,
        batch_size=5,
        insight_generator=insight_generator,
    )

    final_state = loop.run(
        state=state,
        dataframe=dataframe,
    )

    # ---------------------------------------------------------------
    # 8. Display investigation
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AUTONOMOUS INVESTIGATION")
    print("=" * 80)

    for index, execution in enumerate(final_state.executions):
        print(
            f"\n[{index}] {execution.tool_name}"
        )
        print(
            f"    Columns : {execution.target_columns}"
        )
        print(
            f"    Success : {execution.success}"
        )

        if execution.error:
            print(
                f"    Error   : {execution.error}"
            )

    # ---------------------------------------------------------------
    # 9. Final status
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL STATUS")
    print("=" * 80)

    print(final_state.status.value)

    # ---------------------------------------------------------------
    # 10. Business insights
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BUSINESS INSIGHTS")
    print("=" * 80)

    if not final_state.insights:
        print("\nNo business insights were generated.")
        return

    for index, insight in enumerate(
        final_state.insights,
        start=1,
    ):
        print(f"\n{index}. {insight.title}")
        print(f"Finding        : {insight.finding}")
        print(f"Business Impact: {insight.business_impact}")
        print(f"Recommendation : {insight.recommendation}")
        print(f"Priority       : {insight.priority.value}")

        if insight.evidence:
            print("Evidence:")
            for evidence in insight.evidence:
                print(
                    f"  - Execution {evidence.execution_index}: "
                    f"{evidence.fact}"
                )

        if insight.caveats:
            print(
                f"Caveats        : "
                f"{'; '.join(insight.caveats)}"
            )


if __name__ == "__main__":
    main()