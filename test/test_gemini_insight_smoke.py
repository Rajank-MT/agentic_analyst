import os

import pytest

from app.agent.providers.gemini_insight import GeminiInsightGenerator
from app.data.model import (
    AgentObservation,
    CorrelationProfile,
    NumericStatistics,
    AgentState,
    AgentStatus,
)


@pytest.mark.integration
def test_gemini_insight_generation_single_call():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured.")

    observations = [
        AgentObservation(
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
                standard_deviation=624.5,
                variance=389375.0,
                q1=1000.0,
                q3=1800.0,
                skewness=0.0,
            ),
        ),
        AgentObservation(
            tool_name="numeric_statistics",
            target_columns=["cost"],
            success=True,
            result=NumericStatistics(
                column_name="cost",
                count=5,
                mean=840.0,
                median=900.0,
                minimum=500.0,
                maximum=1200.0,
                standard_deviation=290.7,
                variance=84500.0,
                q1=600.0,
                q3=1000.0,
                skewness=0.0,
            ),
        ),
        AgentObservation(
            tool_name="correlation",
            target_columns=["revenue", "cost"],
            success=True,
            result=CorrelationProfile(
                left_column="revenue",
                right_column="cost",
                coefficient=0.996675,
                method="pearson",
                strength="very_strong",
            ),
        ),
    ]

    state = AgentState(
        user_goal="Find important business insights in the customer dataset.",
        dataset_names=["customers"],
        status=AgentStatus.ANALYZING,
    )

    generator = GeminiInsightGenerator(
        model="gemini-2.5-flash",
    )

    # Exactly ONE Gemini API request.
    report = generator.generate(
        state=state,
        observations=observations,
    )

    print("\n" + "=" * 70)
    print("GEMINI SINGLE INSIGHT GENERATION")
    print("=" * 70)

    for index, insight in enumerate(report.insights):
        print(f"\nInsight {index + 1}")
        print(f"Title: {insight.title}")
        print(f"Finding: {insight.finding}")
        print(f"Business Impact: {insight.business_impact}")
        print(f"Recommendation: {insight.recommendation}")
        print(f"Priority: {insight.priority}")
        print(f"Caveats: {insight.caveats}")

        for evidence in insight.evidence:
            print(
                f"Evidence #{evidence.execution_index}: "
                f"{evidence.fact}"
            )

    assert report is not None
    assert len(report.insights) > 0

    for insight in report.insights:
        assert insight.title.strip()
        assert insight.finding.strip()
        assert insight.business_impact.strip()
        assert insight.recommendation.strip()

        assert len(insight.evidence) > 0

        for evidence in insight.evidence:
            assert 0 <= evidence.execution_index < len(observations)
            assert evidence.fact.strip()