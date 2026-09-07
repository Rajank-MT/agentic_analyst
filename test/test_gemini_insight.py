import os

import pytest

from app.agent.providers.gemini_insight import GeminiInsightGenerator

from app.agent.state import create_agent_state
from app.data.model import (
    AgentObservation,
    BusinessInsight,
    BusinessInsightReport,
    InsightEvidence,
    InsightPriority,
)

def test_gemini_insight_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiInsightGenerator()


def test_gemini_insight_initializes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiInsightGenerator()

    assert generator.model == "gemini-2.5-flash"


def test_gemini_insight_custom_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiInsightGenerator(
        model="gemini-2.5-flash"
    )

    assert generator.model == "gemini-2.5-flash"

def test_gemini_insight_generator_rejects_invalid_report(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    generator = GeminiInsightGenerator()

    invalid_report = BusinessInsightReport(
        insights=[
            BusinessInsight(
                title="Invalid insight",
                finding="Unsupported finding",
                evidence=[
                    InsightEvidence(
                        execution_index=99,
                        fact="Invalid evidence",
                    )
                ],
                business_impact="Unknown",
                recommendation="Unknown",
                priority=InsightPriority.LOW,
            )
        ]
    )

    class FakeResponse:
        parsed = invalid_report

    generator.client.models.generate_content = lambda **kwargs: FakeResponse()

    with pytest.raises(
        ValueError,
        match="failed validation",
    ):
        generator.generate(
            state=create_agent_state(
                user_goal="Find insights",
                dataset_names=["customers"],
                metric_profiles=[],
                available_tools=[],
            ),
            observations=[
                AgentObservation(
                    tool_name="correlation",
                    target_columns=["revenue", "cost"],
                    success=True,
                    result={"coefficient": 0.99},
                )
            ],
        )