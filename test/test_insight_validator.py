from app.agent.insight_validator import validate_insight_report
from app.data.model import (
    AgentObservation,
    BusinessInsight,
    BusinessInsightReport,
    InsightEvidence,
    InsightPriority,
)


def _build_observation(success=True):
    return AgentObservation(
        tool_name="correlation",
        target_columns=["revenue", "cost"],
        success=success,
        result={
            "coefficient": 0.996675,
            "strength": "very_strong",
        },
    )


def _build_insight(execution_index=0):
    return BusinessInsight(
        title="Revenue and cost are strongly correlated",
        finding="Revenue and cost have a very strong positive relationship.",
        evidence=[
            InsightEvidence(
                execution_index=execution_index,
                fact="Pearson correlation = 0.996675",
            )
        ],
        business_impact="Cost growth may accompany revenue growth.",
        recommendation="Investigate cost efficiency.",
        priority=InsightPriority.HIGH,
    )


def test_valid_insight_report():
    report = BusinessInsightReport(
        insights=[_build_insight(0)]
    )

    observations = [_build_observation()]

    errors = validate_insight_report(
        report=report,
        observations=observations,
    )

    assert errors == []


def test_invalid_execution_index():
    report = BusinessInsightReport(
        insights=[_build_insight(5)]
    )

    observations = [_build_observation()]

    errors = validate_insight_report(
        report=report,
        observations=observations,
    )

    assert len(errors) == 1
    assert "invalid execution index 5" in errors[0]


def test_failed_execution_cannot_support_insight():
    report = BusinessInsightReport(
        insights=[_build_insight(0)]
    )

    observations = [_build_observation(success=False)]

    errors = validate_insight_report(
        report=report,
        observations=observations,
    )

    assert len(errors) == 1
    assert "failed execution 0" in errors[0]


def test_insight_without_evidence():
    insight = BusinessInsight(
        title="Unsupported insight",
        finding="Something happened.",
        evidence=[],
        business_impact="Unknown.",
        recommendation="Investigate further.",
        priority=InsightPriority.LOW,
    )

    report = BusinessInsightReport(
        insights=[insight]
    )

    observations = [_build_observation()]

    errors = validate_insight_report(
        report=report,
        observations=observations,
    )

    assert len(errors) == 1
    assert "no supporting evidence" in errors[0]


def test_empty_report():
    report = BusinessInsightReport(
        insights=[]
    )

    observations = [_build_observation()]

    errors = validate_insight_report(
        report=report,
        observations=observations,
    )

    assert errors == ["Insight report contains no insights."]