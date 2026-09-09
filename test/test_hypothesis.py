from app.data.model import (
    AgentAction,
    AgentActionType,
    AgentState,
    HypothesisStatus,
    InvestigationHypothesis,
)


def test_investigation_hypothesis_defaults():
    hypothesis = InvestigationHypothesis(
        hypothesis="Revenue differences may be concentrated in specific regions."
    )

    assert hypothesis.hypothesis == (
        "Revenue differences may be concentrated in specific regions."
    )
    assert hypothesis.supporting_execution_indexes == []
    assert hypothesis.status == HypothesisStatus.ACTIVE
    assert hypothesis.conclusion is None


def test_investigation_hypothesis_can_store_evidence():
    hypothesis = InvestigationHypothesis(
        hypothesis="Revenue and cost may be strongly related.",
        supporting_execution_indexes=[0, 2],
        status=HypothesisStatus.SUPPORTED,
        conclusion="Revenue and cost show a strong positive relationship.",
    )

    assert hypothesis.supporting_execution_indexes == [0, 2]
    assert hypothesis.status == HypothesisStatus.SUPPORTED
    assert hypothesis.conclusion == (
        "Revenue and cost show a strong positive relationship."
    )


def test_agent_state_starts_with_empty_hypotheses():
    state = AgentState(
        user_goal="Find important business insights.",
    )

    assert state.hypotheses == []


def test_agent_action_can_contain_hypothesis():
    hypothesis = InvestigationHypothesis(
        hypothesis="Revenue may differ substantially between segments."
    )

    action = AgentAction(
        action_type=AgentActionType.RUN_TOOL,
        reason="Investigate the segment difference.",
        hypothesis=hypothesis,
    )

    assert action.hypothesis is not None
    assert action.hypothesis.hypothesis == (
        "Revenue may differ substantially between segments."
    )