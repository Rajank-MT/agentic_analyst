from app.agent.session import AgentSession
from app.agent.state import create_agent_state
from app.data.model import AgentObservation


def test_agent_session_stores_initial_state():
    state = create_agent_state(
        user_goal="Find important business insights.",
        dataset_names=["sales"],
        metric_profiles=[],
        available_tools=[],
    )

    session = AgentSession(
        session_id="test-session",
        state=state,
    )

    assert session.session_id == "test-session"
    assert session.state is state
    assert session.observations == []
    assert session.conversation_history == []


def test_agent_session_stores_conversation_history():
    state = create_agent_state(
        user_goal="Analyze sales.",
        dataset_names=["sales"],
        metric_profiles=[],
        available_tools=[],
    )

    session = AgentSession(
        session_id="test-session",
        state=state,
    )

    session.add_user_message(
        "Analyze my sales dataset."
    )

    session.add_assistant_message(
        "I found several important business insights."
    )

    assert session.conversation_history == [
        {
            "role": "user",
            "message": "Analyze my sales dataset.",
        },
        {
            "role": "assistant",
            "message": "I found several important business insights.",
        },
    ]


def test_agent_session_stores_observations():
    state = create_agent_state(
        user_goal="Analyze sales.",
        dataset_names=["sales"],
        metric_profiles=[],
        available_tools=[],
    )

    observation = AgentObservation(
        tool_name="numeric_statistics",
        target_columns=["revenue"],
        success=True,
        result={"mean": 1000},
    )

    session = AgentSession(
        session_id="test-session",
        state=state,
    )

    session.observations.append(observation)

    assert len(session.observations) == 1
    assert session.observations[0].tool_name == (
        "numeric_statistics"
    )