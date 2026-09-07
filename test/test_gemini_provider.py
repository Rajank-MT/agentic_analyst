import pytest

from app.agent.providers.gemini import GeminiAgentLLM
from app.agent.state import create_agent_state
from app.data.model import AgentAction, AgentActionType


def test_gemini_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiAgentLLM()


def test_gemini_provider_initializes_with_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeClient:
        pass

    monkeypatch.setattr(
        "app.agent.providers.gemini.genai.Client",
        lambda api_key: FakeClient(),
    )

    provider = GeminiAgentLLM()

    assert provider.model == "gemini-2.5-flash"
    assert isinstance(provider.client, FakeClient)


def test_gemini_provider_custom_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeClient:
        pass

    monkeypatch.setattr(
        "app.agent.providers.gemini.genai.Client",
        lambda api_key: FakeClient(),
    )

    provider = GeminiAgentLLM(model="gemini-test-model")

    assert provider.model == "gemini-test-model"


def test_gemini_decide_returns_agent_action(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        parsed = AgentAction(
            action_type=AgentActionType.FINISH,
            reason="Enough evidence has been collected.",
            tool_call=None,
        )

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    monkeypatch.setattr(
        "app.agent.providers.gemini.genai.Client",
        lambda api_key: FakeClient(),
    )

    provider = GeminiAgentLLM()

    state = create_agent_state(
        user_goal="Find important business insights.",
        dataset_names=["customers"],
        metric_profiles=[],
        available_tools=[],
    )

    action = provider.decide(
        state=state,
        observations=[],
    )

    assert isinstance(action, AgentAction)
    assert action.action_type == AgentActionType.FINISH
    assert action.reason == "Enough evidence has been collected."


def test_gemini_decide_raises_when_response_is_not_parsed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        parsed = None

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    monkeypatch.setattr(
        "app.agent.providers.gemini.genai.Client",
        lambda api_key: FakeClient(),
    )

    provider = GeminiAgentLLM()

    state = create_agent_state(
        user_goal="Find important business insights.",
        dataset_names=["customers"],
        metric_profiles=[],
        available_tools=[],
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini returned no parsed AgentAction",
    ):
        provider.decide(
            state=state,
            observations=[],
        )