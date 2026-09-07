import json

import pytest

from app.agent.providers.ollama import OllamaAgentLLM
from app.data.model import (
    AgentAction,
    AgentActionType,
    AgentState,
)


def _create_state() -> AgentState:
    return AgentState(
        user_goal="Find important business insights.",
        dataset_names=["customers"],
        metric_profiles=[],
        available_tools=[],
    )


def test_ollama_provider_initializes_with_defaults():
    llm = OllamaAgentLLM()

    assert llm.model == "qwen3:8b"
    assert llm.base_url == "http://localhost:11434"
    assert llm.timeout == 120


def test_ollama_provider_accepts_custom_configuration():
    llm = OllamaAgentLLM(
        model="test-model",
        base_url="http://example.com/",
        timeout=30,
    )

    assert llm.model == "test-model"
    assert llm.base_url == "http://example.com"
    assert llm.timeout == 30


def test_ollama_provider_parses_run_tool_response(monkeypatch):
    llm = OllamaAgentLLM()

    response_payload = {
        "message": {
            "content": json.dumps(
                {
                    "action_type": "run_tool",
                    "tool_call": {
                        "tool_name": "numeric_statistics",
                        "target_columns": ["revenue"],
                        "reasoning": "Inspect the revenue distribution.",
                    },
                    "reason": "Revenue is an important business measure.",
                }
            )
        }
    }

    def fake_urlopen(request, timeout):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        return FakeResponse()

    monkeypatch.setattr(
        "app.agent.providers.ollama.urlopen",
        fake_urlopen,
    )

    action = llm.decide(
        state=_create_state(),
        observations=[],
    )

    assert isinstance(action, AgentAction)
    assert action.action_type == AgentActionType.RUN_TOOL
    assert action.tool_call is not None
    assert action.tool_call.tool_name == "numeric_statistics"
    assert action.tool_call.target_columns == ["revenue"]


def test_ollama_provider_parses_finish_response(monkeypatch):
    llm = OllamaAgentLLM()

    response_payload = {
        "message": {
            "content": json.dumps(
                {
                    "action_type": "finish",
                    "tool_call": None,
                    "reason": "Sufficient evidence has been collected.",
                }
            )
        }
    }

    def fake_urlopen(request, timeout):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        return FakeResponse()

    monkeypatch.setattr(
        "app.agent.providers.ollama.urlopen",
        fake_urlopen,
    )

    action = llm.decide(
        state=_create_state(),
        observations=[],
    )

    assert action.action_type == AgentActionType.FINISH
    assert action.tool_call is None


def test_ollama_provider_rejects_invalid_model_response(monkeypatch):
    llm = OllamaAgentLLM()

    response_payload = {
        "message": {
            "content": "This is not valid JSON."
        }
    }

    def fake_urlopen(request, timeout):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        return FakeResponse()

    monkeypatch.setattr(
        "app.agent.providers.ollama.urlopen",
        fake_urlopen,
    )

    with pytest.raises(ValueError, match="not valid JSON"):
        llm.decide(
            state=_create_state(),
            observations=[],
        )


def test_ollama_provider_handles_missing_message(monkeypatch):
    llm = OllamaAgentLLM()

    response_payload = {}

    def fake_urlopen(request, timeout):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        return FakeResponse()

    monkeypatch.setattr(
        "app.agent.providers.ollama.urlopen",
        fake_urlopen,
    )

    with pytest.raises(RuntimeError, match="valid message"):
        llm.decide(
            state=_create_state(),
            observations=[],
        )