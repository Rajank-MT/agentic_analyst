import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.agent.context import build_agent_context
from app.agent.decision import parse_agent_action
from app.agent.llm import AgentLLM
from app.data.model import AgentAction, AgentObservation, AgentState


class OllamaAgentLLM(AgentLLM):
    """
    Ollama-backed implementation of the AgentLLM interface.

    The LLM is responsible for deciding the next analytical action.
    Deterministic Python tools perform the actual calculations.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def decide(
        self,
        state: AgentState,
        observations: list[AgentObservation],
    ) -> AgentAction:
        context = build_agent_context(
            state=state,
            observations=observations,
        )

        system_prompt = self._build_system_prompt()

        user_prompt = (
            "Analyze the current agent context and decide the next action.\n\n"
            "Return ONLY a valid JSON object matching the required AgentAction "
            "schema.\n\n"
            f"AGENT CONTEXT:\n{json.dumps(context, default=str, indent=2)}"
        )

        response = self._call_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return parse_agent_action(response)

    def _build_system_prompt(self) -> str:
        return """
You are the reasoning engine of an autonomous Agentic Data Analyst.

Your job is to decide WHAT analytical investigation should happen next.

You do NOT calculate statistics yourself.
You do NOT invent numerical results.
You do NOT directly manipulate the dataset.

Deterministic Python analysis tools perform calculations.
You choose the appropriate tool based on the available dataset context
and previous observations.

Your objectives are:

1. Discover meaningful business insights.
2. Prefer non-obvious and actionable findings over trivial observations.
3. Use previous observations to decide what should be investigated next.
4. Form hypotheses from available evidence and investigate them with tools.
5. Follow up on interesting findings when additional analysis could explain
   their cause, magnitude, or business significance.
6. Avoid redundant analysis.
7. Stop when sufficient evidence has been collected.

Rules:

- Only use tools listed in the available_tools section.
- Only use columns listed in the metrics section.
- Do not invent tool names.
- Do not invent column names.
- Respect the required roles of each tool.
- RUN_TOOL means another deterministic analysis should be performed.
- FINISH means enough evidence has been collected and analysis can stop.
- Do not claim facts that are not supported by observations.
- Do not fabricate numerical values.
- Return ONLY valid JSON.
- Do not include markdown.
- Do not include explanations outside the JSON object.

For RUN_TOOL, provide:
- action_type = "run_tool"
- tool_call.tool_name
- tool_call.target_columns
- tool_call.reasoning
- reason

For FINISH:
- action_type = "finish"
- tool_call = null
- reason

Think like a senior business analyst:
prioritize investigations that could reveal revenue drivers, profitability
differences, customer behavior, operational issues, unusual patterns,
important trends, concentration risks, or other business-relevant findings.
""".strip()

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "format": AgentAction.model_json_schema(),
        }

        request = Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")

        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP error {exc.code}: {error_body}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}: {exc.reason}"
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        try:
            response_json: dict[str, Any] = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from exc

        message = response_json.get("message")

        if not isinstance(message, dict):
            raise RuntimeError(
                "Ollama response does not contain a valid message."
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Ollama returned an empty model response."
            )

        return content