import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.agent.context import build_agent_context
from app.agent.llm import AgentLLM
from app.data.model import AgentAction, AgentObservation, AgentState


load_dotenv()


class GeminiAgentLLM(AgentLLM):
    """Gemini-powered reasoning engine for the Agentic Data Analyst."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def decide(
        self,
        state: AgentState,
        observations: list[AgentObservation],
    ) -> AgentAction:

        context = build_agent_context(
            state=state,
            observations=observations,
        )

        prompt = f"""
You are the reasoning engine of an autonomous Agentic Data Analyst.

Your job is to decide WHAT analytical investigation should happen next.

You do NOT calculate statistics yourself.
You do NOT invent numerical results.
Deterministic Python tools perform all calculations.

Your objectives:
- Discover meaningful business insights.
- Prefer non-obvious and actionable findings.
- Use previous observations.
- Form hypotheses and investigate them.
- Avoid repeating the same analysis.
- Stop when sufficient evidence exists.

Rules:
- Use ONLY tools listed in available_tools.
- Use ONLY columns listed in metrics.
- Respect the required roles of each tool.
- Never invent tools or columns.
- RUN_TOOL means execute a deterministic Python analysis.
- FINISH means enough evidence has been collected.
- Return only the required structured AgentAction.

CURRENT AGENT CONTEXT:

{context}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AgentAction,
                temperature=0.2,
            ),
        )

        if response.parsed is None:
          raise RuntimeError("Gemini returned no parsed AgentAction.")

        print("\n" + "=" * 70)
        print("GEMINI DECISION")
        print("=" * 70)
        print(response.parsed.model_dump())

        return response.parsed