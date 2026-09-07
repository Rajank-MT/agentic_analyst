import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.agent.context import build_agent_context
from app.agent.insight_generator import InsightGenerator
from app.data.model import (
    AgentObservation,
    AgentState,
    BusinessInsightReport,
)

from app.agent.insight_validator import validate_insight_report

load_dotenv()


class GeminiInsightGenerator(InsightGenerator):

    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self,
        state: AgentState,
        observations: list[AgentObservation],
    ) -> BusinessInsightReport:

        context = build_agent_context(
            state=state,
            observations=observations,
        )

        prompt = f"""
You are the business insight generation engine of an autonomous
Agentic Data Analyst.

Your job is to convert deterministic analytical results into
important, evidence-backed business insights.

STRICT RULES:

1. Never invent numbers.
2. Never invent facts that are not present in the observations.
3. Do not perform calculations yourself.
4. Use only the provided analytical results.
5. Every insight must contain supporting evidence.
6. Evidence must reference the execution_index of an observation.
7. Clearly distinguish observed facts from business interpretation.
8. Do not claim causation from correlation.
9. If evidence is weak or insufficient, explicitly mention it in caveats.
10. Do not create an insight simply because a result exists.
11. Prioritize non-obvious and business-relevant findings.
12. Recommendations must logically follow from the evidence.
13. Return ONLY a valid BusinessInsightReport.

Context:
{context}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BusinessInsightReport,
                temperature=0.2,
            ),
        )

        if response.parsed is None:
            raise RuntimeError(
           "Gemini returned no parsed BusinessInsightReport."
        )

        report = response.parsed

        validation_errors = validate_insight_report(
            report=report,
            observations=observations,
        )

        if validation_errors:
            raise ValueError(
                "Generated insight report failed validation: "
                + "; ".join(validation_errors)
            )

        return report