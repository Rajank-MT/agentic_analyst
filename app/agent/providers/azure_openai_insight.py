
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.agent.context import build_agent_context
from app.agent.insight_generator import InsightGenerator
from app.agent.insight_validator import validate_insight_report
from app.data.model import (
    AgentObservation,
    AgentState,
    BusinessInsightReport,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _json_safe(value):
    """Convert nested Pydantic models and containers to JSON-safe data."""
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class AzureOpenAIInsightGenerator(InsightGenerator):
    """Generate validated business insights using Azure Foundry."""

    def __init__(self, model: str | None = None):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        deployment = (
            model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        ).strip()

        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": endpoint,
                "AZURE_OPENAI_API_KEY": api_key,
                "AZURE_OPENAI_DEPLOYMENT": deployment,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing Azure configuration: " + ", ".join(missing)
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{endpoint}/openai/v1/",
            timeout=90.0,
            max_retries=2,
        )
        self.model = deployment

    def generate(
        self,
        state: AgentState,
        observations: list[AgentObservation],
    ) -> BusinessInsightReport:
        full_context = build_agent_context(
            state=state,
            observations=observations,
        )

        context = {
            "user_goal": full_context.get("user_goal"),
            "datasets": full_context.get("datasets", []),
            "investigation_summary": full_context.get(
                "investigation_summary"
            ),
            "observation_results": [
                {
                    "execution_index": index,
                    "tool_name": observation.tool_name,
                    "target_columns": observation.target_columns,
                    "success": observation.success,
                    "status": (
                        "success" if observation.success else "failed"
                    ),
                    "result": (
                        _json_safe(observation.result)
                        if observation.success
                        else None
                    ),
                    "error": observation.error,
                }
                for index, observation in enumerate(observations)
            ],
        }

        context_json = json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        prompt = f"""
You are the business insight generation engine for an Agentic Data Analyst.

Convert deterministic investigation results into useful business insights.

STRICT RULES:
- Use only evidence present in the supplied context.
- Never invent facts, numbers, calculations, or relationships.
- Do not claim causation from correlation.
- Do not create duplicate or trivial insights.
- Every insight must have supporting evidence.
- Evidence must reference an actual execution_index.
- Recommendations must follow logically from the evidence.
- If evidence is insufficient, return fewer insights rather than
  inventing unsupported findings.
- Return only valid JSON matching BusinessInsightReport.

Every insight must contain:
title, finding, evidence, business_impact, recommendation, priority,
and caveats.

Evidence items must contain:
execution_index and fact.

Valid priority values: high, medium, low.

Return this root structure:
{{
  "insights": [
    {{
      "title": "Short finding title",
      "finding": "What the evidence shows",
      "evidence": [
        {{
          "execution_index": 0,
          "fact": "A fact supported by that execution."
        }}
      ],
      "business_impact": "Why the finding matters.",
      "recommendation": "A proportionate next step.",
      "priority": "medium",
      "caveats": []
    }}
  ]
}}

Use the exact fields required by the application's
BusinessInsightReport schema. Do not add unsupported fields.

INVESTIGATION CONTEXT:
{context_json}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate evidence-backed business insights. "
                        "Return only valid JSON matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=4000,
            response_format={"type": "json_object"},
        )

        raw_response = response.choices[0].message.content
        if not raw_response:
            raise RuntimeError(
                "Azure OpenAI returned an empty insight response."
            )

        try:
            report_data = json.loads(raw_response)
            report = BusinessInsightReport.model_validate(report_data)
        except Exception as exc:
            raise RuntimeError(
                "Azure OpenAI returned invalid BusinessInsightReport JSON: "
                f"{exc}\nRaw response: {raw_response}"
            ) from exc

        validation_errors = validate_insight_report(
            report=report,
            observations=observations,
        )

        if validation_errors:
            raise ValueError(
                "Insight validation failed: "
                + "; ".join(str(error) for error in validation_errors)
            )

        return report
