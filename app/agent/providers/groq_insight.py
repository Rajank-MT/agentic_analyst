import json
import os

from dotenv import load_dotenv
from groq import Groq

from app.agent.context import build_agent_context
from app.agent.insight_generator import InsightGenerator
from app.agent.insight_validator import validate_insight_report
from app.data.model import BusinessInsightReport


load_dotenv()


class GroqInsightGenerator(InsightGenerator):
    """
    Groq-powered business insight generation engine.

    The LLM interprets deterministic investigation results and converts
    them into validated business insights.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    # =============================================================
    # GENERATE BUSINESS INSIGHTS
    # =============================================================

    def generate(
        self,
        state,
        observations,
    ):

        # ---------------------------------------------------------
        # BUILD CONTEXT
        # ---------------------------------------------------------

        full_context = build_agent_context(
            state=state,
            observations=observations,
        )

        context = {
            "user_goal": full_context.get(
                "user_goal"
            ),
            "datasets": full_context.get(
                "datasets"
            ),
            "investigation_summary": full_context.get(
                "investigation_summary"
            ),
            "observation_results": full_context.get(
                "observation_results",
                [],
            ),
        }

        context_json = json.dumps(
            context,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # ---------------------------------------------------------
        # PROMPT
        # ---------------------------------------------------------

        prompt = f"""
You are the Business Insight Generation Engine for an autonomous
data analysis system.

Your job is to convert the available deterministic investigation
results into meaningful business insights.

IMPORTANT:

- Use ONLY evidence present in the context.
- Never invent numbers, facts, relationships, or business results.
- Do not perform calculations yourself.
- Do not claim causation from correlation.
- Do not create duplicate insights.
- Prefer important and non-obvious findings.
- Every insight MUST be supported by evidence.
- If an observation is not strong enough to support an insight,
  do not use it.
- Do not mention an analysis result that is not present in the
  supplied context.

MOST IMPORTANT OUTPUT RULE:

EVERY insight object MUST contain ALL of these fields:

1. title
2. finding
3. evidence
4. business_impact
5. recommendation
6. priority
7. caveats

Never omit any of these fields.

Even if business impact, recommendation, or caveats are limited,
the fields MUST still be present.

For example:

{{
  "title": "Revenue is concentrated in major cities",
  "finding": "The largest cities contribute a substantial share of revenue.",
  "evidence": [
    {{
      "execution_index": 8,
      "fact": "Kolkata contributes 12.71% of total revenue."
    }}
  ],
  "business_impact": "Revenue exposure is distributed across major markets.",
  "recommendation": "Monitor city-level performance and maintain diversification.",
  "priority": "medium",
  "caveats": []
}}

VALID priority values:

- high
- medium
- low

Evidence MUST use this structure:

{{
  "execution_index": 0,
  "fact": "Exact fact supported by the investigation result."
}}

The execution_index MUST refer to an execution present in the
investigation context.

OUTPUT REQUIREMENTS:

Return ONLY valid JSON.

The root object MUST contain:

{{
  "insights": []
}}

Each insight MUST contain:

{{
  "title": "...",
  "finding": "...",
  "evidence": [],
  "business_impact": "...",
  "recommendation": "...",
  "priority": "medium",
  "caveats": []
}}

Do NOT use fields such as:

- id
- description
- tool
- details

Use the exact field names required above.

If there is insufficient evidence for an insight, omit that insight
instead of generating unsupported information.

CURRENT INVESTIGATION CONTEXT:

{context_json}
"""

        # ---------------------------------------------------------
        # TOKEN DEBUG
        # ---------------------------------------------------------

        print("\n" + "=" * 70)
        print("GROQ INSIGHT TOKEN DEBUG")
        print("=" * 70)
        print(
            f"Context characters : {len(context_json):,}"
        )
        print(
            f"Approx context tokens : {len(context_json) // 4:,}"
        )
        print(
            f"Prompt characters  : {len(prompt):,}"
        )
        print(
            f"Approx prompt tokens : {len(prompt) // 4:,}"
        )
        print("=" * 70)

        # ---------------------------------------------------------
        # CALL GROQ
        # ---------------------------------------------------------

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a business insight generation engine. "
                        "Return ONLY valid JSON. "
                        "Every insight must contain all required "
                        "BusinessInsight fields."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_completion_tokens=1800,
            response_format={
                "type": "json_object"
            },
            reasoning_effort="low",
        )

        # ---------------------------------------------------------
        # EXTRACT RESPONSE
        # ---------------------------------------------------------

        raw_response = (
            response.choices[0].message.content
        )

        if not raw_response:
            raise RuntimeError(
                "Groq returned an empty insight response."
            )

        print("\nGROQ INSIGHT RESPONSE:")
        print(raw_response)

        # ---------------------------------------------------------
        # PARSE JSON
        # ---------------------------------------------------------

        try:

            report_data = json.loads(
                raw_response
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                f"Groq returned invalid JSON: {exc}"
            ) from exc

        # ---------------------------------------------------------
        # VALIDATE BUSINESS INSIGHT SCHEMA
        # ---------------------------------------------------------

        try:

            report = BusinessInsightReport.model_validate(
                report_data
            )

        except Exception as exc:

            print(
                "\nINVALID BUSINESS INSIGHT REPORT:"
            )

            print(
                json.dumps(
                    report_data,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

            raise RuntimeError(
                "Groq returned JSON that does not match "
                "BusinessInsightReport.\n"
                f"Validation error: {exc}\n"
                f"Raw response: {raw_response}"
            ) from exc

        # ---------------------------------------------------------
        # DOMAIN VALIDATION
        # ---------------------------------------------------------

        validation_errors = (
            validate_insight_report(
                report=report,
                observations=observations,
            )
        )

        if validation_errors:

            raise ValueError(
                "Insight validation failed: "
                + "; ".join(
                    str(error)
                    for error in validation_errors
                )
            )

        return report