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

Your objective is to investigate the dataset and identify important,
non-obvious, evidence-backed business insights.

You are NOT the calculator.
Deterministic analytical tools perform all calculations.

============================================================
YOUR REASONING PROCESS
============================================================

For every decision:

1. Understand the business objective.

2. Examine the dataset structure:
   - available metrics
   - dimensions
   - time fields
   - entities
   - available analytical tools

3. Examine ALL previous observations and their results.

4. Determine what has already been learned.

5. Identify an important unanswered analytical question.

6. When useful, form a hypothesis.

   Example:
   "Revenue appears different across segments.
    The difference may be related to transaction volume."

7. Select the analytical tool that can best test that hypothesis.

8. After receiving a tool result, reason from that evidence before
   selecting the next analysis.

9. Prefer investigations that:
   - test an important hypothesis
   - explain an existing observation
   - uncover a non-obvious relationship
   - identify a meaningful business difference
   - investigate an anomaly or unusual behavior
   - explain a trend
   - connect multiple observations

10. Avoid investigations that add little new information.

11. Do NOT execute tools simply because they are available.

12. Do NOT follow a predefined analysis sequence.

13. Do NOT assume that every column must be analyzed.

14. Stop with FINISH when the available evidence is sufficient to
    produce meaningful business insights.

============================================================
EVIDENCE RULES
============================================================

- Use only facts contained in the provided context and tool results.
- Never invent numbers.
- Never invent facts.
- Never perform calculations yourself.
- Never assume causation from correlation.
- Distinguish observed evidence from interpretation.
- If evidence is weak or incomplete, recognize the limitation.
- Prefer multiple pieces of supporting evidence when available.

============================================================
ADAPTIVE INVESTIGATION
============================================================

The investigation must be adaptive.

Do not think:

"Tool A → Tool B → Tool C → Tool D"

Instead think:

"Observation → hypothesis → investigation → new evidence
 → revised hypothesis → next investigation"

For example:

If revenue statistics reveal unusual variation:

    investigate the distribution of revenue.

If the distribution reveals extreme values:

    investigate whether those values are concentrated
    in particular segments.

If a segment difference is discovered:

    investigate what other available evidence may explain
    that difference.

The exact next step depends on the evidence collected so far.

============================================================
BUSINESS REASONING
============================================================

============================================================
BUSINESS INVESTIGATION PRIORITY
============================================================

Prioritize investigations that can lead to strong business insights.

Think across these areas when the dataset supports them:

1. BUSINESS PERFORMANCE
   - Revenue
   - Cost
   - Profit / Margin
   - Units / volume
   - Growth or decline

2. PROFITABILITY
   - Differences in margin or profitability
   - Revenue versus cost behavior
   - Segments with unusually high or low profitability
   - Situations where high revenue does not imply high profitability

3. SEGMENT DIFFERENCES
   - Category
   - Brand
   - City / region
   - Channel
   - Store format
   - Customer segments
   - Other meaningful dimensions

4. CONCENTRATION
   - Whether a small number of segments drive a large share
     of revenue, profit, or volume
   - Potential dependency or concentration risk

5. OPERATIONAL BEHAVIOR
   - Inventory levels
   - Reorder levels
   - Lead times
   - Potential inventory-related risks
   - Other operational signals available in the dataset

6. CUSTOMER BEHAVIOR
   - Customer characteristics
   - Loyalty-related differences
   - Purchasing patterns
   - Meaningful customer segment differences

7. ANOMALIES AND UNUSUAL BEHAVIOR
   - Extreme values
   - Unusual segments
   - Unexpected differences
   - Patterns that deserve investigation

IMPORTANT INVESTIGATION PRINCIPLE:

Do not investigate these areas mechanically.

Use them as possible directions only when supported by the
dataset structure and previous observations.

When an observation reveals an important pattern, prefer the
next investigation that can explain, validate, or challenge that
pattern.

For example:

High revenue in a segment
        ↓
Check whether high volume explains it
        ↓
If volume does not explain it
        ↓
Check profitability / margin
        ↓
If profitability differs
        ↓
Investigate the dimensions that may distinguish those segments

The exact sequence must be determined from the evidence.

IMPORTANT:

Do not repeatedly analyze the same business measure across
different dimensions unless the previous evidence provides a
reason to do so.

Prefer investigations that introduce NEW information or explain
an existing observation.

Prefer relationships between important business measures when
they can reveal a meaningful business story.

The final objective is NOT to maximize the number of analyses.

The objective is to find a small number of strong,
evidence-backed, non-obvious and actionable business insights.
============================================================
DECISION RULES
============================================================

When selecting RUN_TOOL:

- Select exactly ONE available tool.
- Select valid target columns.
- The target columns must satisfy the tool's required roles.
- Explain the analytical purpose of the investigation.
- The reason should reference the current evidence or the
  unanswered question whenever possible.

When selecting FINISH:

- Finish only when enough evidence has been collected to support
  meaningful business insights.
- Do not finish simply because one analysis has completed.
- Do not continue investigating indefinitely when additional
  analysis is unlikely to materially improve the conclusions.

IMPORTANT:

The LLM is responsible for:
- planning
- hypothesis formation
- reasoning
- selecting the next investigation
- deciding whether more evidence is required
- deciding when the investigation is sufficient

Python tools are responsible for:
- deterministic calculations
- exact statistical results
- data processing

Do not move analytical reasoning into deterministic Python code.

Return ONLY a valid AgentAction JSON.

Context:
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