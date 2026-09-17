from abc import abstractmethod
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from app.agent.context import build_agent_context
from app.agent.llm import AgentLLM
from app.data.model import AgentAction


load_dotenv()


class GroqAgentLLM(AgentLLM):
    """
    Groq-powered reasoning engine for the Agentic Data Analyst.

    Responsibilities:
    - Decide what to investigate next.
    - Select predefined deterministic tools.
    - Request composed analysis when required.
    - Decide when enough evidence exists and finish.
    - Never perform analytical calculations itself.
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
    # MAIN DECISION METHOD
    # =============================================================

    
    def decide(
        self,
        state,
        observations,
    ) -> AgentAction:

        # ---------------------------------------------------------
        # 1. BUILD COMPLETE AGENT CONTEXT
        # ---------------------------------------------------------

        full_context = build_agent_context(
            state=state,
            observations=observations,
        )

        # ---------------------------------------------------------
        # 2. COMPACT CONTEXT
        #
        # Important information is preserved:
        # - goal
        # - datasets
        # - metrics
        # - tools
        # - coverage
        # - investigation history
        # - recent evidence
        # - hypotheses
        # ---------------------------------------------------------

        context = self._compact_context(full_context)

        context_json = json.dumps(
            context,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # ---------------------------------------------------------
        # 3. TOKEN DEBUG
        # ---------------------------------------------------------

        estimated_context_tokens = len(context_json) // 4

        print("\n" + "=" * 70)
        print("GROQ TOKEN DEBUG")
        print("=" * 70)
        print(
            f"Context characters : {len(context_json):,}"
        )
        print(
            f"Approx tokens      : {estimated_context_tokens:,}"
        )
        print("=" * 70)

        # ---------------------------------------------------------
        # 4. COMPACT DECISION PROMPT
        #
        # Keep instructions short.
        # The context contains the detailed evidence.
        # ---------------------------------------------------------

        prompt = f"""
You are the reasoning engine of an autonomous Agentic Data Analyst.

Your job is to decide the NEXT useful investigation.

You do NOT calculate statistics yourself.
Deterministic Python tools perform calculations.
Use only the supplied context.
DATASET UNDERSTANDING:

The context may contain an adaptive dataset sample.

Use the sample only to understand:
- what kinds of entities, dimensions, measures, and values exist
- possible relationships between columns
- potentially interesting segments or patterns worth investigating
- unfamiliar terminology or dataset structure

Do NOT use sample rows to make final numerical claims.
Do NOT calculate statistics from the sample.
All quantitative evidence must come from deterministic analysis results.

IMPORTANT RULES:

1. Use only exact tool names and column names from the context.
2. Never invent facts, numbers, columns, or tools.
3. Use previous investigations to avoid redundant work.
4. Prefer unexplored measures, dimensions, time patterns, relationships,
   distributions, anomalies, contributions, and validation opportunities.
5. Explore multiple meaningful directions instead of repeatedly analyzing
   the same metric.
6. Deepen a previous finding only when additional evidence can add value.
7. Never claim causation from correlation.
8. Use run_composed_analysis only when existing tools cannot reasonably
   perform the required analysis.
9. The goal is meaningful evidence, not maximum investigation count.
10. When enough evidence exists, choose finish.

BREADTH:

Before choosing an investigation, check:
- unexplored important measures
- unexplored dimensions/segments
- unexplored time patterns
- important variable relationships
- anomalies/distributions
- validation opportunities
- business areas not yet investigated

OUTPUT:

Return ONLY valid JSON.

For run_tool:
{{
  "action_type": "run_tool",
  "reason": "short reason",
  "tool_call": {{
    "tool_name": "exact tool name",
    "target_columns": ["exact column names"],
    "reasoning": "short reasoning"
  }},
  "composed_analysis": null,
  "hypothesis": null
}}

For run_composed_analysis:
{{
  "action_type": "run_composed_analysis",
  "reason": "short reason",
  "tool_call": null,
  "composed_analysis": {{
    "analysis_request": {{}},
    "reasoning": "short reasoning"
  }},
  "hypothesis": null
}}

For finish:
{{
  "action_type": "finish",
  "reason": "short reason",
  "tool_call": null,
  "composed_analysis": null,
  "hypothesis": null
}}

The JSON must contain:
- action_type
- reason
- tool_call when action_type is run_tool
- composed_analysis when action_type is run_composed_analysis

Optional hypothesis format:
{{
  "hypothesis": {{
    "hypothesis": "testable hypothesis",
    "supporting_execution_indexes": [],
    "status": "active",
    "conclusion": null
  }}
}}

Valid hypothesis statuses:
active, supported, weakened, rejected

Return JSON only. Do not return markdown.

CURRENT AGENT CONTEXT:

{context_json}
"""

        # ---------------------------------------------------------
        # 5. PROMPT TOKEN DEBUG
        # ---------------------------------------------------------

        estimated_prompt_tokens = len(prompt) // 4

        print(
            f"Prompt characters  : {len(prompt):,}"
        )
        print(
            f"Approx tokens      : {estimated_prompt_tokens:,}"
        )
        print("=" * 70)

        # ---------------------------------------------------------
        # 6. CALL GROQ
        # ---------------------------------------------------------

        response = None

        try:

            for attempt in range(3):

                try:

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an autonomous data-analysis "
                                    "reasoning engine. "
                                    "Return only valid JSON. "
                                    "The response must be a JSON object "
                                    "matching the AgentAction structure."
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        temperature=0.1,

                        # GPT-OSS can use reasoning tokens.
                        # Give it enough completion budget.
                        max_completion_tokens=1500,

                        # Force JSON output.
                        response_format={
                            "type": "json_object"
                        },

                        # Keep reasoning concise.
                        reasoning_effort="low",
                    )

                    break

                except Exception as exc:

                    error_text = str(exc)

                    # -------------------------------------------------
                    # GROQ RATE LIMIT RETRY
                    # -------------------------------------------------

                    if (
                        "429" in error_text
                        and attempt < 2
                    ):

                        print(
                            "Groq rate limit reached. "
                            "Retrying in 30 seconds..."
                        )

                        time.sleep(30)

                    else:
                        raise

            if response is None:
                raise RuntimeError(
                    "Groq did not return a response."
                )

            # ---------------------------------------------------------
            # 7. EXTRACT RESPONSE
            # ---------------------------------------------------------

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    "Groq returned an empty response."
                )

            print("\nGROQ RESPONSE:")
            print(content)

            # ---------------------------------------------------------
            # 8. PARSE JSON
            # ---------------------------------------------------------

            try:
                parsed = json.loads(content)

            except json.JSONDecodeError as exc:

                print("\nINVALID GROQ JSON:")
                print(content)

                raise RuntimeError(
                    f"Groq returned invalid JSON: {exc}"
                ) from exc

            # ---------------------------------------------------------
            # 9. VALIDATE AGAINST AgentAction
            # ---------------------------------------------------------

            try:

                action = AgentAction.model_validate(
                    parsed
                )

            except Exception as exc:

                print("\nINVALID AgentAction:")
                print(
                    json.dumps(
                        parsed,
                        indent=2,
                        default=str,
                    )
                )

                raise RuntimeError(
                    f"Groq returned invalid AgentAction: {exc}"
                ) from exc

            return action

        except Exception as exc:

            raise RuntimeError(
                f"Groq LLM decision failed: {exc}"
            ) from exc

    # =============================================================
    # CONTEXT COMPACTION
    # =============================================================

    def _compact_context(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Reduce prompt size while preserving information required
        for intelligent agent decisions.
        """

        compact: dict[str, Any] = {}

        # ---------------------------------------------------------
        # USER GOAL
        # ---------------------------------------------------------

        compact["user_goal"] = context.get(
            "user_goal"
        )

        # ---------------------------------------------------------
        # DATASETS
        # ---------------------------------------------------------

        compact["datasets"] = context.get(
            "datasets",
            [],
        )

        # ---------------------------------------------------------
        # METRICS
        #
        # Preserve semantic information about every metric.
        # ---------------------------------------------------------

        compact_metrics = []

        for metric in context.get(
            "metrics",
            [],
        ):

            compact_metrics.append(
                {
                    "column_name": metric.get(
                        "column_name"
                    ),
                    "role": metric.get(
                        "role"
                    ),
                    "confidence": metric.get(
                        "confidence"
                    ),
                    "signals": (
                        metric.get(
                            "signals",
                            [],
                        )[:5]
                    ),
                }
            )

        compact["metrics"] = compact_metrics

        # ---------------------------------------------------------
# ADAPTIVE DATASET SAMPLE
#
# The sample helps the LLM understand unfamiliar datasets.
# It is context only; all numerical evidence comes from
# deterministic analysis on the full dataframe.
# ---------------------------------------------------------

        dataset_sample = context.get("dataset_sample")

        if dataset_sample:
            compact["dataset_sample"] = self._limit_value(
                dataset_sample,
                max_chars=4500,
            )

        # ---------------------------------------------------------
        # AVAILABLE TOOLS
        #
        # Preserve:
        # - name
        # - description
        # - analysis type
        # - required roles
        # - target cardinality if available
        #
        # This is important for avoiding invalid calls such as
        # correlation with 10+ columns.
        # ---------------------------------------------------------

        compact_tools = []

        for tool in context.get(
            "available_tools",
            [],
        ):

            description = str(
                tool.get(
                    "description",
                    "",
                )
            )

            if len(description) > 220:
                description = (
                    description[:220]
                    + "..."
                )

            item = {
                "name": tool.get("name"),
                "description": description,
                "analysis_type": tool.get(
                    "analysis_type"
                ),
                "required_roles": tool.get(
                    "required_roles",
                    [],
                ),
            }

            # Preserve target cardinality metadata
            # if the registry already exposes it.
            for key in (
                "min_target_columns",
                "max_target_columns",
                "target_column_count",
                "required_target_columns",
            ):

                if key in tool:
                    item[key] = tool[key]

            compact_tools.append(item)

        compact["available_tools"] = compact_tools

        # ---------------------------------------------------------
        # INVESTIGATION SUMMARY
        # ---------------------------------------------------------

        compact["investigation_summary"] = (
            self._limit_value(
                context.get(
                    "investigation_summary"
                ),
                max_chars=3000,
            )
        )

        # ---------------------------------------------------------
        # INVESTIGATION COVERAGE
        #
        # Keep complete coverage information because it drives
        # breadth search.
        # ---------------------------------------------------------

        compact["investigation_coverage"] = (
            context.get(
                "investigation_coverage"
            )
        )

        # ---------------------------------------------------------
        # INVESTIGATION HISTORY
        #
        # History tells the agent what has already been tried.
        # Keep recent history.
        # ---------------------------------------------------------

        history = context.get(
            "investigation_history",
            [],
        )

        compact["investigation_history"] = (
            history[-20:]
        )

        # ---------------------------------------------------------
        # OBSERVATION RESULTS
        #
        # Recent evidence is important for adaptive reasoning.
        # Keep recent results but heavily compact large payloads.
        # ---------------------------------------------------------

        observations = context.get(
            "observation_results",
            [],
        )

        compact_observations = []

        for observation in observations[-6:]:

            item = {
                "execution_index": observation.get(
                    "execution_index"
                ),
                "tool_name": observation.get(
                    "tool_name"
                ),
                "target_columns": observation.get(
                    "target_columns",
                    [],
                ),
                "success": observation.get(
                    "success"
                ),
                "status": observation.get(
                    "status"
                ),
            }

            if observation.get(
                "success"
            ):

                item["result"] = (
                    self._limit_value(
                        observation.get(
                            "result"
                        ),
                        max_chars=1400,
                    )
                )

            else:

                item["error"] = (
                    self._limit_value(
                        observation.get(
                            "error"
                        ),
                        max_chars=400,
                    )
                )

            compact_observations.append(
                item
            )

        compact["observation_results"] = (
            compact_observations
        )

        # ---------------------------------------------------------
        # PREVIOUS OBSERVATIONS
        #
        # Keep lightweight history only.
        # Do NOT duplicate result payloads.
        # ---------------------------------------------------------

        previous = context.get(
            "previous_observations",
            [],
        )

        compact_previous = []

        for observation in previous[-20:]:

            item = {
                "tool_name": observation.get(
                    "tool_name"
                ),
                "target_columns": observation.get(
                    "target_columns",
                    [],
                ),
                "success": observation.get(
                    "success"
                ),
                "status": observation.get(
                    "status"
                ),
            }

            if not observation.get(
                "success",
                True,
            ):

                item["error"] = (
                    self._limit_value(
                        observation.get(
                            "error"
                        ),
                        max_chars=250,
                    )
                )

            compact_previous.append(
                item
            )

        compact["previous_observations"] = (
            compact_previous
        )

        # ---------------------------------------------------------
        # HYPOTHESES
        #
        # Preserve hypothesis state if the context provides it.
        # This prevents the agent from losing its investigation
        # reasoning across steps.
        # ---------------------------------------------------------

        for key in (
            "hypotheses",
            "investigation_hypotheses",
            "hypothesis_tracking",
        ):

            if key in context:

                compact[key] = self._limit_value(
                    context[key],
                    max_chars=2500,
                )

        # ---------------------------------------------------------
        # STEP / BUDGET INFORMATION
        #
        # Preserve these if Agent Context provides them.
        # This helps the model decide when to finish.
        # ---------------------------------------------------------

        for key in (
            "current_step",
            "step",
            "max_steps",
            "remaining_steps",
            "investigation_budget",
        ):

            if key in context:
                compact[key] = context[key]

        return compact

    # =============================================================
    # VALUE COMPACTION
    # =============================================================

    def _limit_value(
        self,
        value: Any,
        max_chars: int = 2000,
    ) -> Any:
        """
        Limit a potentially large analytical result while
        preserving JSON structure.
        """

        if value is None:
            return None

        # ---------------------------------------------------------
        # Pydantic models
        # ---------------------------------------------------------

        if hasattr(
            value,
            "model_dump",
        ):

            value = value.model_dump()

        # ---------------------------------------------------------
        # Strings
        # ---------------------------------------------------------

        if isinstance(
            value,
            str,
        ):

            if len(value) <= max_chars:
                return value

            return (
                value[:max_chars]
                + "...[truncated]"
            )

        # ---------------------------------------------------------
        # Dictionaries
        # ---------------------------------------------------------

        if isinstance(
            value,
            dict,
        ):

            compact = {}
            current_size = 0

            for key, item in value.items():

                serialized_item = json.dumps(
                    item,
                    default=str,
                    ensure_ascii=False,
                )

                if (
                    current_size
                    + len(serialized_item)
                    > max_chars
                ):
                    break

                compact[str(key)] = (
                    self._limit_value(
                        item,
                        max_chars=600,
                    )
                )

                current_size += len(
                    serialized_item
                )

            return compact

        # ---------------------------------------------------------
        # Lists
        # ---------------------------------------------------------

        if isinstance(
            value,
            list,
        ):

            compact_list = []
            current_size = 0

            for item in value:

                serialized_item = json.dumps(
                    item,
                    default=str,
                    ensure_ascii=False,
                )

                if (
                    current_size
                    + len(serialized_item)
                    > max_chars
                ):
                    break

                compact_list.append(
                    self._limit_value(
                        item,
                        max_chars=450,
                    )
                )

                current_size += len(
                    serialized_item
                )

            return compact_list

        # ---------------------------------------------------------
        # Tuples
        # ---------------------------------------------------------

        if isinstance(
            value,
            tuple,
        ):

            return self._limit_value(
                list(value),
                max_chars=max_chars,
            )

        # ---------------------------------------------------------
        # Primitive values
        # ---------------------------------------------------------

        return value