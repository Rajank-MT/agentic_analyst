import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.agent.context import build_agent_context
from app.agent.llm import AgentLLM
from app.data.model import AgentAction, AgentObservation, AgentState


# Load the project .env regardless of the process's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class AzureOpenAIAgentLLM(AgentLLM):
    """
    Azure OpenAI reasoning engine for the Agentic Data Analyst.

    The model selects the next investigation action. Deterministic Python
    tools perform calculations; the LLM must not calculate data statistics.
    """

    def __init__(self, model: str | None = None) -> None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        deployment = (
            model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        ).strip()

        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_DEPLOYMENT", deployment),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing Azure OpenAI environment variables: "
                + ", ".join(missing)
            )

        # Supports a Foundry project endpoint such as:
        # https://<resource>.services.ai.azure.com/api/projects/<project>
        # or an endpoint already ending in /openai/v1.
        if endpoint.endswith("/openai/v1"):
            base_url = endpoint + "/"
        else:
            base_url = endpoint + "/openai/v1/"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=90.0,
            max_retries=2,
        )
        # For Azure's OpenAI-compatible endpoint, `model` is the deployment
        # name configured in Azure AI Foundry, not necessarily a model ID.
        self.model = deployment

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        if not content or not content.strip():
            raise ValueError("Azure OpenAI returned an empty response.")

        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Azure OpenAI returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Azure OpenAI response must be a JSON object."
            )
        return parsed

    @staticmethod
    def _normalize_action(raw_action: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize known LLM formatting deviations without changing the
        selected action or inventing a hypothesis.

        The application's prompt permits hypothesis=null or a structured
        hypothesis object. If the object has hypothesis=null, treat that as
        no hypothesis and normalize the entire field to null.
        """
        hypothesis = raw_action.get("hypothesis")

        if isinstance(hypothesis, str):
            # Older/less constrained model responses sometimes return text.
            # An empty string means no meaningful hypothesis was supplied.
            text = hypothesis.strip()
            if not text:
                raw_action["hypothesis"] = None
            else:
                raw_action["hypothesis"] = {
                    "hypothesis": text,
                    "supporting_execution_indexes": [],
                    "status": "active",
                    "conclusion": None,
                }

        elif isinstance(hypothesis, dict):
            hypothesis_text = hypothesis.get("hypothesis")
            if not isinstance(hypothesis_text, str) or not hypothesis_text.strip():
                # This is the exact failure mode previously observed:
                # {"hypothesis": {"hypothesis": null, ...}}
                raw_action["hypothesis"] = None
            else:
                if hypothesis.get("supporting_execution_indexes") is None:
                    hypothesis["supporting_execution_indexes"] = []
                hypothesis.setdefault("supporting_execution_indexes", [])
                hypothesis.setdefault("status", "active")
                hypothesis.setdefault("conclusion", None)
                raw_action["hypothesis"] = hypothesis

        elif hypothesis is not None:
            # Unsupported hypothesis shape. Fail clearly rather than
            # silently converting an arbitrary value.
            raise ValueError(
                "The 'hypothesis' field must be null, a string, or an object."
            )

        # Missing hypothesis is equivalent to no hypothesis for this prompt.
        elif "hypothesis" not in raw_action:
            raw_action["hypothesis"] = None

        return raw_action

    def decide(
        self,
        state: AgentState,
        observations: list[AgentObservation],
    ) -> AgentAction:
        full_context = build_agent_context(
            state=state,
            observations=observations,
        )
        context = self._compact_context(full_context)
        context_json = self._compact_json(context)

        print("\n" + "=" * 70)
        print("AZURE OPENAI AGENT DECISION")
        print("=" * 70)
        print(f"Context characters : {len(context_json):,}")
        print(f"Approx context tokens: {len(context_json) // 4:,}")

        
        prompt = f"""
        You are the reasoning engine of an autonomous Agentic Data Analyst.

        Decide the NEXT useful investigation. You do NOT calculate statistics.
        Deterministic Python tools and the Analysis Engine perform calculations.
        Use only the supplied context.

        DATASET SAMPLE:
        The context may include an adaptive sample. Use it only to understand
        entities, dimensions, measures, terminology, and possible areas to inspect.
        Never use sample rows for numerical claims. Quantitative evidence must come
        from deterministic analysis results.

        BUSINESS REASONING:
        Prioritize meaningful evidence-backed findings about performance, revenue,
        cost, profitability, customer/product/channel/geographic behavior, time
        patterns, operational efficiency, concentration, anomalies, and relationships
        when the dataset supports them. Do not analyze a column just because it is
        unexplored. Prefer business value over maximum coverage.

        ADAPTIVE INVESTIGATION:
        Balance depth and breadth. Deepen a finding only when further evidence could
        explain, validate, challenge, or strengthen it. Switch business areas when
        the current path becomes repetitive or a different area has greater potential
        for independent business value. Avoid repeating the same tool on the same
        columns unless there is a clear reason. Finish only when evidence is sufficient,
        remaining investigation budget is exhausted, or no useful action remains.

        EVIDENCE RULES:
        - Use only dataset metadata, metric profiles, successful tool results,
        investigation history, hypotheses, and other supplied context.
        - Never invent facts, numbers, columns, tools, operations, or results.
        - Do not write Python or SQL.
        - Do not calculate statistics yourself.
        - Do not claim causation from correlation.
        - Separate evidence, interpretation, and hypotheses.
        - Use exact tool names and column names supplied in the context.

        ANALYTICAL PATHS:
        1. run_tool: use one existing tool when it can answer the next business
        question. tool_call must contain tool_name, target_columns, reasoning.
        2. run_composed_analysis: use only when no single existing tool adequately
        answers the question. composed_analysis must contain reasoning and
        analysis_request. Supported aggregations: sum, mean, median, min, max,
        count. Supported derived operations: ratio, difference, percentage_of_total.
        Ranking fields rank_by, rank_ascending, top_n are optional. Use only real
        columns and supported operations. metrics must be a list of objects.
        3. finish: use when further investigation is unlikely to materially improve
        the conclusions.


        TOOL SELECTION RULES:
            - Follow each tool's required target-column count and column roles
            exactly as specified in the available tool registry.
            - Never pass extra columns to a tool.
            - Select columns based on their semantic types and the tool's requirements.
            - If a tool requires one column, provide exactly one.
            - If a tool requires two columns, provide exactly two compatible columns.
            - If no valid column combination exists, choose another useful tool
            or finish if no useful investigation remains.
            - After a tool fails validation, do not repeat the same invalid action.

        OUTPUT CONTRACT:
        Return exactly one valid JSON object matching AgentAction.
        Always include:
        - action_type
        - reason
        - tool_call
        - composed_analysis
        - hypothesis

        HYPOTHESIS GENERATION:
        - For every run_tool or run_composed_analysis action, generate
        a specific, testable hypothesis that explains what pattern or
        relationship the next investigation is intended to examine.
        - Base the hypothesis on the supplied dataset metadata, metric profiles,
        business goal, and previous observations.
        - Do not invent numerical claims or present an untested hypothesis
        as an established fact.
        - A hypothesis should describe something that could be supported,
        weakened, or rejected by the selected analysis.
        - Check existing hypotheses in the context first. Do not create duplicates.
        Continue investigating an existing hypothesis when appropriate.
        - Use status "active" for a newly proposed hypothesis.
        - For a new hypothesis, use an empty supporting_execution_indexes list
        and null conclusion.
        - When evidence is available, update the relevant existing hypothesis
        using the evidence and execution indexes. Do not mark it supported,
        weakened, or rejected without sufficient evidence.
        - For finish, return hypothesis as null unless an existing hypothesis
        needs a meaningful status or conclusion update.

        For run_tool, return an object with this structure:
        {{
        "action_type": "run_tool",
        "reason": "Why this is the best next investigation.",
        "tool_call": {{
            "tool_name": "exact available tool name",
            "target_columns": ["exact column name"],
            "reasoning": "Why this tool and these columns are useful."
        }},
        "composed_analysis": null,
        "hypothesis": {{
            "hypothesis": "A specific, testable assumption relevant to this analysis.",
            "supporting_execution_indexes": [],
            "status": "active",
            "conclusion": null
        }}
        }}

        For run_composed_analysis, return an object with this structure:
        {{
        "action_type": "run_composed_analysis",
        "reason": "Why this analysis is needed.",
        "tool_call": null,
        "composed_analysis": {{
            "reasoning": "Why existing tools are insufficient.",
            "analysis_request": {{
            "analysis_name": "short_name",
            "purpose": "business question",
            "group_by": ["Category"],
            "metrics": [{{"column": "Revenue", "aggregation": "sum"}}],
            "derived_metrics": []
            }}
        }},
        "hypothesis": {{
            "hypothesis": "A specific, testable assumption relevant to this analysis.",
            "supporting_execution_indexes": [],
            "status": "active",
            "conclusion": null
        }}
        }}

        For finish, return an object with this structure:
        {{
        "action_type": "finish",
        "reason": "Why the evidence is sufficient or no useful investigation remains.",
        "tool_call": null,
        "composed_analysis": null,
        "hypothesis": null
        }}

        The hypothesis object must contain exactly these fields:
        - hypothesis: non-empty string
        - supporting_execution_indexes: list of integer execution indexes
        - status: active, supported, weakened, or rejected
        - conclusion: string or null

        Use only exact available tool names, real columns, and supported
        operations. For composed analysis, supported aggregations are sum,
        mean, median, min, max, count. Supported derived operations are ratio,
        difference, percentage_of_total. Metrics must be a list of objects.

        Do not include markdown or commentary outside JSON.

        {context_json}
        """


        print(f"Prompt characters  : {len(prompt):,}")
        print(f"Approx prompt tokens: {len(prompt) // 4:,}")
        print("=" * 70)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an autonomous data-analysis reasoning "
                            "engine. Return only a JSON object matching the "
                            "AgentAction contract. Never invent tool results."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=2500,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Azure OpenAI decision request failed: {exc}"
            ) from exc

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Azure OpenAI returned an empty decision response.")

        print("\nAZURE OPENAI RAW DECISION:")
        print(content)

        try:
            raw_action = self._parse_json(content)
            raw_action = self._normalize_action(raw_action)
            action = AgentAction.model_validate(raw_action)
        except Exception as exc:
            raise RuntimeError(
                "Azure OpenAI returned invalid AgentAction JSON: "
                f"{exc}\nRaw response: {content}"
            ) from exc

        print("\nAZURE OPENAI VALIDATED DECISION:")
        print(action.model_dump())
        print("=" * 70)
        return action

    def _compact_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Keep decision-relevant metadata and recent evidence while reducing
        token usage. This follows the compact-context strategy in groq.py.
        """
        compact: dict[str, Any] = {
            "user_goal": context.get("user_goal"),
            "datasets": context.get("datasets", []),
        }

        compact["metrics"] = [
            {
                "column_name": metric.get("column_name"),
                "role": metric.get("role"),
                "confidence": metric.get("confidence"),
                "signals": metric.get("signals", [])[:5],
            }
            for metric in context.get("metrics", [])
            if isinstance(metric, dict)
        ]

        dataset_sample = context.get("dataset_sample")
        if dataset_sample:
            compact["dataset_sample"] = self._limit_value(
                dataset_sample, max_chars=4500
            )

        compact_tools = []
        for tool in context.get("available_tools", []):
            if not isinstance(tool, dict):
                continue
            description = str(tool.get("description", ""))
            if len(description) > 220:
                description = description[:220] + "..."
            item = {
                "name": tool.get("name"),
                "description": description,
                "analysis_type": tool.get("analysis_type"),
                "required_roles": tool.get("required_roles", []),
            }
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

        compact["investigation_summary"] = self._limit_value(
            context.get("investigation_summary"), max_chars=3000
        )
        compact["investigation_coverage"] = context.get(
            "investigation_coverage"
        )
        compact["investigation_history"] = context.get(
            "investigation_history", []
        )[-20:]

        observations = context.get("observation_results", [])
        compact["observation_results"] = [
            self._compact_observation(item)
            for item in observations[-6:]
            if isinstance(item, dict)
        ]

        previous = context.get("previous_observations", [])
        compact["previous_observations"] = [
            self._compact_observation(item, include_result=False)
            for item in previous[-20:]
            if isinstance(item, dict)
        ]

        for key in (
            "hypotheses",
            "investigation_hypotheses",
            "hypothesis_tracking",
            "current_step",
            "step",
            "max_steps",
            "remaining_steps",
            "investigation_budget",
        ):
            if key in context:
                compact[key] = self._limit_value(
                    context[key], max_chars=2500
                )

        return compact

    def _compact_observation(
        self,
        observation: dict[str, Any],
        include_result: bool = True,
    ) -> dict[str, Any]:
        item = {
            key: observation.get(key)
            for key in (
                "execution_index",
                "tool_name",
                "target_columns",
                "success",
                "status",
            )
            if key in observation
        }
        if include_result and observation.get("success"):
            item["result"] = self._limit_value(
                observation.get("result"), max_chars=1400
            )
        elif not observation.get("success"):
            item["error"] = self._limit_value(
                observation.get("error"), max_chars=400
            )
        return item

    def _limit_value(self, value: Any, max_chars: int = 2000) -> Any:
        """Limit large values while preserving a JSON-compatible structure."""
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            value = value.model_dump()

        if isinstance(value, str):
            return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"

        if isinstance(value, dict):
            result = {}
            current_size = 0
            for key, item in value.items():
                serialized = json.dumps(
                    item, default=str, ensure_ascii=False
                )
                if current_size + len(serialized) > max_chars:
                    break
                result[str(key)] = self._limit_value(item, max_chars=600)
                current_size += len(serialized)
            return result

        if isinstance(value, (list, tuple)):
            result = []
            current_size = 0
            for item in value:
                serialized = json.dumps(
                    item, default=str, ensure_ascii=False
                )
                if current_size + len(serialized) > max_chars:
                    break
                result.append(self._limit_value(item, max_chars=450))
                current_size += len(serialized)
            return result

        return value
