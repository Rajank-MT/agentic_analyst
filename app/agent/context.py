from typing import Any

from app.data.model import (
    AgentObservation,
    AgentState,
)

from app.agent.investigation import build_investigation_summary


def build_agent_context(
    state: AgentState,
    observations: list[AgentObservation],
) -> dict[str, Any]:
    """
    Build a compact, structured context for the LLM.

    The LLM receives:
        - user objective
        - dataset names
        - metric/semantic roles
        - available analytical tools
        - previous observations

    The raw DataFrame is intentionally NOT included.
    Deterministic tools operate on the DataFrame.
    """

    return {
    "user_goal": state.user_goal,
    "datasets": state.dataset_names,

    "metrics": [
        {
            "column_name": profile.column_name,
            "role": profile.role.value,
            "confidence": profile.confidence,
            "signals": profile.signals,
        }
        for profile in state.metric_profiles
    ],

    "available_tools": [
        {
            "name": tool.name,
            "description": tool.description,
            "analysis_type": tool.analysis_type.value,
            "required_roles": [
                role.value
                for role in tool.required_roles
            ],
        }
        for tool in state.available_tools
    ],

    "investigation_summary": build_investigation_summary(
    observations
    ),


    "investigation_history": [
        {
            "execution_index": index,
            "tool_name": observation.tool_name,
            "target_columns": observation.target_columns,
            "success": observation.success,
            "status": "success" if observation.success else "failed",
        }
        for index, observation in enumerate(observations)
    ],

    "observation_results": [
        {
            "execution_index": index,
            "tool_name": observation.tool_name,
            "target_columns": observation.target_columns,
            "success": observation.success,
            "status": "success" if observation.success else "failed",
            "result": _serialize_result(
                observation.result
            ),
            "error": observation.error,
        }
        for index, observation in enumerate(observations)
    ],

    # Backward compatibility with existing pipeline tests
    "previous_observations": [
        {
            "tool_name": observation.tool_name,
            "target_columns": observation.target_columns,
            "success": observation.success,
            "status": "success" if observation.success else "failed",
            "result": _serialize_result(
                observation.result
            ),
            "error": observation.error,
        }
        for observation in observations
    ],
}


def _serialize_result(result: Any) -> Any:
    """
    Convert analytical result objects into
    JSON-compatible Python structures.
    """

    if result is None:
        return None

    if hasattr(result, "model_dump"):
        return result.model_dump()

    if isinstance(result, dict):
        return {
            str(key): _serialize_result(value)
            for key, value in result.items()
        }

    if isinstance(result, list):
        return [
            _serialize_result(item)
            for item in result
        ]

    if isinstance(result, tuple):
        return [
            _serialize_result(item)
            for item in result
        ]

    return result

def _build_investigation_history(observations):
    history = []

    for index, observation in enumerate(observations):
        history.append(
            {
                "execution_index": index,
                "tool_name": observation.tool_name,
                "target_columns": observation.target_columns,
                "success": observation.success,
            }
        )

    return history


def _build_observation_results(observations):
    results = []

    for index, observation in enumerate(observations):
        if not observation.success:
            results.append(
                {
                    "execution_index": index,
                    "tool_name": observation.tool_name,
                    "target_columns": observation.target_columns,
                    "status": "failed",
                    "error": observation.error,
                }
            )
            continue

        results.append(
            {
                "execution_index": index,
                "tool_name": observation.tool_name,
                "target_columns": observation.target_columns,
                "status": "success",
                "result": _serialize_result(observation.result),
            }
        )

    return results