from typing import Any

from app.data.model import (
    AgentObservation,
    AgentState,
    MetricRole,
)

from app.agent.investigation import build_investigation_summary


def build_agent_context(
    state: AgentState,
    observations: list[AgentObservation],
) -> dict[str, Any]:
    """
    Build structured context for the reasoning engine.

    The context helps the LLM reason about the dataset as a whole
    instead of repeatedly following the most recent investigation.
    """

    metrics = [
        {
            "column_name": profile.column_name,
            "role": profile.role.value,
            "confidence": profile.confidence,
            "signals": profile.signals,
        }
        for profile in state.metric_profiles
    ]

    investigation_coverage = _build_investigation_coverage(
        state,
        observations,
    )

    return {
        "user_goal": state.user_goal,
        "datasets": state.dataset_names,

        "metrics": metrics,

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

        "investigation_coverage": investigation_coverage,

        "investigation_history": [
            {
                "execution_index": index,
                "tool_name": observation.tool_name,
                "target_columns": observation.target_columns,
                "success": observation.success,
                "status": (
                    "success"
                    if observation.success
                    else "failed"
                ),
            }
            for index, observation in enumerate(observations)
        ],

        "observation_results": [
            {
                "execution_index": index,
                "tool_name": observation.tool_name,
                "target_columns": observation.target_columns,
                "success": observation.success,
                "status": (
                    "success"
                    if observation.success
                    else "failed"
                ),
                "result": _serialize_result(
                    observation.result
                ),
                "error": observation.error,
            }
            for index, observation in enumerate(observations)
        ],

        # Backward compatibility
        "previous_observations": [
            {
                "tool_name": observation.tool_name,
                "target_columns": observation.target_columns,
                "success": observation.success,
                "status": (
                    "success"
                    if observation.success
                    else "failed"
                ),
                "result": _serialize_result(
                    observation.result
                ),
                "error": observation.error,
            }
            for observation in observations
        ],
    }


def _build_investigation_coverage(
    state: AgentState,
    observations: list[AgentObservation],
) -> dict[str, Any]:
    """
    Describe which dataset elements have and have not been explored.

    This is generic and works with arbitrary datasets.
    """

    all_columns = {
        profile.column_name
        for profile in state.metric_profiles
    }

    all_measures = {
        profile.column_name
        for profile in state.metric_profiles
        if profile.role.value == MetricRole.MEASURE.value
    }

    all_dimensions = {
        profile.column_name
        for profile in state.metric_profiles
        if profile.role.value == MetricRole.DIMENSION.value
    }

    all_time_columns = {
        profile.column_name
        for profile in state.metric_profiles
        if profile.role.value == MetricRole.TIME.value
    }

    investigated_columns: set[str] = set()
    investigated_tools: set[str] = set()

    for observation in observations:
        investigated_tools.add(
            observation.tool_name
        )

        investigated_columns.update(
            observation.target_columns
        )

    investigated_measures = (
        investigated_columns & all_measures
    )

    investigated_dimensions = (
        investigated_columns & all_dimensions
    )

    investigated_time_columns = (
        investigated_columns & all_time_columns
    )

    return {
        "total_columns": len(all_columns),

        "investigated_columns": sorted(
            investigated_columns
        ),

        "uninvestigated_columns": sorted(
            all_columns - investigated_columns
        ),

        "measures": {
            "all": sorted(all_measures),
            "investigated": sorted(
                investigated_measures
            ),
            "uninvestigated": sorted(
                all_measures - investigated_measures
            ),
        },

        "dimensions": {
            "all": sorted(all_dimensions),
            "investigated": sorted(
                investigated_dimensions
            ),
            "uninvestigated": sorted(
                all_dimensions - investigated_dimensions
            ),
        },

        "time_columns": {
            "all": sorted(all_time_columns),
            "investigated": sorted(
                investigated_time_columns
            ),
            "uninvestigated": sorted(
                all_time_columns
                - investigated_time_columns
            ),
        },

        "investigated_tools": sorted(
            investigated_tools
        ),
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


def _build_investigation_history(
    observations,
):
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


def _build_observation_results(
    observations,
):
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
                "result": _serialize_result(
                    observation.result
                ),
            }
        )

    return results