from typing import Any

from app.data.model import AgentObservation


def build_investigation_summary(
    observations: list[AgentObservation],
) -> dict[str, Any]:
    """
    Build a deterministic summary of the investigation performed so far.

    This summary helps the reasoning LLM understand what has already
    been explored without asking the LLM to calculate or infer facts.
    """

    successful_analyses = []
    failed_analyses = []
    explored_columns = set()
    tool_usage = {}

    for index, observation in enumerate(observations):
        tool_usage[observation.tool_name] = (
            tool_usage.get(observation.tool_name, 0) + 1
        )

        explored_columns.update(observation.target_columns)

        analysis_record = {
            "execution_index": index,
            "tool_name": observation.tool_name,
            "target_columns": observation.target_columns,
        }

        if observation.success:
            successful_analyses.append(analysis_record)
        else:
            analysis_record["error"] = observation.error
            failed_analyses.append(analysis_record)

    return {
        "total_analyses": len(observations),
        "successful_analyses": successful_analyses,
        "failed_analyses": failed_analyses,
        "explored_columns": sorted(explored_columns),
        "tool_usage": tool_usage,
    }