from app.data.model import (
    AgentObservation,
    BusinessInsightReport,
)


def validate_insight_report(
    report: BusinessInsightReport,
    observations: list[AgentObservation],
) -> list[str]:
    errors = []

    if not report.insights:
        errors.append("Insight report contains no insights.")
        return errors

    observation_count = len(observations)

    for insight_index, insight in enumerate(report.insights):
        if not insight.evidence:
            errors.append(
                f"Insight {insight_index} has no supporting evidence."
            )
            continue

        for evidence in insight.evidence:
            execution_index = evidence.execution_index

            if execution_index >= observation_count:
                errors.append(
                    f"Insight {insight_index} references invalid "
                    f"execution index {execution_index}."
                )
                continue

            observation = observations[execution_index]

            if not observation.success:
                errors.append(
                    f"Insight {insight_index} references failed "
                    f"execution {execution_index}."
                )

    return errors