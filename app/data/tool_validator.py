import pandas as pd

from app.data.model import (
    AnalysisToolRegistry,
    MetricProfile,
    ToolRequest,
    ToolValidationResult,
)


def validate_tool_request(
    request: ToolRequest,
    dataframe: pd.DataFrame,
    registry: AnalysisToolRegistry,
    metric_profiles: list[MetricProfile] | None = None,
) -> ToolValidationResult:

    errors = []

    tool = next(
        (
            tool
            for tool in registry.tools
            if tool.name == request.tool_name
        ),
        None,
    )

    if tool is None:
        return ToolValidationResult(
            valid=False,
            tool_name=request.tool_name,
            target_columns=request.target_columns,
            errors=[f"Unknown tool: {request.tool_name}"],
        )

    # Check target count.
    target_count = len(request.target_columns)

    if not (
    tool.min_targets
    <= target_count
    <= tool.max_targets
):
       
       if tool.min_targets == tool.max_targets:
        errors.append(
            f"{request.tool_name} requires "
            f"{tool.min_targets} target column(s)."
        )
       else:

        errors.append(
            f"{request.tool_name} requires between "
            f"{tool.min_targets} and {tool.max_targets} "
            f"target column(s)."
        )

    # Check that requested columns exist.
    missing_columns = [
        column
        for column in request.target_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        errors.append(
            f"Columns not found: {missing_columns}"
        )

    if errors:
        return ToolValidationResult(
            valid=False,
            tool_name=request.tool_name,
            target_columns=request.target_columns,
            errors=errors,
        )

    # Validate semantic roles.
    if metric_profiles is not None:

        profile_map = {
            profile.column_name: profile
            for profile in metric_profiles
        }

        required_roles = tool.required_roles

        # Case 1:
        # One required role can apply to multiple target columns.
        if len(required_roles) == 1:

            required_role = required_roles[0]

            for column in request.target_columns:

                profile = profile_map.get(column)

                if profile is None:
                    errors.append(
                        f"No metric profile found for column: {column}"
                    )
                    continue

                if profile.role != required_role:
                    errors.append(
                        f"Column '{column}' has role "
                        f"'{profile.role.value}' but "
                        f"'{request.tool_name}' requires "
                        f"'{required_role.value}'."
                    )

        # Case 2:
        # Each target position has a specific role.
        else:

            if len(request.target_columns) != len(required_roles):
                errors.append(
                    f"{request.tool_name} requires roles: "
                    f"{[role.value for role in required_roles]}"
                )
            else:

                for column, required_role in zip(
                    request.target_columns,
                    required_roles,
                ):

                    profile = profile_map.get(column)

                    if profile is None:
                        errors.append(
                            f"No metric profile found for column: {column}"
                        )
                        continue

                    if profile.role != required_role:
                        errors.append(
                            f"Column '{column}' has role "
                            f"'{profile.role.value}' but "
                            f"'{request.tool_name}' requires "
                            f"'{required_role.value}'."
                        )

    return ToolValidationResult(
        valid=len(errors) == 0,
        tool_name=request.tool_name,
        target_columns=request.target_columns,
        errors=errors,
    )