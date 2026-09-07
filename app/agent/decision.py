import json
from typing import Any

from pydantic import ValidationError

from app.data.model import AgentAction


def parse_agent_action(response: Any) -> AgentAction:
    """
    Convert an LLM response into a validated AgentAction.

    Supported input:
        - AgentAction
        - Python dictionary
        - JSON string

    Raises:
        ValueError: if the response is malformed or invalid.
    """

    # ---------------------------------------------------------
    # 1. Already a validated AgentAction
    # ---------------------------------------------------------

    if isinstance(response, AgentAction):
        return response

    # ---------------------------------------------------------
    # 2. Dictionary response
    # ---------------------------------------------------------

    if isinstance(response, dict):
        return _validate_action(response)

    # ---------------------------------------------------------
    # 3. JSON string response
    # ---------------------------------------------------------

    if isinstance(response, str):

        response = response.strip()

        if not response:
            raise ValueError(
                "LLM response is empty."
            )

        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM response is not valid JSON."
            ) from exc

        if not isinstance(parsed_response, dict):
            raise ValueError(
                "LLM JSON response must be an object."
            )

        return _validate_action(parsed_response)

    # ---------------------------------------------------------
    # 4. Unsupported response type
    # ---------------------------------------------------------

    raise ValueError(
        "Unsupported LLM response type."
    )


def _validate_action(
    data: dict[str, Any],
) -> AgentAction:
    """
    Validate a dictionary against the AgentAction schema.
    """

    try:
        return AgentAction.model_validate(data)

    except ValidationError as exc:
        raise ValueError(
            f"Invalid agent action: {exc}"
        ) from exc