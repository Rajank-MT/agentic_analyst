from dataclasses import dataclass, field
from typing import Any

from app.agent.state import AgentState
from app.data.model import AgentObservation


@dataclass
class AgentSession:
    """
    Persistent state for one dataset-analysis conversation.

    The session survives after the initial autonomous analysis so
    follow-up questions can reuse the same investigation.
    """

    session_id: str
    state: AgentState

    dataframe: Any | None = None

    observations: list[AgentObservation] = field(
        default_factory=list
    )

    conversation_history: list[dict[str, str]] = field(
        default_factory=list
    )

    def add_user_message(self, message: str) -> None:
        self.conversation_history.append(
            {
                "role": "user",
                "message": message,
            }
        )

    def add_assistant_message(self, message: str) -> None:
        self.conversation_history.append(
            {
                "role": "assistant",
                "message": message,
            }
        )