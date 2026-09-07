from abc import ABC, abstractmethod

from app.data.model import (
    AgentAction,
    AgentState,
    AgentObservation,
)


class AgentLLM(ABC):

    @abstractmethod
    def decide(
        self,
        state: AgentState,
        observations: list[AgentObservation],
    ) -> AgentAction:
        pass