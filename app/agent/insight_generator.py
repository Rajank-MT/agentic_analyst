from abc import ABC, abstractmethod

from app.data.model import (
    AgentObservation,
    AgentState,
    BusinessInsightReport,
)


class InsightGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        state: AgentState,
        observations: list[AgentObservation],
    ) -> BusinessInsightReport:
        """Generate structured business insights from agent observations."""
        pass