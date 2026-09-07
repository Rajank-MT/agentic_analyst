from app.agent.llm import AgentLLM
from app.data.model import (
    AgentAction,
    AgentActionType,
    AgentState,
)


class MockAgentLLM(AgentLLM):

    def __init__(self, action: AgentAction):
        self.action = action

    def decide(self, state, observations):
        return self.action