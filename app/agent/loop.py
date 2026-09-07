from typing import Any

import pandas as pd

from app.agent.state import add_tool_execution, update_agent_status
from app.data.model import (
    AgentActionType,
    AgentObservation,
    AgentState,
    AgentStatus,
    ToolExecution,
    ToolRequest,
)
from app.data.tool_executor import execute_tool
from app.data.tool_validator import validate_tool_request

from app.agent.insight_generator import InsightGenerator


class AgentLoop:
    """
    Autonomous execution loop for the Agentic Data Analyst.

    Flow:

        Agent State
             ↓
            LLM
             ↓
        Agent Action
             ↓
          Validate
             ↓
        Execute Tool
             ↓
         Observation
             ↓
            LLM
             ↓
        Next Action
             ↓
            ...
             ↓
           Finish
    """

    def __init__(
        self,
        llm,
        registry,
        max_steps=10,
        insight_generator= None,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")

        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.insight_generator = insight_generator

    def _is_duplicate_tool_call(
        self,
        tool_name: str,
        target_columns: list[str],
        observations: list[AgentObservation],
    ) -> bool:
        """
        Check whether the exact same tool request has already been executed.

        Tool name and target columns must both match.
        """

        requested_columns = tuple(target_columns)

        for observation in observations:
            previous_columns = tuple(observation.target_columns)

            if (
                observation.tool_name == tool_name
                and previous_columns == requested_columns
            ):
                return True

        return False

    def _record_failed_execution(
        self,
        state: AgentState,
        observations: list[AgentObservation],
        tool_name: str,
        target_columns: list[str],
        error: str,
    ) -> None:
        """
        Store a failed tool attempt both as an observation
        and as an execution in Agent State.
        """

        observation = AgentObservation(
            tool_name=tool_name,
            target_columns=target_columns,
            success=False,
            error=error,
        )

        observations.append(observation)

        execution = ToolExecution(
            tool_name=tool_name,
            target_columns=target_columns,
            success=False,
            error=error,
        )

        add_tool_execution(
            state,
            execution,
        )

    def run(
        self,
        state: AgentState,
        dataframe: pd.DataFrame,
    ) -> AgentState:
        """
        Run the autonomous agent.

        The LLM decides what to analyze next.
        Deterministic tools perform the actual calculations.
        """

        observations: list[AgentObservation] = []

        update_agent_status(
            state,
            AgentStatus.PLANNING,
        )

        for _ in range(self.max_steps):

            # ---------------------------------------------------------
            # 1. Ask LLM for next action
            # ---------------------------------------------------------

            # 1. Ask LLM for next action
            try:
                action = self.llm.decide(
                    state=state,
                    observations=observations,
                )
            except Exception as exc:
                error_message = f"LLM decision failed: {exc}"

                execution = ToolExecution(
                    tool_name="llm_decision",
                    target_columns=[],
                    success=False,
                    error=error_message,
                )

                add_tool_execution(state, execution)

                update_agent_status(
                    state,
                    AgentStatus.FAILED,
                )

                return state

            # ---------------------------------------------------------
            # 2. Finish
            # ---------------------------------------------------------

            if action.action_type == AgentActionType.FINISH:

                if self.insight_generator is not None:
                    try:
                        report = self.insight_generator.generate(
                            state=state,
                            observations=observations,
                        )

                        state.insights = report.insights

                    except Exception as exc:
                        state.status = AgentStatus.FAILED

                        state.executions.append(
                            ToolExecution(
                                tool_name="insight_generation",
                                success=False,
                                error=str(exc),
                            )
                        )

                        return state

                state.status = AgentStatus.COMPLETED
                return state

            # ---------------------------------------------------------
            # 3. Validate action type
            # ---------------------------------------------------------

            if action.action_type != AgentActionType.RUN_TOOL:

                update_agent_status(
                    state,
                    AgentStatus.FAILED,
                )

                return state

            # ---------------------------------------------------------
            # 4. RUN_TOOL requires a tool call
            # ---------------------------------------------------------

            if action.tool_call is None:

                update_agent_status(
                    state,
                    AgentStatus.FAILED,
                )

                return state

            tool_call = action.tool_call

            tool_name = tool_call.tool_name
            target_columns = tool_call.target_columns

            # ---------------------------------------------------------
            # 5. Prevent exact duplicate tool calls
            # ---------------------------------------------------------

            if self._is_duplicate_tool_call(
                tool_name=tool_name,
                target_columns=target_columns,
                observations=observations,
            ):
                self._record_failed_execution(
                    state=state,
                    observations=observations,
                    tool_name=tool_name,
                    target_columns=target_columns,
                    error=(
                        "Duplicate tool call detected. "
                        "The same tool with the same target columns "
                        "has already been executed."
                    ),
                )

                update_agent_status(
                    state,
                    AgentStatus.ANALYZING,
                )

                continue

            # ---------------------------------------------------------
            # 6. Agent enters execution phase
            # ---------------------------------------------------------

            update_agent_status(
                state,
                AgentStatus.EXECUTING,
            )

            request = ToolRequest(
                tool_name=tool_name,
                target_columns=target_columns,
            )

            # ---------------------------------------------------------
            # 7. Validate tool request
            # ---------------------------------------------------------

            validation = validate_tool_request(
                request=request,
                dataframe=dataframe,
                registry=self.registry,
                metric_profiles=state.metric_profiles,
            )

            # ---------------------------------------------------------
            # 8. Handle validation failure
            # ---------------------------------------------------------

            if not validation.valid:

                error_message = "; ".join(
                    validation.errors
                )

                self._record_failed_execution(
                    state=state,
                    observations=observations,
                    tool_name=tool_name,
                    target_columns=target_columns,
                    error=error_message,
                )

                # Give the LLM another opportunity to recover.
                update_agent_status(
                    state,
                    AgentStatus.ANALYZING,
                )

                continue

            # ---------------------------------------------------------
            # 9. Execute deterministic tool
            # ---------------------------------------------------------

            try:

                result = execute_tool(
                    tool_name=tool_name,
                    dataframe=dataframe,
                    target_columns=target_columns,
                )

                observation = AgentObservation(
                    tool_name=tool_name,
                    target_columns=target_columns,
                    success=True,
                    result=result,
                )

                execution = ToolExecution(
                    tool_name=tool_name,
                    target_columns=target_columns,
                    success=True,
                    result=result,
                )

            # ---------------------------------------------------------
            # 10. Handle execution failure
            # ---------------------------------------------------------

            except Exception as exc:

                error_message = str(exc)

                observation = AgentObservation(
                    tool_name=tool_name,
                    target_columns=target_columns,
                    success=False,
                    error=error_message,
                )

                execution = ToolExecution(
                    tool_name=tool_name,
                    target_columns=target_columns,
                    success=False,
                    error=error_message,
                )

            # ---------------------------------------------------------
            # 11. Store observation
            # ---------------------------------------------------------

            observations.append(
                observation
            )

            # ---------------------------------------------------------
            # 12. Store execution in Agent State
            # ---------------------------------------------------------

            add_tool_execution(
                state,
                execution,
            )

            # ---------------------------------------------------------
            # 13. Return control to LLM
            # ---------------------------------------------------------

            update_agent_status(
                state,
                AgentStatus.ANALYZING,
            )

        # -------------------------------------------------------------
        # 14. Maximum steps reached
        # -------------------------------------------------------------
        
        update_agent_status(
            state,
            AgentStatus.MAX_STEPS_REACHED,
            
        )

        execution = ToolExecution(
            tool_name="agent_loop",
            target_columns=[],
            success=False,
            error=(
                f"Maximum reasoning steps ({self.max_steps}) "
                "reached before the agent returned FINISH."
            ),
        )

        add_tool_execution(
            state,
            execution,
        )

        return state