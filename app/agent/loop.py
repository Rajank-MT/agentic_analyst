from typing import Any

import pandas as pd

from app.agent.state import add_tool_execution, update_agent_status
from app.data.analysis_executor import AnalysisExecutor
from app.data.analysis_request import AnalysisRequest
from app.data.analysis_validator import AnalysisRequestValidator
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


class AgentLoop:
    """
    Autonomous execution loop for the Agentic Data Analyst.

    The LLM can:
        1. Run an existing analytical tool.
        2. Run a composed analysis.
        3. Finish the investigation.

    Existing tools and composed analyses are both deterministic.
    """

    def __init__(
        self,
        llm,
        registry,
        max_steps=10,
        insight_generator=None,
        session=None,
        analysis_validator=None,
        analysis_executor=None,
        batch_size=None,
    ):
      if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")

      if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

      self.llm = llm
      self.registry = registry
      self.max_steps = max_steps
      self.insight_generator = insight_generator
      self.session = session
      self.batch_size = batch_size
 
      self.analysis_validator = (
        analysis_validator or AnalysisRequestValidator()
    )

      self.analysis_executor = (
        analysis_executor or AnalysisExecutor()
    )

    def _is_duplicate_tool_call(
        self,
        tool_name: str,
        target_columns: list[str],
        observations: list[AgentObservation],
    ) -> bool:

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

        observation = AgentObservation(
            tool_name=tool_name,
            target_columns=target_columns,
            success=False,
            error=error,
        )

        observations.append(observation)

        if self.session is not None:
            self.session.observations.append(observation)

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

    def _generate_insights(
        self,
        state: AgentState,
        observations: list[AgentObservation],
    ) -> bool:

        if self.insight_generator is None:
            return True

        try:
            report = self.insight_generator.generate(
                state=state,
                observations=observations,
            )

            state.insights = report.insights

            return True

        except Exception as exc:

            add_tool_execution(
                state,
                ToolExecution(
                    tool_name="insight_generation",
                    target_columns=[],
                    success=False,
                    error=str(exc),
                ),
            )

            return False

    def _execute_composed_analysis(
        self,
        action,
        dataframe: pd.DataFrame,
        observations: list[AgentObservation],
        state: AgentState,
    ) -> bool:

        if action.composed_analysis is None:

            self._record_failed_execution(
                state=state,
                observations=observations,
                tool_name="composed_analysis",
                target_columns=[],
                error="Missing composed analysis request.",
            )

            return False

        try:
            request = AnalysisRequest.model_validate(
                action.composed_analysis.analysis_request
            )

            errors = self.analysis_validator.validate(
                request=request,
                dataframe=dataframe,
            )

            if errors:

                self._record_failed_execution(
                    state=state,
                    observations=observations,
                    tool_name="composed_analysis",
                    target_columns=request.group_by,
                    error="; ".join(errors),
                )

                return False

            result = self.analysis_executor.execute(
                request=request,
                dataframe=dataframe,
            )

            observation = AgentObservation(
                tool_name="composed_analysis",
                target_columns=request.group_by,
                success=True,
                result=result,
            )

            execution = ToolExecution(
                tool_name="composed_analysis",
                target_columns=request.group_by,
                success=True,
                result=result,
            )

            observations.append(observation)

            if self.session is not None:
                self.session.observations.append(observation)

            add_tool_execution(
                state,
                execution,
            )

            return True

        except Exception as exc:

            self._record_failed_execution(
                state=state,
                observations=observations,
                tool_name="composed_analysis",
                target_columns=[],
                error=str(exc),
            )

            return False

    def run(
        self,
        state: AgentState,
        dataframe: pd.DataFrame,
    ) -> AgentState:
        """
        Run the autonomous agent.

        LLM decides what to investigate.
        Python performs deterministic calculations.
        """

        observations: list[AgentObservation] = []

        update_agent_status(
            state,
            AgentStatus.PLANNING,
        )

        for step in range(self.max_steps):

             # ---------------------------------------------------------
    # Batch pause
    # ---------------------------------------------------------
            if (
                self.batch_size
                and step > 0
                and step % self.batch_size == 0
            ):
                print()
                print("=" * 80)
                print(
                    f"PAUSED AFTER {step} INVESTIGATION STEPS"
                )
                print("=" * 80)

                input(
                    "Press ENTER to continue..."
                )

            # ---------------------------------------------------------
            # 1. Ask LLM for next action
            # ---------------------------------------------------------

            try:
                action = self.llm.decide(
                    state=state,
                    observations=observations,
                )

            except Exception as exc:

                self._record_failed_execution(
                    state=state,
                    observations=observations,
                    tool_name="llm_decision",
                    target_columns=[],
                    error=f"LLM decision failed: {exc}",
                )

                update_agent_status(
                    state,
                    AgentStatus.FAILED,
                )

                return state

            # ---------------------------------------------------------
            # 2. Store hypothesis
            # ---------------------------------------------------------

            if action.hypothesis is not None:
                state.hypotheses.append(
                    action.hypothesis
                )

            # ---------------------------------------------------------
            # 3. Finish
            # ---------------------------------------------------------

            if action.action_type == AgentActionType.FINISH:

                if not self._generate_insights(
                    state=state,
                    observations=observations,
                ):
                    update_agent_status(
                        state,
                        AgentStatus.FAILED,
                    )

                    return state

                update_agent_status(
                    state,
                    AgentStatus.COMPLETED,
                )

                return state

            # ---------------------------------------------------------
            # 4. Composed analysis
            # ---------------------------------------------------------

            if (
                action.action_type
                == AgentActionType.RUN_COMPOSED_ANALYSIS
            ):

                update_agent_status(
                    state,
                    AgentStatus.EXECUTING,
                )

                self._execute_composed_analysis(
                    action=action,
                    dataframe=dataframe,
                    observations=observations,
                    state=state,
                )

                update_agent_status(
                    state,
                    AgentStatus.ANALYZING,
                )

                continue

            # ---------------------------------------------------------
            # 5. Existing tool
            # ---------------------------------------------------------

            if action.action_type != AgentActionType.RUN_TOOL:

                self._record_failed_execution(
                    state=state,
                    observations=observations,
                    tool_name="agent_decision",
                    target_columns=[],
                    error=(
                        f"Unsupported action type: "
                        f"{action.action_type}"
                    ),
                )

                update_agent_status(
                    state,
                    AgentStatus.FAILED,
                )

                return state

            # ---------------------------------------------------------
            # 6. RUN_TOOL requires tool_call
            # ---------------------------------------------------------

            if action.tool_call is None:

                self._record_failed_execution(
                    state=state,
                    observations=observations,
                    tool_name="agent_decision",
                    target_columns=[],
                    error="RUN_TOOL requires tool_call.",
                )

                update_agent_status(
                    state,
                    AgentStatus.FAILED,
                )

                return state

            tool_call = action.tool_call

            tool_name = tool_call.tool_name
            target_columns = tool_call.target_columns

            # ---------------------------------------------------------
            # 7. Prevent duplicate tool calls
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
                        "The same tool with the same target "
                        "columns has already been executed."
                    ),
                )

                update_agent_status(
                    state,
                    AgentStatus.ANALYZING,
                )

                continue

            # ---------------------------------------------------------
            # 8. Execution phase
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
            # 9. Validate existing tool
            # ---------------------------------------------------------

            validation = validate_tool_request(
                request=request,
                dataframe=dataframe,
                registry=self.registry,
                metric_profiles=state.metric_profiles,
            )

            # ---------------------------------------------------------
            # 10. Handle validation failure
            # ---------------------------------------------------------

            if not validation.valid:

                self._record_failed_execution(
                    state=state,
                    observations=observations,
                    tool_name=tool_name,
                    target_columns=target_columns,
                    error="; ".join(validation.errors),
                )

                update_agent_status(
                    state,
                    AgentStatus.ANALYZING,
                )

                continue

            # ---------------------------------------------------------
            # 11. Execute existing deterministic tool
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

            except Exception as exc:

                observation = AgentObservation(
                    tool_name=tool_name,
                    target_columns=target_columns,
                    success=False,
                    error=str(exc),
                )

                execution = ToolExecution(
                    tool_name=tool_name,
                    target_columns=target_columns,
                    success=False,
                    error=str(exc),
                )

            # ---------------------------------------------------------
            # 12. Store observation
            # ---------------------------------------------------------

            observations.append(observation)

            if self.session is not None:
                self.session.observations.append(
                    observation
                )

            # ---------------------------------------------------------
            # 13. Store execution
            # ---------------------------------------------------------

            add_tool_execution(
                state,
                execution,
            )

            # ---------------------------------------------------------
            # 14. Return control to LLM
            # ---------------------------------------------------------

            update_agent_status(
                state,
                AgentStatus.ANALYZING,
            )

        # -------------------------------------------------------------
        # 15. Maximum steps reached
        # -------------------------------------------------------------

        update_agent_status(
            state,
            AgentStatus.MAX_STEPS_REACHED,
        )

        add_tool_execution(
            state,
            ToolExecution(
                tool_name="agent_loop",
                target_columns=[],
                success=False,
                error=(
                    f"Maximum reasoning steps ({self.max_steps}) "
                    "reached before the agent returned FINISH."
                ),
            ),
        )

        # -------------------------------------------------------------
        # 16. Generate partial insights
        # -------------------------------------------------------------

        if not self._generate_insights(
            state=state,
            observations=observations,
        ):
            update_agent_status(
                state,
                AgentStatus.FAILED,
            )

        return state