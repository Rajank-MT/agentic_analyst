from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from app.api.postgres import router as postgres_router

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.loop import AgentLoop
#from app.agent.providers.gemini import GeminiAgentLLM
#from app.agent.providers.gemini_insight import GeminiInsightGenerator
#from app.agent.providers.groq import GroqAgentLLM
#from app.agent.providers.groq_insight import GroqInsightGenerator
from app.agent.providers.azure_openai import AzureOpenAIAgentLLM
from app.agent.providers.azure_openai_insight import (
    AzureOpenAIInsightGenerator,
)

from app.agent.state import create_agent_state
from app.data.metrics import discover_metric_role
from app.data.profiling import profile_dataset
from app.data.sampling import build_adaptive_sample
from app.data.tool_registry import build_analysis_tool_registry
from app.api.postgres import router as postgres_router


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Agentic Data Analyst API",
    version="1.0.0",
)
app.include_router(postgres_router)

# =============================================================================
# CORS
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def make_json_safe(value: Any) -> Any:
    """
    Recursively convert analysis results into
    standard JSON-compatible Python objects.

    Handles:
        - Pydantic models
        - Enum values
        - NumPy scalars
        - NumPy arrays
        - dictionaries
        - lists / tuples / sets
        - pandas timestamps
        - NaN / Infinity
        - normal Python primitives
    """

    # -------------------------------------------------------------------------
    # None
    # -------------------------------------------------------------------------

    if value is None:
        return None

    # -------------------------------------------------------------------------
    # Pydantic models
    # -------------------------------------------------------------------------

    if isinstance(value, BaseModel):
        return make_json_safe(
            value.model_dump()
        )

    # -------------------------------------------------------------------------
    # Enum
    # -------------------------------------------------------------------------

    if isinstance(value, Enum):
        return make_json_safe(
            value.value
        )

    # -------------------------------------------------------------------------
    # NumPy scalar
    # -------------------------------------------------------------------------

    if isinstance(value, np.generic):
        return make_json_safe(
            value.item()
        )

    # -------------------------------------------------------------------------
    # NumPy array
    # -------------------------------------------------------------------------

    if isinstance(value, np.ndarray):
        return make_json_safe(
            value.tolist()
        )

    # -------------------------------------------------------------------------
    # pandas Timestamp / datetime-like values
    # -------------------------------------------------------------------------

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    # -------------------------------------------------------------------------
    # Dictionary
    # -------------------------------------------------------------------------

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    # -------------------------------------------------------------------------
    # List / tuple / set
    # -------------------------------------------------------------------------

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    # -------------------------------------------------------------------------
    # Float
    #
    # JSON does not reliably support NaN / Infinity.
    # Convert them to None.
    # -------------------------------------------------------------------------

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

        return value

    # -------------------------------------------------------------------------
    # Normal JSON-compatible values
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        (str, int, bool),
    ):
        return value

    # -------------------------------------------------------------------------
    # Last-resort conversion
    # -------------------------------------------------------------------------

    return str(value)


# =============================================================================
# SERIALIZATION HELPER
# =============================================================================

def _serialize(value: Any) -> Any:
    """
    Convert project models and nested structures into
    JSON-safe structures.

    The final response is passed through make_json_safe()
    as an additional protection against NumPy / pandas values.
    """

    if value is None:
        return None

    if isinstance(value, BaseModel):
        return value.model_dump()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {
            str(key): _serialize(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _serialize(item)
            for item in value
        ]

    return value


# =============================================================================
# BUILD API RESPONSE
# =============================================================================

def _build_response(
    state,
    dataframe,
    dataset_name,
):
    """
    Convert the final AgentState into a frontend-friendly
    response structure.

    This function does NOT modify the agent pipeline.
    """

    profiles = state.metric_profiles

    # -------------------------------------------------------------------------
    # Ignore entity columns when calculating investigation coverage.
    # -------------------------------------------------------------------------

    non_entity_profiles = [
        profile
        for profile in profiles
        if profile.role.value != "entity"
    ]

    # -------------------------------------------------------------------------
    # Determine which columns were successfully investigated.
    # -------------------------------------------------------------------------

    investigated_columns = {
        column
        for execution in state.executions
        if execution.success
        for column in execution.target_columns
    }

    available_columns = {
        profile.column_name
        for profile in non_entity_profiles
    }

    # -------------------------------------------------------------------------
    # Investigation coverage
    # -------------------------------------------------------------------------

    if available_columns:
        coverage = round(
            100
            * len(
                investigated_columns
                & available_columns
            )
            / len(available_columns),
            1,
        )
    else:
        coverage = 0.0

    # -------------------------------------------------------------------------
    # Investigation history
    # -------------------------------------------------------------------------

    investigations = []

    for index, execution in enumerate(
        state.executions
    ):
        investigations.append(
            {
                "execution_index": index,
                "tool_name": execution.tool_name,
                "target_columns": execution.target_columns,
                "success": execution.success,
                "result": _serialize(
                    execution.result
                ),
                "error": execution.error,
            }
        )

    # -------------------------------------------------------------------------
    # Final response
    # -------------------------------------------------------------------------

    response = {
        "status": state.status.value,

        "dataset": {
            "name": dataset_name,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        },

        "tools_available": len(
            state.available_tools
        ),

        "coverage": coverage,

        "investigated_columns": sorted(
            investigated_columns
        ),

        "metric_profiles": [
            {
                "column_name": profile.column_name,
                "role": profile.role.value,
                "confidence": profile.confidence,
                "signals": profile.signals,
            }
            for profile in profiles
        ],

        "investigations": investigations,

        "insights": [
            _serialize(insight)
            for insight in state.insights
        ],

        "hypotheses": [
            _serialize(hypothesis)
            for hypothesis in state.hypotheses
        ],
    }

    return response


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "agentic-data-analyst",
    }


# =============================================================================
# ANALYZE DATASET
# =============================================================================

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...)
):
    """
    Upload a CSV and run the existing
    autonomous Agentic Data Analyst pipeline.
    """

    # -------------------------------------------------------------------------
    # Validate filename
    # -------------------------------------------------------------------------

    filename = file.filename or "dataset.csv"

    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    # -------------------------------------------------------------------------
    # Read uploaded file
    # -------------------------------------------------------------------------

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    temp_path = None

    try:

        # ---------------------------------------------------------------------
        # Save uploaded CSV temporarily
        # ---------------------------------------------------------------------

        with NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as temp:

            temp.write(data)
            temp_path = Path(
                temp.name
            )

        # ---------------------------------------------------------------------
        # Load dataset
        # ---------------------------------------------------------------------

        dataframe = pd.read_csv(
            temp_path
        )

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV contains no rows.",
            )

        dataset_name = Path(
            filename
        ).stem

        # ---------------------------------------------------------------------
        # Dataset profiling
        # ---------------------------------------------------------------------

        dataset_profile = profile_dataset(
            dataframe=dataframe,
            dataset_name=dataset_name,
        )

        # ---------------------------------------------------------------------
        # Discover metric / semantic roles
        # ---------------------------------------------------------------------

        metric_profiles = [
            discover_metric_role(column)
            for column in dataset_profile.columns
        ]

        # ---------------------------------------------------------------------
        # Adaptive dataset understanding
        #
        # The sample is compact context for the LLM only. All quantitative
        # analysis continues to run on the complete dataframe.
        # ---------------------------------------------------------------------

        dataset_sample = build_adaptive_sample(
            dataframe=dataframe,
            random_sample_size=8,
            representative_row_count=8,
            categorical_example_count=8,
            rare_value_example_count=5,
        )

        # ---------------------------------------------------------------------
        # Build analysis tools
        # ---------------------------------------------------------------------

        registry = build_analysis_tool_registry()

        # ---------------------------------------------------------------------
        # Create agent state
        # ---------------------------------------------------------------------

        state = create_agent_state(
            user_goal=(
                "Automatically analyze this dataset and identify important, "
                "non-obvious, statistically supported business insights. "
                "Explore the dataset broadly rather than repeatedly investigating "
                "one measure. Use deterministic analysis results as evidence, "
                "prioritize valuable business findings, avoid redundant low-value "
                "comparisons, and do not invent facts or numbers."
            ),
            dataset_names=[
                dataset_name
            ],
            metric_profiles=metric_profiles,
            available_tools=registry.tools,
        )

        # Store only the compact adaptive sample in agent state.
        # Deterministic analysis still uses the complete dataframe.
        state.dataset_sample = dataset_sample.model_dump()

        # ---------------------------------------------------------------------
        # Initialize Gemini
        # ---------------------------------------------------------------------

        llm = AzureOpenAIAgentLLM()

        # ---------------------------------------------------------------------
        # Initialize insight generator
        # ---------------------------------------------------------------------

        insight_generator = (
             AzureOpenAIInsightGenerator()
        )

        # ---------------------------------------------------------------------
        # Initialize autonomous investigation loop
        # ---------------------------------------------------------------------

        loop = AgentLoop(
            llm=llm,
            registry=registry,
            max_steps=10,
            batch_size=5,
            insight_generator=insight_generator,
        )

        # ---------------------------------------------------------------------
        # Run existing autonomous pipeline
        # ---------------------------------------------------------------------

        final_state = loop.run(
            state=state,
            dataframe=dataframe,
        )

        # ---------------------------------------------------------------------
        # Build frontend response
        # ---------------------------------------------------------------------

        response = _build_response(
            final_state,
            dataframe,
            dataset_name,
        )

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # Convert every nested NumPy / pandas / Pydantic value
        # before FastAPI attempts JSON serialization.
        # ---------------------------------------------------------------------

        return make_json_safe(
            response
        )

    # -------------------------------------------------------------------------
    # Preserve FastAPI HTTP errors
    # -------------------------------------------------------------------------

    except HTTPException:
        raise

    # -------------------------------------------------------------------------
    # Convert unexpected errors to HTTP 500
    # -------------------------------------------------------------------------

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    # -------------------------------------------------------------------------
    # Remove temporary CSV
    # -------------------------------------------------------------------------

    finally:

        if temp_path is not None:
            temp_path.unlink(
                missing_ok=True
            )