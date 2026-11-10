
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from app.sources.postgres import (
    list_postgres_tables,
    preview_postgres_table,
    test_postgres_connection,
    load_postgres_table,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/postgres",
    tags=["PostgreSQL"],
)


class PostgresAnalyzeRequest(BaseModel):
    schema_name: str = Field(
        default="public",
        min_length=1,
        max_length=255,
    )
    table_name: str = Field(
        min_length=1,
        max_length=255,
    )
    row_limit: int = Field(
        default=50000,
        ge=1,
        le=50000,
    )


@router.get("/test-connection")
def test_connection():
    """Test the PostgreSQL connection."""

    try:
        test_postgres_connection()

        return {
            "success": True,
            "message": "PostgreSQL connection successful.",
        }

    except ValueError as exc:
        logger.exception("PostgreSQL configuration error")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        logger.exception("PostgreSQL connection failed")

        raise HTTPException(
            status_code=400,
            detail="Could not connect to PostgreSQL. Check configuration and firewall.",
        ) from exc


@router.get("/tables")
def get_tables(
    schema_name: str = Query(
        default="public",
        min_length=1,
        max_length=255,
    ),
):
    """List tables in a PostgreSQL schema."""

    try:
        tables = list_postgres_tables(schema=schema_name)

        return {
            "schema": schema_name,
            "tables": tables,
            "count": len(tables),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        logger.exception("Could not retrieve PostgreSQL tables")

        raise HTTPException(
            status_code=400,
            detail="Could not retrieve tables. Check connection and schema.",
        ) from exc


@router.get("/preview")
def preview_table(
    table_name: str = Query(
        min_length=1,
        max_length=255,
    ),
    schema_name: str = Query(
        default="public",
        min_length=1,
        max_length=255,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
):
    """Preview rows from a PostgreSQL table."""

    try:
        dataframe = preview_postgres_table(
            table_name=table_name,
            schema=schema_name,
            limit=limit,
        )

        records = dataframe.astype(object).where(
            dataframe.notna(),
            None,
        ).to_dict(orient="records")

        return {
            "schema": schema_name,
            "table": table_name,
            "rows_returned": len(records),
            "columns": list(dataframe.columns),
            "rows": records,
        }

    except ValueError as exc:
        logger.exception("PostgreSQL preview validation failed")

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        logger.exception("PostgreSQL SQLAlchemy preview failed")

        raise HTTPException(
            status_code=500,
            detail=f"PostgreSQL preview failed: {str(exc)}",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected PostgreSQL preview error")

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected preview error: {str(exc)}",
        ) from exc


@router.post("/analyze")
def analyze_postgres_table(request: PostgresAnalyzeRequest):
    """Analyze a PostgreSQL table using the existing agent pipeline."""

    from app.api.main import _build_response, make_json_safe
    from app.agent.loop import AgentLoop
    from app.agent.providers.azure_openai import AzureOpenAIAgentLLM
    from app.agent.providers.azure_openai_insight import (
        AzureOpenAIInsightGenerator,
    )
    from app.agent.state import create_agent_state
    from app.data.metrics import discover_metric_role
    from app.data.profiling import profile_dataset
    from app.data.sampling import build_adaptive_sample
    from app.data.tool_registry import build_analysis_tool_registry

    try:
        dataframe = load_postgres_table(
            table_name=request.table_name,
            schema=request.schema_name,
            limit=request.row_limit,
        )

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="The selected table contains no rows.",
            )

        dataset_name = (
            f"{request.schema_name}.{request.table_name}"
        )

        dataset_profile = profile_dataset(
            dataframe=dataframe,
            dataset_name=dataset_name,
        )

        metric_profiles = [
            discover_metric_role(column)
            for column in dataset_profile.columns
        ]

        dataset_sample = build_adaptive_sample(
            dataframe=dataframe,
            random_sample_size=8,
            representative_row_count=8,
            categorical_example_count=8,
            rare_value_example_count=5,
        )

        registry = build_analysis_tool_registry()

        state = create_agent_state(
            user_goal=(
                "Automatically analyze this dataset and identify important, "
                "non-obvious, statistically supported business insights. "
                "Explore broadly, use deterministic analysis as evidence, "
                "avoid redundant comparisons, and do not invent facts or numbers."
            ),
            dataset_names=[dataset_name],
            metric_profiles=metric_profiles,
            available_tools=registry.tools,
        )

        state.dataset_sample = dataset_sample.model_dump()

        loop = AgentLoop(
            llm=AzureOpenAIAgentLLM(),
            registry=registry,
            max_steps=10,
            batch_size=5,
            insight_generator=AzureOpenAIInsightGenerator(),
        )

        final_state = loop.run(
            state=state,
            dataframe=dataframe,
        )

        response = _build_response(
            final_state,
            dataframe,
            dataset_name,
        )

        return make_json_safe(response)

    except HTTPException:
        raise

    except ValueError as exc:
        logger.exception("PostgreSQL analysis validation failed")

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        logger.exception("PostgreSQL analysis database error")

        raise HTTPException(
            status_code=400,
            detail="Could not read the PostgreSQL table.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected PostgreSQL analysis error")

        raise HTTPException(
            status_code=500,
            detail=(
                "PostgreSQL analysis failed. "
                "Check the backend terminal for diagnostic details."
            ),
        ) from exc
