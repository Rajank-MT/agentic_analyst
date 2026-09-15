from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DerivedMetricOperation(str, Enum):
    RATIO = "ratio"
    DIFFERENCE = "difference"
    PERCENTAGE_OF_TOTAL = "percentage_of_total"


class AggregationRequest(BaseModel):
    column: str
    aggregation: str


class DerivedMetricRequest(BaseModel):
    name: str
    operation: DerivedMetricOperation

    numerator: str | None = None
    denominator: str | None = None

    left: str | None = None
    right: str | None = None

    source_column: str | None = None


class AnalysisRequest(BaseModel):
    """
    Structured request created by the LLM.
    Python validates and executes it deterministically.
    """

    analysis_name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)

    group_by: list[str] = Field(default_factory=list)

    metrics: list[AggregationRequest] = Field(
        default_factory=list
    )

    derived_metrics: list[DerivedMetricRequest] = Field(
        default_factory=list
    )

    rank_by: str | None = None
    rank_ascending: bool = False

    top_n: int | None = Field(
        default=None,
        gt=0,
    )