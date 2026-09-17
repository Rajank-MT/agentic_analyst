
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ColumnSemanticType(str, Enum):
    """High-level semantic classification of a dataset column."""

    IDENTIFIER = "identifier"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


class KeyProfile(BaseModel):
    """Key-related characteristics for a dataset column."""

    column_name: str

    is_unique: bool

    uniqueness_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    is_candidate_primary_key: bool


class RelationshipCandidate(BaseModel):
    """Potential relationship between two dataset columns."""

    left_dataset: str
    left_column: str

    right_dataset: str
    right_column: str

    name_similarity: float = Field(
        ge=0.0,
        le=1.0,
    )

    semantic_compatibility: float = Field(
        ge=0.0,
        le=1.0,
    )

    dtype_compatibility: float = Field(
        ge=0.0,
        le=1.0,
    )

    candidate_score: float = Field(
        ge=0.0,
        le=1.0,
    )


class ValueOverlapProfile(BaseModel):
    """Measures value overlap between two columns."""

    common_value_count: int = Field(
        ge=0,
    )

    left_unique_count: int = Field(
        ge=0,
    )

    right_unique_count: int = Field(
        ge=0,
    )

    left_overlap_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    right_overlap_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )


class CardinalityType(str, Enum):
    """Relationship cardinality between two columns."""

    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_ONE = "N:1"
    MANY_TO_MANY = "N:N"
    UNKNOWN = "unknown"


class CardinalityProfile(BaseModel):
    """Describes the inferred cardinality of two columns."""

    cardinality: CardinalityType

    left_unique: bool
    right_unique: bool

    left_unique_count: int = Field(
        ge=0,
    )

    right_unique_count: int = Field(
        ge=0,
    )

class RelationshipConfidence(BaseModel):
    """Confidence score and evidence for a relationship."""

    score: float = Field(
        ge=0.0,
        le=1.0,
    )

    name_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    semantic_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    dtype_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    overlap_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    cardinality_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    key_score: float = Field(
        ge=0.0,
        le=1.0,
    )

class RelationshipStatus(str, Enum):
    """Final status of a discovered relationship."""

    CONFIRMED = "confirmed"
    POSSIBLE = "possible"
    REJECTED = "rejected"


class DiscoveredRelationship(BaseModel):
    """Final relationship discovered between two columns."""

    left_dataset: str
    left_column: str

    right_dataset: str
    right_column: str

    status: RelationshipStatus

    confidence: RelationshipConfidence

    overlap: ValueOverlapProfile

    cardinality: CardinalityProfile


class ColumnProfile(BaseModel):
    """Profile information for a single column."""

    name: str
    dtype: str
    semantic_type: ColumnSemanticType

    nullable: bool

    null_count: int = Field(
        ge=0,
    )

    null_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    unique_count: int = Field(
        ge=0,
    )

    unique_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    sample_values: list[Any] = Field(
        default_factory=list,
    )

    min_value: Any | None = None
    max_value: Any | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None

    key_profile: KeyProfile | None = None


class DatasetProfile(BaseModel):
    """Profile information for an entire dataset."""

    dataset_name: str

    row_count: int = Field(
        ge=0,
    )

    column_count: int = Field(
        ge=0,
    )

    duplicate_row_count: int = Field(
        ge=0,
    )

    columns: list[ColumnProfile] = Field(
        default_factory=list,
    )


class NumericStatistics(BaseModel):
    """Statistical summary for a numeric column."""

    column_name: str

    count: int = Field(
        ge=0,
    )

    mean: float | None = None
    median: float | None = None

    minimum: float | None = None
    maximum: float | None = None

    standard_deviation: float | None = None
    variance: float | None = None

    q1: float | None = None
    q3: float | None = None

    skewness: float | None = None


class CorrelationProfile(BaseModel):
    left_column: str
    right_column: str
    coefficient: float = Field(ge=-1.0, le=1.0)
    method: str
    strength: str


class CategoryStatistics(BaseModel):
    category: str
    count: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None


class CategoricalAnalysisProfile(BaseModel):
    categorical_column: str
    numeric_column: str
    categories: list[CategoryStatistics] = Field(default_factory=list)

class AnomalyProfile(BaseModel):
    column_name: str
    method: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    anomaly_count: int = Field(ge=0)
    anomaly_percentage: float = Field(ge=0.0, le=100.0)
    anomaly_values: list[float] = Field(default_factory=list)

class MetricRole(str, Enum):
    MEASURE = "measure"
    DIMENSION = "dimension"
    TIME = "time"
    ENTITY = "entity"
    UNKNOWN = "unknown"


class MetricProfile(BaseModel):
    column_name: str
    role: MetricRole
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)

class AnalysisType(str, Enum):
    NUMERIC_STATISTICS = "numeric_statistics"
    CORRELATION = "correlation"
    CATEGORICAL_ANALYSIS = "categorical_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    TREND_ANALYSIS = "trend_analysis"


class AnalysisPlanItem(BaseModel):
    analysis_type: AnalysisType
    target_columns: list[str] = Field(default_factory=list)
    reason: str


class AnalysisPlan(BaseModel):
    analyses: list[AnalysisPlanItem] = Field(
        default_factory=list
    )

class TrendDirection(str, Enum):
    GROWTH = "growth"
    DECLINE = "decline"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


class TrendPoint(BaseModel):
    period: str
    value: float
    change: float | None = None
    change_percentage: float | None = None


class TrendProfile(BaseModel):
    time_column: str
    measure_column: str
    frequency: str
    aggregation: str
    points: list[TrendPoint] = Field(default_factory=list)
    overall_direction: TrendDirection
    total_change: float | None = None
    total_change_percentage: float | None = None


class ContributionStatistics(BaseModel):
    category: str
    value: float
    contribution_percentage: float
    rank: int


class ContributionAnalysisProfile(BaseModel):
    categorical_column: str
    measure_column: str
    aggregation: str
    total_value: float
    categories: list[ContributionStatistics] = Field(default_factory=list)

class DistributionProfile(BaseModel):
    column_name: str
    count: int = Field(ge=0)

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None

    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None

    skewness: float | None = None

    lower_tail_percentage: float = Field(ge=0.0, le=100.0)
    upper_tail_percentage: float = Field(ge=0.0, le=100.0)

    distribution_shape: str

class SegmentComparison(BaseModel):
    category: str
    count: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    difference_from_overall: float | None = None
    difference_percentage: float | None = None


class SegmentRelationshipProfile(BaseModel):
    categorical_column: str
    measure_column: str
    overall_mean: float | None = None
    overall_median: float | None = None
    segments: list[SegmentComparison] = Field(default_factory=list)
    highest_segment: str | None = None
    lowest_segment: str | None = None

class InsightSignalType(str, Enum):
    STRONG_CORRELATION = "strong_correlation"
    ANOMALY = "anomaly"
    TREND = "trend"
    DOMINANT_SEGMENT = "dominant_segment"
    SEGMENT_DIFFERENCE = "segment_difference"
    SKEWED_DISTRIBUTION = "skewed_distribution"


class InsightSignal(BaseModel):
    signal_type: InsightSignalType
    columns: list[str] = Field(default_factory=list)
    strength: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    reason: str


class InsightSignalProfile(BaseModel):
    signals: list[InsightSignal] = Field(default_factory=list)


class InsightPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InsightEvidence(BaseModel):
    execution_index: int = Field(ge=0)
    fact: str


class BusinessInsight(BaseModel):
    title: str
    finding: str
    evidence: list[InsightEvidence] = Field(default_factory=list)
    business_impact: str
    recommendation: str
    priority: InsightPriority
    caveats: list[str] = Field(default_factory=list)


class BusinessInsightReport(BaseModel):
    insights: list[BusinessInsight] = Field(default_factory=list)

class AnalysisTool(BaseModel):
    name: str
    description: str
    analysis_type: AnalysisType
    required_roles: list[MetricRole] = Field(default_factory=list)
    min_targets: int = Field(default=1, ge=0)
    max_targets: int = Field(default=1, ge=0)


class AnalysisToolRegistry(BaseModel):
    tools: list[AnalysisTool] = Field(default_factory=list)

class ToolRequest(BaseModel):
    tool_name: str
    target_columns: list[str] = Field(default_factory=list)


class ToolValidationResult(BaseModel):
    valid: bool
    tool_name: str
    target_columns: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

class AgentStatus(str, Enum):
    INITIALIZED = "initialized"
    PLANNING = "planning"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    FAILED = "failed"


class ToolExecution(BaseModel):
    tool_name: str
    target_columns: list[str] = Field(default_factory=list)
    success: bool
    result: Any | None = None
    error: str | None = None

class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    REJECTED = "rejected"


class InvestigationHypothesis(BaseModel):
    hypothesis: str
    supporting_execution_indexes: list[int] = Field(
        default_factory=list
    )
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    conclusion: str | None = None


class AgentState(BaseModel):
    user_goal: str
    dataset_names: list[str] = Field(default_factory=list)
    metric_profiles: list[MetricProfile] = Field(default_factory=list)
    available_tools: list[AnalysisTool] = Field(default_factory=list)
    analysis_plan: AnalysisPlan | None = None
    executions: list[ToolExecution] = Field(default_factory=list)
    insight_signals: list[InsightSignal] = Field(default_factory=list)
    hypotheses: list[InvestigationHypothesis] = Field(
        default_factory=list
    )

    # Compact dataset-understanding context produced by adaptive sampling.
    # This is context for the LLM only; deterministic tools still use the
    # complete dataframe for evidence and calculations.
    dataset_sample: dict[str, Any] | None = None

    insights: list[BusinessInsight] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.INITIALIZED

class AgentDecision(BaseModel):
    decision_type: str
    reason: str
    tool_name: str | None = None
    target_columns: list[str] = Field(default_factory=list)

class AgentToolCall(BaseModel):
    tool_name: str
    target_columns: list[str] = Field(default_factory=list)
    reasoning: str

class AgentComposedAnalysis(BaseModel):
    analysis_request: dict[str, Any]
    reasoning: str



class AgentObservation(BaseModel):
    tool_name: str
    target_columns: list[str] = Field(default_factory=list)
    success: bool
    result: Any | None = None
    error: str | None = None



class AgentActionType(str, Enum):
    RUN_TOOL = "run_tool"
    RUN_COMPOSED_ANALYSIS = "run_composed_analysis"
    FINISH = "finish"


class AgentAction(BaseModel):
    action_type: AgentActionType
    tool_call: AgentToolCall | None = None
    composed_analysis: AgentComposedAnalysis | None = None
    reason: str
    hypothesis: InvestigationHypothesis | None = None