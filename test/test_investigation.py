from app.agent.investigation import build_investigation_summary
from app.data.model import AgentObservation


def test_build_investigation_summary_tracks_successful_analysis():
    observations = [
        AgentObservation(
            tool_name="numeric_statistics",
            target_columns=["revenue"],
            success=True,
            result={"mean": 1000},
        ),
        AgentObservation(
            tool_name="correlation",
            target_columns=["revenue", "cost"],
            success=True,
            result={"coefficient": 0.85},
        ),
    ]

    summary = build_investigation_summary(observations)

    assert summary["total_analyses"] == 2

    assert len(summary["successful_analyses"]) == 2
    assert len(summary["failed_analyses"]) == 0

    assert summary["explored_columns"] == [
        "cost",
        "revenue",
    ]

    assert summary["tool_usage"] == {
        "numeric_statistics": 1,
        "correlation": 1,
    }


def test_build_investigation_summary_tracks_failed_analysis():
    observations = [
        AgentObservation(
            tool_name="correlation",
            target_columns=["revenue", "region"],
            success=False,
            error="Invalid metric role.",
        )
    ]

    summary = build_investigation_summary(observations)

    assert summary["total_analyses"] == 1
    assert len(summary["successful_analyses"]) == 0
    assert len(summary["failed_analyses"]) == 1

    assert summary["failed_analyses"][0]["execution_index"] == 0
    assert summary["failed_analyses"][0]["tool_name"] == "correlation"
    assert summary["failed_analyses"][0]["error"] == "Invalid metric role."

    assert summary["explored_columns"] == [
        "region",
        "revenue",
    ]

    assert summary["tool_usage"] == {
        "correlation": 1,
    }


def test_build_investigation_summary_handles_no_observations():
    summary = build_investigation_summary([])

    assert summary["total_analyses"] == 0
    assert summary["successful_analyses"] == []
    assert summary["failed_analyses"] == []
    assert summary["explored_columns"] == []
    assert summary["tool_usage"] == {}