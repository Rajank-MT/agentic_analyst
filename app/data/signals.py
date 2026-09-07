from app.data.model import (
    InsightSignal,
    InsightSignalProfile,
    InsightSignalType,
)


def detect_insight_signals(
    correlations=None,
    anomalies=None,
    trends=None,
    contributions=None,
    segment_relationships=None,
    distributions=None,
):
    signals = []

    # Strong correlations
    if correlations:
        for correlation in correlations:
            coefficient = abs(correlation.coefficient)

            if coefficient >= 0.6:
                signals.append(
                    InsightSignal(
                        signal_type=InsightSignalType.STRONG_CORRELATION,
                        columns=[
                            correlation.left_column,
                            correlation.right_column,
                        ],
                        strength=round(coefficient, 6),
                        evidence={
                            "coefficient": correlation.coefficient,
                            "method": correlation.method,
                        },
                        reason="The two numeric measures have a strong statistical relationship.",
                    )
                )

    # Anomalies
    if anomalies:
        for anomaly in anomalies:
            if anomaly.anomaly_count > 0:
                strength = min(
                    anomaly.anomaly_percentage / 20,
                    1.0,
                )

                signals.append(
                    InsightSignal(
                        signal_type=InsightSignalType.ANOMALY,
                        columns=[anomaly.column_name],
                        strength=round(strength, 6),
                        evidence={
                            "anomaly_count": anomaly.anomaly_count,
                            "anomaly_percentage": anomaly.anomaly_percentage,
                            "method": anomaly.method,
                        },
                        reason="The measure contains observations outside the expected statistical range.",
                    )
                )

    # Trends
    if trends:
        for trend in trends:
            if trend.overall_direction.value in {
                "growth",
                "decline",
            }:
                percentage = trend.total_change_percentage

                if percentage is not None:
                    strength = min(abs(percentage) / 100, 1.0)
                else:
                    strength = 0.0

                signals.append(
                    InsightSignal(
                        signal_type=InsightSignalType.TREND,
                        columns=[
                            trend.time_column,
                            trend.measure_column,
                        ],
                        strength=round(strength, 6),
                        evidence={
                            "direction": trend.overall_direction.value,
                            "total_change": trend.total_change,
                            "total_change_percentage": percentage,
                        },
                        reason="The business measure shows a measurable change over time.",
                    )
                )

    # Dominant segments
    if contributions:
        for contribution in contributions:
            if contribution.categories:
                top_segment = contribution.categories[0]

                if top_segment.contribution_percentage >= 50:
                    strength = min(
                        top_segment.contribution_percentage / 100,
                        1.0,
                    )

                    signals.append(
                        InsightSignal(
                            signal_type=InsightSignalType.DOMINANT_SEGMENT,
                            columns=[
                                contribution.categorical_column,
                                contribution.measure_column,
                            ],
                            strength=round(strength, 6),
                            evidence={
                                "category": top_segment.category,
                                "contribution_percentage": (
                                    top_segment.contribution_percentage
                                ),
                            },
                            reason="A single segment contributes a large share of the overall business measure.",
                        )
                    )

    # Segment differences
    if segment_relationships:
        for relationship in segment_relationships:
            for segment in relationship.segments:
                difference_percentage = segment.difference_percentage

                if (
                    difference_percentage is not None
                    and abs(difference_percentage) >= 20
                ):
                    strength = min(
                        abs(difference_percentage) / 100,
                        1.0,
                    )

                    signals.append(
                        InsightSignal(
                            signal_type=InsightSignalType.SEGMENT_DIFFERENCE,
                            columns=[
                                relationship.categorical_column,
                                relationship.measure_column,
                            ],
                            strength=round(strength, 6),
                            evidence={
                                "category": segment.category,
                                "difference_percentage": (
                                    difference_percentage
                                ),
                                "mean": segment.mean,
                                "overall_mean": (
                                    relationship.overall_mean
                                ),
                            },
                            reason="A segment's average measure differs substantially from the overall average.",
                        )
                    )

    # Distribution skew
    if distributions:
        for distribution in distributions:
            if distribution.distribution_shape in {
                "right_skewed",
                "left_skewed",
            }:
                skew_strength = min(
                    abs(distribution.skewness) / 2,
                    1.0,
                )

                signals.append(
                    InsightSignal(
                        signal_type=InsightSignalType.SKEWED_DISTRIBUTION,
                        columns=[distribution.column_name],
                        strength=round(skew_strength, 6),
                        evidence={
                            "skewness": distribution.skewness,
                            "distribution_shape": (
                                distribution.distribution_shape
                            ),
                        },
                        reason="The measure distribution is substantially asymmetric.",
                    )
                )

    return InsightSignalProfile(signals=signals)