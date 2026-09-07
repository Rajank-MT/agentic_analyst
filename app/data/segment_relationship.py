import pandas as pd

from app.data.model import (
    SegmentComparison,
    SegmentRelationshipProfile,
)


def calculate_segment_relationship(
    categorical_column: str,
    categorical_series: pd.Series,
    measure_column: str,
    measure_series: pd.Series,
) -> SegmentRelationshipProfile:

    dataframe = pd.DataFrame({
        "category": categorical_series,
        "value": pd.to_numeric(measure_series, errors="coerce"),
    }).dropna()

    if dataframe.empty:
        return SegmentRelationshipProfile(
            categorical_column=categorical_column,
            measure_column=measure_column,
            segments=[],
        )

    overall_mean = float(dataframe["value"].mean())
    overall_median = float(dataframe["value"].median())

    grouped = dataframe.groupby("category")["value"]

    segments = []

    for category, values in grouped:
        mean = float(values.mean())
        median = float(values.median())

        difference = mean - overall_mean

        if overall_mean != 0:
            difference_percentage = (
                difference / abs(overall_mean)
            ) * 100
        else:
            difference_percentage = None

        segments.append(
            SegmentComparison(
                category=str(category),
                count=int(values.count()),
                mean=round(mean, 6),
                median=round(median, 6),
                difference_from_overall=round(difference, 6),
                difference_percentage=(
                    None
                    if difference_percentage is None
                    else round(difference_percentage, 6)
                ),
            )
        )

    segments.sort(
        key=lambda segment: (
            segment.mean if segment.mean is not None else float("-inf")
        ),
        reverse=True,
    )

    highest_segment = segments[0].category if segments else None
    lowest_segment = segments[-1].category if segments else None

    return SegmentRelationshipProfile(
        categorical_column=categorical_column,
        measure_column=measure_column,
        overall_mean=round(overall_mean, 6),
        overall_median=round(overall_median, 6),
        segments=segments,
        highest_segment=highest_segment,
        lowest_segment=lowest_segment,
    )