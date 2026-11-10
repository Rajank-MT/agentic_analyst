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
    """
    Compare a numeric measure across categorical segments.

    Rows with a missing category or a non-numeric/missing measure are excluded.
    The two Series are aligned by their pandas indexes when constructing the
    temporary DataFrame.
    """
    dataframe = pd.DataFrame(
        {
            "category": categorical_series,
            "value": pd.to_numeric(measure_series, errors="coerce"),
        }
    ).dropna(subset=["category", "value"])

    if dataframe.empty:
        return SegmentRelationshipProfile(
            categorical_column=categorical_column,
            measure_column=measure_column,
            segments=[],
        )

    overall_mean = float(dataframe["value"].mean())
    overall_median = float(dataframe["value"].median())

    # observed=True avoids generating empty groups for unused categorical
    # levels when the input category has pandas Categorical dtype.
    grouped = dataframe.groupby(
        "category",
        observed=True,
    )["value"]

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

    # Highest mean first; lowest mean last.
    segments.sort(
        key=lambda segment: (
            segment.mean
            if segment.mean is not None
            else float("-inf")
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
