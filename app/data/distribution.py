import pandas as pd

from app.data.model import DistributionProfile


def calculate_distribution_analysis(
    column_name: str,
    series: pd.Series,
) -> DistributionProfile:

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric_series.empty:
        return DistributionProfile(
            column_name=column_name,
            count=0,
            lower_tail_percentage=0.0,
            upper_tail_percentage=0.0,
            distribution_shape="insufficient_data",
        )

    minimum = float(numeric_series.min())
    maximum = float(numeric_series.max())
    mean = float(numeric_series.mean())
    median = float(numeric_series.median())

    q1 = float(numeric_series.quantile(0.25))
    q3 = float(numeric_series.quantile(0.75))
    iqr = q3 - q1

    skewness = float(numeric_series.skew())

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    lower_tail_count = int(
        (numeric_series < lower_bound).sum()
    )

    upper_tail_count = int(
        (numeric_series > upper_bound).sum()
    )

    count = len(numeric_series)

    lower_tail_percentage = (
        lower_tail_count / count
    ) * 100

    upper_tail_percentage = (
        upper_tail_count / count
    ) * 100

    # Deterministic distribution classification.
    if abs(skewness) < 0.5:
        distribution_shape = "approximately_symmetric"
    elif skewness >= 0.5:
        distribution_shape = "right_skewed"
    else:
        distribution_shape = "left_skewed"

    return DistributionProfile(
        column_name=column_name,
        count=count,
        minimum=round(minimum, 6),
        maximum=round(maximum, 6),
        mean=round(mean, 6),
        median=round(median, 6),
        q1=round(q1, 6),
        q3=round(q3, 6),
        iqr=round(iqr, 6),
        skewness=round(skewness, 6),
        lower_tail_percentage=round(
            lower_tail_percentage,
            6,
        ),
        upper_tail_percentage=round(
            upper_tail_percentage,
            6,
        ),
        distribution_shape=distribution_shape,
    )