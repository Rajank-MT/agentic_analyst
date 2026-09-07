import pandas as pd

from app.data.model import NumericStatistics


def calculate_numeric_statistics(
    column_name: str,
    series: pd.Series,
) -> NumericStatistics:
    """
    Calculate descriptive statistics for a numeric column.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric_series.empty:
        return NumericStatistics(
            column_name=column_name,
            count=0,
        )

    return NumericStatistics(
        column_name=column_name,
        count=int(numeric_series.count()),
        mean=float(numeric_series.mean()),
        median=float(numeric_series.median()),
        minimum=float(numeric_series.min()),
        maximum=float(numeric_series.max()),
        standard_deviation=float(
            numeric_series.std()
        ),
        variance=float(
            numeric_series.var()
        ),
        q1=float(
            numeric_series.quantile(0.25)
        ),
        q3=float(
            numeric_series.quantile(0.75)
        ),
        skewness=float(
            numeric_series.skew()
        ),
    )