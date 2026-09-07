import pandas as pd

from app.data.model import CorrelationProfile


def _get_correlation_strength(coefficient: float) -> str:
    absolute_value = abs(coefficient)

    if absolute_value >= 0.8:
        return "very_strong"

    if absolute_value >= 0.6:
        return "strong"

    if absolute_value >= 0.4:
        return "moderate"

    if absolute_value >= 0.2:
        return "weak"

    return "very_weak"


def calculate_correlation(
    left_column: str,
    left_series: pd.Series,
    right_column: str,
    right_series: pd.Series,
) -> CorrelationProfile:

    left_numeric = pd.to_numeric(
        left_series,
        errors="coerce",
    )

    right_numeric = pd.to_numeric(
        right_series,
        errors="coerce",
    )

    dataframe = pd.DataFrame({
        "left": left_numeric,
        "right": right_numeric,
    }).dropna()

    if len(dataframe) < 2:
        coefficient = 0.0
    else:
        coefficient = float(
            dataframe["left"].corr(
                dataframe["right"],
                method="pearson",
            )
        )

        if pd.isna(coefficient):
            coefficient = 0.0

    return CorrelationProfile(
        left_column=left_column,
        right_column=right_column,
        coefficient=round(coefficient, 6),
        method="pearson",
        strength=_get_correlation_strength(coefficient),
    )