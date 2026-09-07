import pandas as pd

from app.data.model import (
    ContributionAnalysisProfile,
    ContributionStatistics,
)


def calculate_contribution_analysis(
    categorical_column: str,
    categorical_series: pd.Series,
    measure_column: str,
    measure_series: pd.Series,
    aggregation: str = "sum",
) -> ContributionAnalysisProfile:

    dataframe = pd.DataFrame({
        "category": categorical_series,
        "value": pd.to_numeric(measure_series, errors="coerce"),
    }).dropna()

    if dataframe.empty:
        return ContributionAnalysisProfile(
            categorical_column=categorical_column,
            measure_column=measure_column,
            aggregation=aggregation,
            total_value=0.0,
            categories=[],
        )

    if aggregation == "sum":
        grouped = dataframe.groupby("category")["value"].sum()
    elif aggregation == "mean":
        grouped = dataframe.groupby("category")["value"].mean()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    grouped = grouped.sort_values(ascending=False)

    total_value = float(grouped.sum())

    categories = []

    for rank, (category, value) in enumerate(grouped.items(), start=1):

        if total_value != 0:
            contribution_percentage = (value / total_value) * 100
        else:
            contribution_percentage = 0.0

        categories.append(
            ContributionStatistics(
                category=str(category),
                value=round(float(value), 6),
                contribution_percentage=round(
                    float(contribution_percentage), 6
                ),
                rank=rank,
            )
        )

    return ContributionAnalysisProfile(
        categorical_column=categorical_column,
        measure_column=measure_column,
        aggregation=aggregation,
        total_value=round(total_value, 6),
        categories=categories,
    )