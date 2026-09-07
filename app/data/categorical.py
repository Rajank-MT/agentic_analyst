import pandas as pd

from app.data.model import (
    CategoricalAnalysisProfile,
    CategoryStatistics,
)


def calculate_categorical_analysis(
    categorical_column: str,
    categorical_series: pd.Series,
    numeric_column: str,
    numeric_series: pd.Series,
) -> CategoricalAnalysisProfile:

    dataframe = pd.DataFrame(
        {
            "category": categorical_series,
            "value": pd.to_numeric(
                numeric_series,
                errors="coerce",
            ),
        }
    )

    dataframe = dataframe.dropna(subset=["category", "value"])

    categories = []

    for category, group in dataframe.groupby("category"):
        categories.append(
            CategoryStatistics(
                category=str(category),
                count=int(group["value"].count()),
                mean=float(group["value"].mean()),
                median=float(group["value"].median()),
                minimum=float(group["value"].min()),
                maximum=float(group["value"].max()),
            )
        )

    return CategoricalAnalysisProfile(
        categorical_column=categorical_column,
        numeric_column=numeric_column,
        categories=categories,
    )