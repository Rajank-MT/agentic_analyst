import pandas as pd

from app.data.model import AnomalyProfile


def detect_iqr_anomalies(
    column_name: str,
    series: pd.Series,
) -> AnomalyProfile:
    """
    Detect numeric anomalies using the IQR method.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric_series.empty:
        return AnomalyProfile(
            column_name=column_name,
            method="iqr",
            anomaly_count=0,
            anomaly_percentage=0.0,
        )

    q1 = numeric_series.quantile(0.25)
    q3 = numeric_series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    anomalies = numeric_series[
        (numeric_series < lower_bound)
        | (numeric_series > upper_bound)
    ]

    anomaly_count = int(len(anomalies))

    anomaly_percentage = (
        anomaly_count / len(numeric_series)
    ) * 100

    return AnomalyProfile(
        column_name=column_name,
        method="iqr",
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        anomaly_count=anomaly_count,
        anomaly_percentage=round(
            anomaly_percentage,
            6,
        ),
        anomaly_values=[
            float(value)
            for value in anomalies.tolist()
        ],
    )