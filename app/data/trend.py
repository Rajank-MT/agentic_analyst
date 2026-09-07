import pandas as pd

from app.data.model import TrendDirection, TrendPoint, TrendProfile


def calculate_trend(
    time_column: str,
    time_series: pd.Series,
    measure_column: str,
    measure_series: pd.Series,
    frequency: str = "M",
    aggregation: str = "sum",
) -> TrendProfile:

    dataframe = pd.DataFrame({
        "date": pd.to_datetime(time_series, errors="coerce"),
        "value": pd.to_numeric(measure_series, errors="coerce"),
    }).dropna()

    if dataframe.empty:
        return TrendProfile(
            time_column=time_column,
            measure_column=measure_column,
            frequency=frequency,
            aggregation=aggregation,
            points=[],
            overall_direction=TrendDirection.INSUFFICIENT_DATA,
        )

    dataframe["period"] = dataframe["date"].dt.to_period(frequency)

    if aggregation == "sum":
        grouped = dataframe.groupby("period")["value"].sum()
    elif aggregation == "mean":
        grouped = dataframe.groupby("period")["value"].mean()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    grouped = grouped.sort_index()

    points = []
    previous_value = None

    for period, value in grouped.items():

        change = None
        change_percentage = None

        if previous_value is not None:
            change = float(value - previous_value)

            if previous_value != 0:
                change_percentage = float(
                    ((value - previous_value) / abs(previous_value)) * 100
                )

        points.append(
            TrendPoint(
                period=str(period),
                value=round(float(value), 6),
                change=None if change is None else round(change, 6),
                change_percentage=(
                    None
                    if change_percentage is None
                    else round(change_percentage, 6)
                ),
            )
        )

        previous_value = value

    if len(grouped) < 2:
        return TrendProfile(
            time_column=time_column,
            measure_column=measure_column,
            frequency=frequency,
            aggregation=aggregation,
            points=points,
            overall_direction=TrendDirection.INSUFFICIENT_DATA,
        )

    first_value = float(grouped.iloc[0])
    last_value = float(grouped.iloc[-1])

    total_change = last_value - first_value

    if first_value != 0:
        total_change_percentage = (
            total_change / abs(first_value)
        ) * 100
    else:
        total_change_percentage = None

    if total_change > 0:
        direction = TrendDirection.GROWTH
    elif total_change < 0:
        direction = TrendDirection.DECLINE
    else:
        direction = TrendDirection.STABLE

    return TrendProfile(
        time_column=time_column,
        measure_column=measure_column,
        frequency=frequency,
        aggregation=aggregation,
        points=points,
        overall_direction=direction,
        total_change=round(total_change, 6),
        total_change_percentage=(
            None
            if total_change_percentage is None
            else round(total_change_percentage, 6)
        ),
    )