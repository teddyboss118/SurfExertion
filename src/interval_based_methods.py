import numpy as np
import pandas as pd
from src.instantaneous_methods import velocity_magnitude_method

def partition_data(data, dependent, bucket_size, value_name="avg_value"):
    if bucket_size <= 0:
        raise ValueError("bucket_size must be greater than zero.")

    dependent = np.asarray(dependent).squeeze()
    if dependent.ndim != 1:
        raise ValueError(
            f"dependent must be 1-dimensional after squeezing, got shape {dependent.shape}."
        )

    time = np.asarray(data["time"])[-len(dependent):]  # align lengths

    df = pd.DataFrame({
        "time": time,
        "y": dependent,
    })

    df["bucket"] = (df["time"] // bucket_size).astype(int)

    result = (
        df.groupby("bucket", as_index=False)
        .agg(
            start_time=("time", "min"),
            end_time=("time", "max"),
            bucket_length=("y", "size"),
            **{value_name: ("y", "mean")}
        )
    )

    return result

def partitioned_average_over_interval(data, dependent, t_1, t_2):
    dependent = np.asarray(dependent).squeeze()
    time = np.asarray(data["time"])[-len(dependent):]  # align lengths

    mask = (time >= t_1) & (time < t_2)

    if not mask.any():
        raise ValueError("No data points exist in the selected interval.")

    return dependent[mask].mean()