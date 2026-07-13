import numpy as np
import pandas as pd
from src.instantaneous_methods import velocity_magnitude_method

def partition_data(data, dependent, bucket_size):
    if bucket_size <= 0:
        raise ValueError("bucket_size must be greater than zero.")

    time = data["time"]

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
            avg_VM=("y", "mean"),
        )
    )

    return result

def partitioned_average_over_interval(data, dependent, t_1, t_2):
    time = data["time"]

    mask = (time >= t_1) & (time < t_2)

    if not mask.any():
        raise ValueError("No data points exist in the selected interval.")

    return dependent[mask].mean()