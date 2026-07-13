import numpy as np
import pandas as pd
from src.instantaneous_methods import velocity_magnitude_method

def velocity_magnitude_method_partitioned(data, bucket_size=5):
    if bucket_size <= 0:
        raise ValueError("bucket_size must be greater than zero.")

    vm = velocity_magnitude_method(data)
    time = data["time"]

    interval_df = pd.DataFrame({
        "time": time,
        "VM": vm,
    })

    interval_df["bucket"] = (
        interval_df["time"] // bucket_size
    ).astype(int)

    result = (
        interval_df.groupby("bucket", as_index=False)
        .agg(
            start_time=("time", "min"),
            end_time=("time", "max"),
            bucket_length=("VM", "size"),
            avg_VM=("VM", "mean"),
        )
    )

    return result

def average_velocity_magnitude(data, t_1, t_2):
    time = data["time"]
    vm = velocity_magnitude_method(data)

    mask = (time >= t_1) & (time < t_2)

    if not mask.any():
        raise ValueError("No data points exist in the selected interval.")

    return vm[mask].mean()