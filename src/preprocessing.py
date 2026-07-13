from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


HAND_COLUMNS = {
    "left": [
        "Left_HandAnchor_Position_x",
        "Left_HandAnchor_Position_y",
        "Left_HandAnchor_Position_z",
    ],
    "right": [
        "Right_HandAnchor_Position_x",
        "Right_HandAnchor_Position_y",
        "Right_HandAnchor_Position_z",
    ],
    "board": [
        "Surfboard_Position_x",
        "Surfboard_Position_y",
        "Surfboard_Position_z",
    ],
    "paddle" : [
        "Paddle_Contribution",
    ]
}


def load_csv(file_path, trim_rows=10):
    """Load and lightly clean one surf-data CSV file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = [
        "Simulation_Time",
        *HAND_COLUMNS["left"],
        *HAND_COLUMNS["right"],
        *HAND_COLUMNS["board"],
        *HAND_COLUMNS["paddle"],
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if trim_rows > 0:
        if len(df) <= 2 * trim_rows:
            raise ValueError("The file has too few rows for the requested trimming.")

        df = df.iloc[trim_rows:-trim_rows].copy()

    df = df.reset_index(drop=True)
    return df


def remove_snaps(position, time, threshold=20.0):
    """
    Replace frames producing impossible speeds with interpolated positions.
    """
    position = position.copy()

    delta_time = np.diff(time)
    delta_position = np.diff(position, axis=0)

    if np.any(delta_time <= 0):
        raise ValueError("Simulation_Time must continuously increase.")

    speed = np.linalg.norm(
        delta_position / delta_time[:, None],
        axis=1,
    )

    # np.diff compares frame i to frame i+1,
    # so flag the destination frame.
    bad_frames = np.where(speed > threshold)[0] + 1

    if len(bad_frames) > 0:
        position[bad_frames] = np.nan

        for axis in range(3):
            position[:, axis] = (
                pd.Series(position[:, axis])
                .interpolate(limit_direction="both")
                .to_numpy()
            )

    return position, bad_frames


def preprocess_data(
    df,
    smoothing_window=31,
    polynomial_order=3,
    snap_threshold=20.0,
    do_savgol=False,
):
    """Create positions, velocities, and accelerations used by all methods."""
    time = df["Simulation_Time"].to_numpy()

    left_hand = df[HAND_COLUMNS["left"]].to_numpy()
    right_hand = df[HAND_COLUMNS["right"]].to_numpy()
    board = df[HAND_COLUMNS["board"]].to_numpy()

    left_relative = left_hand - board
    right_relative = right_hand - board

    # Remove obvious tracking snaps before smoothing.
    left_relative, left_snap_frames = remove_snaps(
        left_relative,
        time,
        threshold=snap_threshold,
    )

    right_relative, right_snap_frames = remove_snaps(
        right_relative,
        time,
        threshold=snap_threshold,
    )

    if do_savgol:
        if smoothing_window >= len(df):
            raise ValueError(
                "The smoothing window must be smaller than the number of rows."
            )

        if smoothing_window % 2 == 0:
            raise ValueError("The smoothing window must be an odd number.")

        left_relative = savgol_filter(
            left_relative,
            window_length=smoothing_window,
            polyorder=polynomial_order,
            axis=0,
        )

        right_relative = savgol_filter(
            right_relative,
            window_length=smoothing_window,
            polyorder=polynomial_order,
            axis=0,
        )

    left_velocity = np.gradient(left_relative, time, axis=0)
    right_velocity = np.gradient(right_relative, time, axis=0)

    left_acceleration = np.gradient(left_velocity, time, axis=0)
    right_acceleration = np.gradient(right_velocity, time, axis=0)

    #paddle contribution
    paddle_contribution = (
        df[HAND_COLUMNS["paddle"]]
        .fillna(0)
        .to_numpy()
    )

    return {
        "time": time,
        "left_position": left_hand,
        "right_position": right_hand,
        "board_position": board,
        "left_relative_position": left_relative,
        "right_relative_position": right_relative,
        "left_velocity": left_velocity,
        "right_velocity": right_velocity,
        "left_acceleration": left_acceleration,
        "right_acceleration": right_acceleration,
        "left_snap_frames": left_snap_frames,
        "right_snap_frames": right_snap_frames,
        "paddle_contribution": paddle_contribution,
    }