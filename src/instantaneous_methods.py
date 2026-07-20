import numpy as np
import pandas as pd


def velocity_magnitude_method(data):
    """
    VM = magnitude of left-hand velocity
       + magnitude of right-hand velocity
    """
    left_speed = np.linalg.norm(data["left_velocity"], axis=1)
    right_speed = np.linalg.norm(data["right_velocity"], axis=1)

    return left_speed + right_speed, "Velocity Magnitude (m/s)"


def velocity_acceleration_method(data, acceleration_weight=0.1):
    """
    E = alpha * total acceleration
      + (1 - alpha) * total velocity
    *** normalized using max velocity, acceleration ***
    """
    if not 0 <= acceleration_weight <= 1:
        raise ValueError("acceleration_weight must be between 0 and 1.")

    velocity_weight = 1 - acceleration_weight

    left_speed = np.linalg.norm(data["left_velocity"], axis=1)
    right_speed = np.linalg.norm(data["right_velocity"], axis=1)
    total_speed = left_speed + right_speed

    left_acceleration = np.linalg.norm(
        data["left_acceleration"],
        axis=1,
    )
    right_acceleration = np.linalg.norm(
        data["right_acceleration"],
        axis=1,
    )
    total_acceleration = left_acceleration + right_acceleration

    max_accleration = max(total_acceleration)
    max_velocity = max(total_speed)

    velocity_norm = total_speed / max_velocity
    acceleration_norm = total_acceleration / max_accleration

    exertion = (
        acceleration_weight * acceleration_norm
        + velocity_weight * velocity_norm
    )

    return exertion, "Normalized Velocity & Acceleration"


def energy_method(
    data,
    hand_mass=0.4,
    gravity=9.81,
    kinetic_weight=0.85,
):
    """
    Weighted combination of hand kinetic and potential energy.
    """
    if not 0 <= kinetic_weight <= 1:
        raise ValueError("kinetic_weight must be between 0 and 1.")

    potential_weight = 1 - kinetic_weight

    left_speed = np.linalg.norm(data["left_velocity"], axis=1)
    right_speed = np.linalg.norm(data["right_velocity"], axis=1)

    kinetic_left = 0.5 * hand_mass * left_speed**2
    kinetic_right = 0.5 * hand_mass * right_speed**2
    kinetic_total = kinetic_left + kinetic_right

    left_height = data["left_position"][:, 1].copy()
    right_height = data["right_position"][:, 1].copy()

    left_height -= left_height.min()
    right_height -= right_height.min()

    potential_left = hand_mass * gravity * left_height
    potential_right = hand_mass * gravity * right_height
    potential_total = potential_left + potential_right

    exertion = (
        kinetic_weight * kinetic_total
        + potential_weight * potential_total
    )

    return exertion, "Weighted Total Energy (J)"


def work_method(data, hand_mass=0.4):
    """
    Approximate net work between consecutive frames using F · dx.
    """
    left_force = hand_mass * data["left_acceleration"][:-1]
    right_force = hand_mass * data["right_acceleration"][:-1]

    left_displacement = np.diff(
        data["left_relative_position"],
        axis=0,
    )
    right_displacement = np.diff(
        data["right_relative_position"],
        axis=0,
    )

    left_work = np.sum(left_force * left_displacement, axis=1)
    right_work = np.sum(right_force * right_displacement, axis=1)

    return np.abs(left_work) + np.abs(right_work), "Work Approximation (J)"

def paddle_method(data):
    return abs(data["paddle_contribution"]), "Paddle Contribution (m/s)"