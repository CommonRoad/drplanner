from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.cost_function import CostFunction
from scipy.integrate import simps


class DefaultCostFunction(CostFunction):
    """
    Default cost function for comfort driving
    """

    def __init__(
        self,
        desired_speed: Optional[float] = None,
        desired_d: float = 0.0,
        desired_s: Optional[float] = None,
    ):
        super(DefaultCostFunction, self).__init__()
        # target states
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.desired_s = desired_s

        # weights
        self.w_a = 5  # acceleration weight

    def evaluate(self, trajectory: TrajectorySample):
        costs = 0.0
        # acceleration costs
        costs += np.sum((self.w_a * trajectory.cartesian.a) ** 2)
        # velocity costs
        if self.desired_speed is not None:
            costs += (
                np.sum((5 * (trajectory.cartesian.v - self.desired_speed)) ** 2)
                + (50 * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2)
                + (
                    100
                    * (
                        trajectory.cartesian.v[int(len(trajectory.cartesian.v) / 2)]
                        - self.desired_speed
                    )
                    ** 2
                )
            )
        if self.desired_s is not None:
            costs += (
                np.sum((0.25 * (self.desired_s - trajectory.curvilinear.s)) ** 2)
                + (20 * (self.desired_s - trajectory.curvilinear.s[-1])) ** 2
            )

        # distance costs
        costs += (
            np.sum((0.25 * (self.desired_d - trajectory.curvilinear.d)) ** 2)
            + (20 * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2
        )
        # orientation costs
        costs += (
            np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2)
            + (5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2
        )
        cost
        return costs

    @staticmethod
    def acceleration_costs(trajectory: TrajectorySample) -> float:
        """
        Calculates the integral of acceleration squared.
        """
        acceleration = trajectory.cartesian.a
        acceleration_sq = np.square(acceleration)
        cost = simps(acceleration_sq, dx=trajectory.dt)

        return cost

    @staticmethod
    def jerk_costs(trajectory: TrajectorySample) -> float:
        """
        Calculates the integral of jerk squared.
        """
        acceleration = trajectory.cartesian.a
        jerk = np.diff(acceleration) / trajectory.dt
        jerk_sq = np.square(jerk)
        cost = simps(jerk_sq, dx=trajectory.dt)

        return cost

    @staticmethod
    def lateral_jerk_costs(trajectory: TrajectorySample) -> float:
        """
        Same as jerk_costs but only for the lateral trajectory
        """
        cost = trajectory.trajectory_lat.squared_jerk_integral(trajectory.dt)
        return cost

    @staticmethod
    def longitudinal_jerk_costs(trajectory: TrajectorySample) -> float:
        """
        Same as jerk_costs but only for the longitudinal_jerk_costs trajectory
        """
        cost = trajectory.trajectory_long.squared_jerk_integral(trajectory.dt)
        return cost

    @staticmethod
    def orientation_offset_costs(trajectory: TrajectorySample) -> float:
        """
        Calculates the Orientation Offset cost.
        """
        theta = trajectory.curvilinear.theta
        theta = np.diff(theta) / trajectory.dt
        theta = np.square(theta)
        cost = simps(theta, dx=trajectory.dt)

        return cost

    @staticmethod
    def distance_to_reference_path_costs(trajectory: TrajectorySample) -> float:
        """
        Calculates the average lateral distance to the reference path,
        but with a special emphasis on the final state
        """
        d = trajectory.curvilinear.d
        cost = (np.sum(np.abs(d)) + np.abs(d[-1]) * 5) / len(d + 4)

        return float(cost)

    @staticmethod
    def path_length_costs(trajectory: TrajectorySample) -> float:
        """
        Calculates the length of the given Trajectory
        """
        velocity = trajectory.cartesian.v
        cost = simps(velocity, dx=trajectory.dt)
        return cost