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
        return costs
