from typing import Union

import numpy as np
from commonroad.common.solution import CostFunction
from commonroad_dc.costs.evaluation import (
    PlanningProblemCostResult,
    PartialCostFunction,
    cost_function_mapping,
)
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.scenario import Scenario

from drplanner.describer.base import DescriptionBase


CostFunctionMeaningMapping = {
    PartialCostFunction.A: "squared sum of acceleration",
    PartialCostFunction.J: "squared sum of jerk",
    PartialCostFunction.Jlat: "squared sum of lateral jerk",
    PartialCostFunction.Jlon: "squared sum of longitudinal jerk",
    PartialCostFunction.SA: "squared sum of steering angle",
    PartialCostFunction.SR: "squared sum of steering velocity",
    PartialCostFunction.Y: "squared sum of way rate",
    PartialCostFunction.LC: "squared sum of the distance deviation to the lane center",
    PartialCostFunction.V: "squared sum of the deviation to the desired velocity",
    PartialCostFunction.Vlon: "squared sum of the deviation to the desired longitudinal velocity",
    PartialCostFunction.O: "squared sum of the deviation to the desired orientation",
    PartialCostFunction.D: "sum of the inverse of the distance to other obstacles",
    PartialCostFunction.L: "sum of the path length",
    PartialCostFunction.T: "arriving time at the goal region",
    PartialCostFunction.ID: "inverse of the duration of the trajectory",
}


def get_infinite_cost_result(cost_function: CostFunction) -> PlanningProblemCostResult:
    partial_cost_functions = cost_function_mapping[cost_function]
    cost_result = PlanningProblemCostResult(cost_function, 0)
    for pcf, weight in partial_cost_functions:
        cost_result.add_partial_costs(pcf, np.inf, weight)
    return cost_result


class TrajectoryCostDescription(DescriptionBase):
    domain = "trajectory_cost"

    def __init__(self, cost_result: PlanningProblemCostResult):
        super().__init__()
        self.cost_result = cost_result

    def generate(self, update: Union[PlanningProblemCostResult, None]) -> str:
        if update:
            return self._update(update)

        description = f"The current total cost is calculated to be {self.cost_result.total_costs:.2f}, "
        description += "it includes "
        for item, cost in self.cost_result.partial_costs.items():
            description += (
                f"{CostFunctionMeaningMapping[item]}, valued at {cost:.2f} "
                f"with a weight of {self.cost_result.weights[item]}; "
            )
        self.description = description[:-2] + ". "
        return self.description

    def _update(self, update: PlanningProblemCostResult):
        description = (
            "What follows is a performance comparison between the last planner version and the current "
            "planner version.\n"
        )
        description += self._compare(
            "total cost", self.cost_result.total_costs, update.total_costs
        )
        description += "Total cost is calculated by a weighted sum of:\n"
        for (item, initial_cost), (_, repaired_cost) in zip(
            self.cost_result.partial_costs.items(), update.partial_costs.items()
        ):
            item = f"{CostFunctionMeaningMapping[item]} weighted at {self.cost_result.weights[item]}"
            description += self._compare(item, initial_cost, repaired_cost)
        self.description = description[:-2] + ". "

        threshold = 1.0
        if abs(self.cost_result.total_costs - update.total_costs) < threshold:
            self.description += (
                f"The performance of the motion planner is stagnating. You might need to try "
                f"something completely new!"
            )
        elif self.cost_result.total_costs < update.total_costs:
            self.description += (
                f"The performance of the motion planner is getting worse. Please reconsider your last "
                f"changes and keep trying!"
            )
        elif self.cost_result.total_costs > update.total_costs:
            self.description += (
                f"The performance of the motion planner is getting better. Great job, continue like "
                f"that!"
            )
        return self.description

    @staticmethod
    def _compare(item: str, initial: float, repaired: float):
        threshold = 0.01
        if abs(initial - repaired) < threshold:
            compare = "equal to"
        elif repaired > initial:
            compare = "worse than"
        else:
            compare = "better than"
        return f"- {item}: current result ({repaired:.2f}) is {compare} last result ({initial:.2f})\n"


class TrajectoryStateDescription(DescriptionBase):
    domain = "trajectory_state"

    def __init__(self, planned_trajectory: Trajectory, scenario: Scenario):
        super().__init__()
        self.traj = planned_trajectory
        self.scenario = scenario

    def generate(self):
        description = "Here is the state description of the planned trajectory: "
        for state in self.traj.state_list:
            lanelet = self.scenario.lanelet_network.find_most_likely_lanelet_by_state(
                [state]
            )
            description += (
                f"step {state.time_step} - "
                f"position: [{state.position[0]}m, {state.position[1]}m], "
                f"velocity: {state.velocity}m/s, "
                f"orientation: {state.orientation}rad, "
                f"lanelet: {lanelet}; "
            )
        self.description = description[:-2] + "."
        return self.description
