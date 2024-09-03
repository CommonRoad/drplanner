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

    def __init__(self):
        super().__init__()

    def generate(
        self, cost_result: PlanningProblemCostResult, desired_cost: int
    ) -> str:
        description = f"To the chosen trajectory a total penalty of {cost_result.total_costs:.2f} was issued. "
        description += "It consists of "
        for item, cost in cost_result.partial_costs.items():
            cost = cost * cost_result.weights[item]
            if cost == 0.0:
                continue
            description += f"{CostFunctionMeaningMapping[item]}, valued at {cost:.2f}; "
        self.description = description[:-2] + ". "
        self.description += (
            f"The objective is to decrease this penalty of the planned trajectory to closely "
            f"align with desired value {desired_cost}.\n"
        )
        return self.description

    def compare(
        self,
        a: PlanningProblemCostResult,
        version_a: str,
        b: PlanningProblemCostResult,
        desired_cost: int,
    ):
        self.description = TrajectoryCostDescription.evaluate(
            a, b, version_a, "current"
        )
        self.description += (
            f"The objective is to adjust the total cost of the planned trajectory to closely "
            f"align with the desired value {desired_cost}.\n"
        )
        return self.description

    @staticmethod
    def evaluate(
        a: PlanningProblemCostResult,
        b: PlanningProblemCostResult,
        a_name: str,
        b_name: str,
    ) -> str:
        description = f"Here is a performance comparison between the {a_name} and {b_name} version\n"
        description += TrajectoryCostDescription._compare(
            "total cost", a.total_costs, b.total_costs, a_name, b_name
        )
        description += "Total cost is calculated by a weighted sum of:\n"
        for (item, a_cost), (_, b_cost) in zip(
            a.partial_costs.items(), b.partial_costs.items()
        ):
            a_cost *= a.weights[item]
            b_cost *= a.weights[item]
            if a_cost == 0.0 and b_cost == 0.0:
                continue
            description += TrajectoryCostDescription._compare(
                CostFunctionMeaningMapping[item], a_cost, b_cost, a_name, b_name
            )
        description = description[:-2] + ". "

        threshold = 1.0
        if abs(a.total_costs - b.total_costs) < threshold:
            description += "The performance of the motion planner is stagnating."
        elif a.total_costs < b.total_costs:
            description += "The performance of the motion planner is getting worse."
        elif a.total_costs > b.total_costs:
            description += "The performance of the motion planner is getting better."
        return description

    @staticmethod
    def _compare(
        item: str,
        initial: float,
        repaired: float,
        version_initial: str,
        version_repaired: str,
    ):
        threshold_1 = 1.0
        threshold_2 = 0.1
        if abs(initial - repaired) < threshold_2:
            compare = "about equal to"
        elif threshold_1 > repaired - initial > 0:
            compare = "slightly worse than"
        elif repaired > initial:
            compare = "worse than"
        elif threshold_1 > initial - repaired > 0:
            compare = "slightly better than"
        else:
            compare = "better than"
        return f"- {item}: {version_repaired} result ({repaired:.2f}) is {compare} {version_initial} result ({initial:.2f})\n"


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
