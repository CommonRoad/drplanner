from commonroad_dc.costs.evaluation import (
    PlanningProblemCostResult,
    PartialCostFunction,
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


class TrajectoryCostDescription(DescriptionBase):
    domain = "trajectory_cost"

    def __init__(self, cost_result: PlanningProblemCostResult):
        super().__init__()
        self.cost_result = cost_result

    def generate(self, desired_value: float):
        description = (
            f"The objective is to adjust the total cost of the planned trajectory to closely "
            f"align with the desired value {desired_value}. "
        )
        description += f"The current total cost is calculated to be {self.cost_result.total_costs:.2f}, "
        description += "includes "
        for item, cost in self.cost_result.partial_costs.items():
            description += (
                f"{CostFunctionMeaningMapping[item]}, valued at {cost:.2f} "
                f"with a weight of {self.cost_result.weights[item]}; "
            )
        self.description = description[:-2] + "."
        return self.description

    def update(self):
        description = f"the updated total cost is calculated to be {self.cost_result.total_costs:.2f}, "
        description += "which includes "
        for item, cost in self.cost_result.partial_costs.items():
            description += (
                f"{CostFunctionMeaningMapping[item]}, which is {cost:.2f} "
                f"with a weight of {self.cost_result.weights[item]}; "
            )
        self.description = description[:-2] + "."
        return self.description


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
