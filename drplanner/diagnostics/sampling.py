import importlib
from types import MethodType
from typing import Union, Optional

import numpy as np
from commonroad.common.solution import CostFunction as CostFunctionType, VehicleType, VehicleModel
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.evaluation import CostFunctionEvaluator, PlanningProblemCostResult
from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.cost_function import DefaultCostFunction

from drplanner.planners.reactive_planner import get_planner, run_planner
from drplanner.diagnostics.base import DrPlannerBase
from drplanner.utils.config import DrPlannerConfiguration


class DrSamplingPlanner(DrPlannerBase):
    def __init__(
            self,
            scenario: Scenario,
            scenario_path: str,
            planning_problem_set: PlanningProblemSet,
            config: DrPlannerConfiguration,
            cost_function_id: str,
    ):
        super().__init__(scenario, planning_problem_set, config, cost_function_id)

        # initialize the motion planner
        self.motion_planner_config, self.motion_planner = get_planner(scenario_path)

        # import the cost function
        cost_function_name = f"drplanner.planners.student_{cost_function_id}"
        cost_function_module = importlib.import_module(cost_function_name)
        self.DefaultCostFunction = getattr(cost_function_module, "DefaultCostFunction")
        self.cost_function = self.DefaultCostFunction(self.motion_planner.x_0.velocity, desired_d=0.0,
                                                      desired_s=None)

        # initialize meta parameters
        self.cost_type = CostFunctionType.SM1
        self.vehicle_type = VehicleType.BMW_320i
        self.vehicle_model = VehicleModel.KS
        self.cost_evaluator = CostFunctionEvaluator(
            self.cost_type, VehicleType.BMW_320i
        )

    def repair(self, diagnosis_result: Union[str, None]):
        # ----- heuristic function -----
        updated_heuristic_function = diagnosis_result[
            self.prompter.LLM.HEURISTIC_FUNCTION
        ]
        # Create a namespace dictionary to hold the compiled function
        function_namespace = {}
        function_namespace.update(self.motion_planner.__dict__)
        # initialize imports:
        function_namespace["np"] = np
        function_namespace["Optional"] = Optional
        function_namespace["TrajectorySample"] = TrajectorySample

        # Execute the updated heuristic function string
        try:
            exec(updated_heuristic_function, globals(), function_namespace)
        except Exception as e:
            # Handle exceptions (e.g., compilation errors)
            raise RuntimeError(f"Error compiling heuristic function: {e}")

        # Extract the new function
        new_heuristic = function_namespace["evaluate"]
        if not callable(new_heuristic):
            raise ValueError("No valid 'heuristic_function' found after execution")

        # Bind the function to the StudentMotionPlanner instance
        self.cost_function.evaluate = new_heuristic.__get__(
            self.cost_function
        )

        self.cost_function = self.DefaultCostFunction(self.motion_planner.x_0.velocity, desired_d=0.0,
                                                      desired_s=None)

        self.cost_function.evaluate = MethodType(
            new_heuristic, self.cost_function
        )

    def describe(self, planned_trajectory: Union[Trajectory, None]) -> (str, PlanningProblemCostResult):
        template = self.prompter.astar_template

        planner_description = self.prompter.generate_planner_description(
            self.StudentMotionPlanner, self.motion_primitives_id
        )
        template = template.replace("[PLANNER]", planner_description)

        if planned_trajectory:
            evaluation_trajectory = self.evaluate_trajectory(planned_trajectory)

            traj_description = self.prompter.generate_cost_description(
                evaluation_trajectory, self.desired_cost
            )
        else:
            traj_description = "*\t no trajectory is generated"
            evaluation_trajectory = None
        template = template.replace("[PLANNED_TRAJECTORY]", traj_description)
        return template, evaluation_trajectory

    def add_feedback(self, updated_trajectory: Trajectory, iteration: int):
        pass

    def plan(self, nr_iter: int) -> Trajectory:
        pass

    def evaluate_trajectory(self, trajectory: Trajectory) -> PlanningProblemCostResult:
        pass
