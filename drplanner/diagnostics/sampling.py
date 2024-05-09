import importlib
import os
from types import MethodType
from typing import Union, Optional, Tuple

import numpy as np
from commonroad.common.solution import CommonRoadSolutionWriter
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.evaluation import PlanningProblemCostResult
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.utility.evaluation import run_evaluation

from drplanner.diagnostics.base import DrPlannerBase
from drplanner.utils.config import DrPlannerConfiguration


def get_planner(filename) -> Tuple[ReactivePlannerConfiguration, ReactivePlanner]:
    # Build config object
    config = ReactivePlannerConfiguration.load(f"standard-config.yaml", filename)
    config.update()
    # run route planner and add reference path to config
    route_planner = RoutePlanner(config.scenario, config.planning_problem)
    route = route_planner.plan_routes().retrieve_first_route()

    # initialize reactive planner
    planner = ReactivePlanner(config)

    # set reference path for curvilinear coordinate system
    planner.set_reference_path(route.reference_path)
    return config, planner


def run_planner(planner, config):
    # Add first state to recorded state and input list
    planner.record_state_and_input(planner.x_0)

    while not planner.goal_reached():
        current_count = len(planner.record_state_list) - 1

        # check if planning cycle or not
        plan_new_trajectory = current_count % config.planning.replanning_frequency == 0
        if plan_new_trajectory:
            # new planning cycle -> plan a new optimal trajectory
            planner.set_desired_velocity(current_speed=planner.x_0.velocity)
            optimal = planner.plan()
            if not optimal:
                break

            planner.record_state_and_input(optimal[0].state_list[1])
            planner.reset(
                initial_state_cart=planner.record_state_list[-1],
                initial_state_curv=(optimal[2][1], optimal[3][1]),
                collision_checker=planner.collision_checker,
                coordinate_system=planner.coordinate_system,
            )
        else:
            # continue on optimal trajectory
            temp = current_count % config.planning.replanning_frequency

            planner.record_state_and_input(optimal[0].state_list[1 + temp])
            planner.reset(
                initial_state_cart=planner.record_state_list[-1],
                initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
                collision_checker=planner.collision_checker,
                coordinate_system=planner.coordinate_system,
            )
    solution, _ = run_evaluation(
        planner.config, planner.record_state_list, planner.record_input_list
    )
    return solution


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
        self.cost_function = self.DefaultCostFunction(
            self.motion_planner.x_0.velocity, desired_d=0.0, desired_s=None
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
        self.cost_function.evaluate = new_heuristic.__get__(self.cost_function)

        self.cost_function = self.DefaultCostFunction(
            self.motion_planner.x_0.velocity, desired_d=0.0, desired_s=None
        )

        self.cost_function.evaluate = MethodType(new_heuristic, self.cost_function)

    def describe(
            self, planned_trajectory: Union[Trajectory, None]
    ) -> (str, PlanningProblemCostResult):

        template = self.prompter.reactive_template

        planner_description = self.prompter.generate_reactive_planner_description(self.DefaultCostFunction)

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

    def plan(self, nr_iter: int) -> Trajectory:
        solution = run_planner(self.motion_planner, self.motion_planner_config)
        planning_problem_solution = solution.planning_problem_solutions[0]
        trajectory_solution = planning_problem_solution.trajectory

        # todo: find a good way to visualize solution

        if self._save_solution:
            # write solution to a CommonRoad XML file
            csw = CommonRoadSolutionWriter(solution)
            target_folder = self.dir_output + "search/solutions/"
            os.makedirs(
                os.path.dirname(target_folder), exist_ok=True
            )  # Ensure the directory exists
            csw.write_to_file(
                output_path=target_folder,
                filename=f"solution_{solution.benchmark_id}_iter_{nr_iter}.xml",
                overwrite=True,
            )
        return trajectory_solution
