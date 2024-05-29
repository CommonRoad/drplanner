import importlib
import os
import textwrap
from types import MethodType
from typing import Union, Optional, Tuple, Type

import numpy as np
from commonroad.common.solution import CostFunction as CF
from commonroad.common.solution import CommonRoadSolutionWriter, VehicleType
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.evaluation import (
    PlanningProblemCostResult,
    CostFunctionEvaluator,
)
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_rp.cost_function import CostFunction, DefaultCostFunction
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.utility.evaluation import run_evaluation

from drplanner.diagnostics.base import DrPlannerBase
from drplanner.prompter.sampling import PrompterSampling
from drplanner.utils.config import DrPlannerConfiguration


# Custom Exception if no optimal Trajectory was found
class PlanningException(Exception):
    def __init__(self):
        super().__init__("The planner failed: No optimal trajectory could be found!")


def get_planner(config: ReactivePlannerConfiguration) -> ReactivePlanner:
    # run route planner and add reference path to config
    route_planner = RoutePlanner(
        config.scenario.lanelet_network, config.planning_problem
    )
    route = route_planner.plan_routes().retrieve_first_route()
    # initialize reactive planner
    planner = ReactivePlanner(config)
    # set reference path for curvilinear coordinate system
    planner.set_reference_path(route.reference_path)
    return planner


# helper function to run a ReactivePlanner with a given CostFunction
def run_planner(
    planner: ReactivePlanner,
    config: ReactivePlannerConfiguration,
    cost_function: Type[CostFunction | None],
):
    # update cost function
    planner.set_cost_function(cost_function)

    print(f"The current planning horizon is {planner.horizon}!")
    # Add first state to recorded state and input list
    planner.record_state_and_input(planner.x_0)
    optimal = None
    while not planner.goal_reached():
        current_count = len(planner.record_state_list) - 1

        # check if planning cycle or not
        plan_new_trajectory = current_count % config.planning.replanning_frequency == 0

        if plan_new_trajectory:
            # new planning cycle -> plan a new optimal trajectory
            planner.set_desired_velocity(current_speed=planner.x_0.velocity)
            optimal = planner.plan()

            if not optimal:
                raise PlanningException

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

            if not optimal:
                raise PlanningException

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
        # Build config object
        self.scenario_path = scenario_path
        self.motion_planner_config = ReactivePlannerConfiguration.load(
            f"drplanner/planners/standard-config.yaml", self.scenario_path
        )
        self.motion_planner_config.update()
        # initialize motion planner
        self.motion_planner = get_planner(self.motion_planner_config)

        # initialize prompter
        self.prompter = PrompterSampling(
            self.scenario,
            self.planning_problem,
            self.config.openai_api_key,
            self.config.gpt_version,
            mockup=self.config.mockup_openAI,
        )
        self.prompter.LLM.temperature = self.config.temperature
        self.cost_function = self.motion_planner.cost_function

    def repair(self, diagnosis_result: Union[str, None]):
        # ----- planner configuration -----
        updated_time_step_amount = diagnosis_result[self.prompter.PLANNER_CONFIG]
        if updated_time_step_amount:
            try:
                self.motion_planner_config = ReactivePlannerConfiguration.load(
                    f"drplanner/planners/standard-config.yaml", self.scenario_path
                )
                self.motion_planner_config.update()
                self.motion_planner_config.planning.time_steps_computation = int(
                    updated_time_step_amount
                )
                self.motion_planner = get_planner(self.motion_planner_config)
            except Exception as e:
                raise RuntimeError(f"Could not convert time steps into an int: {e}")

        # ----- heuristic function -----
        updated_cost_function = diagnosis_result[self.prompter.COST_FUNCTION]
        updated_cost_function = textwrap.dedent(updated_cost_function)
        # Create a namespace dictionary to hold the compiled function
        function_namespace = {}
        function_namespace.update(self.motion_planner.__dict__)
        # initialize imports:
        function_namespace["np"] = np
        function_namespace["Optional"] = Optional
        function_namespace["TrajectorySample"] = TrajectorySample

        # Execute the updated heuristic function string
        try:
            exec(updated_cost_function, globals(), function_namespace)
        except Exception as e:
            # Handle exceptions (e.g., compilation errors)
            raise RuntimeError(f"Error compiling heuristic function: {e}")

        # Extract the new function
        new_cost_function = function_namespace["evaluate"]

        if not callable(new_cost_function):
            raise ValueError("No valid 'heuristic_function' found after execution")

        # Bind the function to the StudentMotionPlanner instance
        self.cost_function.evaluate = new_cost_function.__get__(self.cost_function)

        self.cost_function = DefaultCostFunction(
            self.motion_planner.x_0.velocity, desired_d=0.0, desired_s=None
        )

        self.cost_function.evaluate = MethodType(new_cost_function, self.cost_function)

    def describe(
        self,
        planned_trajectory: Union[Trajectory, str],
        diagnosis_result: Union[str, None],
    ) -> (str, PlanningProblemCostResult):

        template = self.prompter.algorithm_template
        # --- GENERATE PLANNER DESCRIPTION ---
        # if there was no diagnosis provided describe starting cost function
        if diagnosis_result is None:
            planner_description = self.prompter.generate_planner_description(
                self.cost_function,
                self.motion_planner_config,
            )
        # otherwise describe the repaired version of the cost function
        else:
            updated_cost_function = diagnosis_result[self.prompter.COST_FUNCTION]
            updated_cost_function = textwrap.dedent(updated_cost_function)
            planner_description = self.prompter.generate_planner_description(
                updated_cost_function
            )
        template = template.replace("[PLANNER]", planner_description)

        # --- GENERATE TRAJECTORY DESCRIPTION ---
        if isinstance(planned_trajectory, Trajectory):
            evaluation_trajectory = self.evaluate_trajectory(planned_trajectory)

            traj_description = self.prompter.generate_cost_description(
                evaluation_trajectory, self.desired_cost
            )
        else:
            traj_description = f" The planner failed: {planned_trajectory}"
            evaluation_trajectory = None
        template = template.replace("[PLANNED_TRAJECTORY]", traj_description)
        return template, evaluation_trajectory

    def plan(self, nr_iter: int) -> Trajectory:
        solution = run_planner(
            self.motion_planner, self.motion_planner_config, self.cost_function
        )
        planning_problem_solution = solution.planning_problem_solutions[0]
        trajectory_solution = planning_problem_solution.trajectory

        # todo: find a good way to visualize solution

        if self._save_solution:
            # write solution to a CommonRoad XML file
            csw = CommonRoadSolutionWriter(solution)
            target_folder = self.dir_output + "sampling/solutions/"
            os.makedirs(
                os.path.dirname(target_folder), exist_ok=True
            )  # Ensure the directory exists
            csw.write_to_file(
                output_path=target_folder,
                filename=f"solution_{solution.benchmark_id}_iter_{nr_iter}.xml",
                overwrite=True,
            )
        return trajectory_solution
