import os
import textwrap
import time
from types import MethodType
from typing import Union, Optional, Type

import numpy as np
from commonroad.common.solution import CommonRoadSolutionWriter
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.feasibility.vehicle_dynamics import StateException
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_rp.cost_function import CostFunction, DefaultCostFunction
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.utility.evaluation import run_evaluation, create_full_solution_trajectory
from drplanner.describer.base import (
    PlanningException,
    CompilerException,
    MissingParameterException,
    MissingSignatureException,
)

from drplanner.diagnostics.base import DrPlannerBase
from drplanner.prompter.sampling import PrompterSampling
from drplanner.utils.config import DrPlannerConfiguration


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
    max_planning_time = 40
    current_time = time.time()
    # Add first state to recorded state and input list
    planner.record_state_and_input(planner.x_0)
    optimal = None
    while not planner.goal_reached():
        if time.time() - current_time > max_planning_time:
            cause = "Planning took too much time and was terminated!"
            solution = (
                "The vehicle might be driving to slow or is stuck without moving."
            )
            raise PlanningException(cause, solution)
        current_count = len(planner.record_state_list) - 1

        # check if planning cycle or not
        plan_new_trajectory = current_count % config.planning.replanning_frequency == 0

        if plan_new_trajectory:
            # new planning cycle -> plan a new optimal trajectory
            planner.set_desired_velocity(current_speed=planner.x_0.velocity)
            optimal = planner.plan()

            if not optimal:
                cause = "No optimal trajectory could be found!"
                solution = (
                    "Redefine the cost function or adjust the configuration parameters."
                )
                raise PlanningException(cause, solution)

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
                cause = "No optimal trajectory could be found!"
                solution = (
                    "Redefine the cost function or adjust the configuration parameters."
                )
                raise PlanningException(cause, solution)

            planner.record_state_and_input(optimal[0].state_list[1 + temp])
            planner.reset(
                initial_state_cart=planner.record_state_list[-1],
                initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
                collision_checker=planner.collision_checker,
                coordinate_system=planner.coordinate_system,
            )
    return create_full_solution_trajectory(config, planner.record_state_list)


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
            self.config.temperature,
            self.config.gpt_version,
            mockup=self.config.mockup_openAI,
        )
        self.cost_function = self.motion_planner.cost_function

    def repair(self, diagnosis_result: Union[str, None]):
        # reset configuration
        self.motion_planner_config = ReactivePlannerConfiguration.load(
            f"drplanner/planners/standard-config.yaml", self.scenario_path
        )
        self.motion_planner_config.update()

        # ----- planner configuration -----
        t_min = float(diagnosis_result[self.prompter.PLANNER_CONFIG[0][0]])
        t_max = float(diagnosis_result[self.prompter.PLANNER_CONFIG[1][0]])
        d_max = float(diagnosis_result[self.prompter.PLANNER_CONFIG[2][0]])
        time_steps_computation = int(t_max / self.motion_planner_config.planning.dt)
        self.motion_planner_config.planning.time_steps_computation = (
            time_steps_computation
        )
        self.motion_planner_config.sampling.t_min = t_min
        self.motion_planner_config.sampling.d_min = -d_max
        self.motion_planner_config.sampling.d_max = d_max
        # reset planner
        self.motion_planner = get_planner(self.motion_planner_config)

        # ----- heuristic function -----
        try:
            updated_cost_function = diagnosis_result[self.prompter.COST_FUNCTION]
        except Exception as _:
            raise MissingParameterException(self.prompter.COST_FUNCTION)

        # format string to proper python code
        updated_cost_function = textwrap.dedent(updated_cost_function)
        if not updated_cost_function.startswith("def"):
            raise MissingSignatureException()

        # Create a namespace dictionary to hold the compiled function
        function_namespace = {}
        function_namespace.update(self.motion_planner.__dict__)
        # initialize imports:
        function_namespace["np"] = np
        function_namespace["Optional"] = Optional
        function_namespace["TrajectorySample"] = TrajectorySample

        # Execute the updated heuristic function string
        try:
            # TODO: compile function first
            exec(updated_cost_function, globals(), function_namespace)
        except Exception as e:
            # Handle exceptions (e.g., compilation errors)
            raise CompilerException(e)

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

    def describe_planner(
        self,
        diagnosis_result: Union[str, None],
    ) -> str:
        # if there was no diagnosis provided describe starting cost function
        if diagnosis_result is None:
            return self.prompter.generate_planner_description(
                self.cost_function,
                self.motion_planner_config,
            )
        # otherwise describe the repaired version of the cost function
        else:
            updated_cost_function = diagnosis_result[self.prompter.COST_FUNCTION]
            updated_cost_function = textwrap.dedent(updated_cost_function)
            return self.prompter.generate_planner_description(
                updated_cost_function,
                self.motion_planner_config,
            )

    def plan(self, nr_iter: int) -> Trajectory:
        solution = run_planner(
            self.motion_planner, self.motion_planner_config, self.cost_function
        )

        # todo: find a good way to visualize solution

        #if self._save_solution:
        #    # write solution to a CommonRoad XML file
        #    csw = CommonRoadSolutionWriter(solution)
        #    target_folder = self.dir_output + "sampling/solutions/"
        #    os.makedirs(
        #        os.path.dirname(target_folder), exist_ok=True
        #    )  # Ensure the directory exists
        #    csw.write_to_file(
        #        output_path=target_folder,
        #        filename=f"solution_{solution.benchmark_id}_iter_{nr_iter}.xml",
        #        overwrite=True,
        #    )
        return solution
