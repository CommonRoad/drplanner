import inspect
import textwrap
from types import MethodType
from typing import Optional

import numpy as np
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory

from commonroad_rp.cost_function import DefaultCostFunction
from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from drplanner.describer.base import (
    CompilerException,
    MissingParameterException,
    MissingSignatureException,
)

from drplanner.diagnostics.base import DrPlannerBase
from drplanner.prompter.sampling import PrompterSampling
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.memory.memory import FewShotMemory
from drplanner.planners.reactive_planner import get_planner, run_planner


class DrSamplingPlanner(DrPlannerBase):
    def __init__(
        self,
        memory: FewShotMemory,
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
            memory,
            self.config,
        )
        self.cost_function = self.motion_planner.cost_function
        self.code_current = ""
        self.code_previous = ""

    def repair(self):
        if not self.diagnosis_result:
            return
        # reset configuration
        self.motion_planner_config = ReactivePlannerConfiguration.load(
            f"drplanner/planners/standard-config.yaml", self.scenario_path
        )
        self.motion_planner_config.update()

        # ----- planner configuration -----
        if self.config.repair_sampling_parameters:
            t_min = float(self.diagnosis_result[self.prompter.PLANNER_CONFIG[0][0]])
            t_max = float(self.diagnosis_result[self.prompter.PLANNER_CONFIG[1][0]])
            d_max = float(self.diagnosis_result[self.prompter.PLANNER_CONFIG[2][0]])
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
            updated_cost_function = self.diagnosis_result[self.prompter.COST_FUNCTION]
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

        # Bind the function
        self.cost_function.evaluate = new_cost_function.__get__(self.cost_function)

        self.cost_function = DefaultCostFunction(
            self.motion_planner.x_0.velocity, desired_d=0.0, desired_s=None
        )

        self.cost_function.evaluate = MethodType(new_cost_function, self.cost_function)

    def describe_planner(
        self,
        update: bool = False,
        improved: bool = False,
    ):
        # if at loop start
        if not self.diagnosis_result:
            self.prompter.update_planner_prompt(
                self.cost_function, self.code_previous, self.config.feedback_mode
            )
            self.code_previous = textwrap.dedent(
                inspect.getsource(self.cost_function.evaluate)
            )
        # if new cost function should be described
        elif update:
            try:
                updated_cost_function = self.diagnosis_result[
                    self.prompter.COST_FUNCTION
                ]
            except Exception as _:
                raise MissingParameterException(self.prompter.COST_FUNCTION)
            updated_cost_function = textwrap.dedent(updated_cost_function)
            self.code_current = updated_cost_function
            if improved:
                self.code_previous = self.code_current
            self.prompter.update_planner_prompt(
                self.code_current, self.code_previous, self.config.feedback_mode
            )

        if self.config.repair_sampling_parameters:
            self.prompter.update_config_prompt(self.motion_planner_config)

    def add_memory(self, diagnosis_result: dict):
        summary = diagnosis_result["summary"]
        self.prompter.update_memory_prompt(summary)

    def plan(self, nr_iter: int) -> Trajectory:
        solution = run_planner(
            self.motion_planner, self.motion_planner_config, self.cost_function
        )

        # todo: find a good way to save and visualize solution
        return solution
