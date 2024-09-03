import os
import textwrap
from typing import Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad_dc.costs.evaluation import PlanningProblemCostResult

from drplanner.describer.base import MissingParameterException
from drplanner.diagnostics.base import DrPlannerBase
from drplanner.prompter.sampling import PrompterSampling
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.planners.reactive_planner import ReactiveMotionPlanner


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
        self.absolute_scenario_path = scenario_path
        if config.include_plot:
            self.absolute_save_path = os.path.join(
                os.path.dirname(config.project_path), "plots", "img.png"
            )
        else:
            self.absolute_save_path = None
        self.motion_planner = ReactiveMotionPlanner(None, None, None, None)
        self.last_motion_planner = None

        # initialize prompter
        self.prompter = PrompterSampling(
            self.scenario,
            self.planning_problem,
            self.config,
        )

    def repair(self):
        if not self.diagnosis_result:
            return
        try:
            updated_cost_function = self.diagnosis_result[self.prompter.COST_FUNCTION]
            updated_cost_function = textwrap.dedent(updated_cost_function)
            helper_methods = self.diagnosis_result[self.prompter.HELPER_METHODS]
            helper_methods = [textwrap.dedent(x) for x in helper_methods]
            if (
                self.config.repair_sampling_parameters
                and self.prompter.PLANNING_HORIZON in self.diagnosis_result.keys()
            ):
                max_time_steps = self.diagnosis_result[self.prompter.PLANNING_HORIZON]
                max_time_steps = int(max_time_steps * 10)
            else:
                max_time_steps = self.motion_planner.max_time_steps
            if (
                self.config.repair_sampling_parameters
                and self.prompter.SAMPLING_D in self.diagnosis_result.keys()
            ):
                sampling_d = self.diagnosis_result[self.prompter.SAMPLING_D]
            else:
                sampling_d = self.motion_planner.d
        except Exception as _:
            raise MissingParameterException(self.prompter.COST_FUNCTION)
        self.last_motion_planner = ReactiveMotionPlanner(
            self.motion_planner.cost_function_string,
            self.motion_planner.helper_methods,
            self.motion_planner.max_time_steps,
            self.motion_planner.d
        )
        self.motion_planner = ReactiveMotionPlanner(
            updated_cost_function, helper_methods, max_time_steps, sampling_d
        )

    def describe_planner(
        self,
        update: bool = False,
        improved: bool = False,
    ):
        if update:
            if not self.last_motion_planner:
                last_cf = None
            else:
                last_cf = self.last_motion_planner.cost_function_string
            self.prompter.update_planner_prompt(
                self.motion_planner.cost_function_string,
                last_cf,
                self.config.feedback_mode,
            )
            if self.config.repair_sampling_parameters:
                self.prompter.update_config_prompt(self.motion_planner.max_time_steps)

    def plan(self, nr_iter: int) -> Union[PlanningProblemCostResult, Exception]:
        try:
            solution, missing_hf = self.motion_planner.evaluate_on_scenario(
                self.absolute_scenario_path, absolute_save_path=self.absolute_save_path
            )
            self.statistic.flawed_helper_methods_count += missing_hf
        except Exception as e:
            solution = e
            ReactiveMotionPlanner.create_plot(
                self.absolute_scenario_path, self.absolute_save_path
            )

        return solution

    def generate_emergency_prescription(self) -> str:
        return self.motion_planner.cost_function_string
