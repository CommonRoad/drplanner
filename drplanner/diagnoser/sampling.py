import os
import sys
import textwrap
from typing import Union

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad_dc.costs.evaluation import PlanningProblemCostResult

from drplanner.describer.base import MissingParameterException
from drplanner.diagnoser.base import DrPlannerBase
from drplanner.prompter.sampling import PrompterSampling
from drplanner.utils.config import DrPlannerConfiguration

# Adjust sys.path before any imports
current_file_dir = os.path.dirname(os.path.abspath(__file__))
smp_path = os.path.normpath(
    os.path.join(current_file_dir, "../../planners/reactive-planner/")
)
sys.path.append(smp_path)
print(f"[DrPlanner] Use the reactive planner under {smp_path}.")

try:
    import commonroad_rp.trajectories
except ImportError as e:
    print(
        f"Failed to import commonroad_rp.trajectories after adding {smp_path} to sys.path."
    )
    raise e  # Re-raise the exception or handle it appropriately

from drplanner.planners.reactive_planner_wrapper import ReactiveMotionPlannerWrapper, plot_planner


class DrSamplingPlanner(DrPlannerBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem_set: PlanningProblemSet,
        config: DrPlannerConfiguration,
    ):
        super().__init__(scenario, planning_problem_set, config)
        self.absolute_scenario_path = config.scenarios_path + str(scenario.scenario_id) + ".xml"
        self.motion_planner = ReactiveMotionPlannerWrapper()
        self.last_motion_planner = None

        self.prompter = PrompterSampling(
            self.scenario,
            self.planning_problem,
            self.config,
        )

    def repair(self):
        """
        Repair the motion planner by updating the cost function and helper methods.
        """
        if not self.diagnosis_result:
            raise MissingParameterException("diagnosis result")
        try:
            # retrieve cost function code or raise exception otherwise
            updated_cost_function = self.diagnosis_result[self.prompter.COST_FUNCTION]
            updated_cost_function = textwrap.dedent(updated_cost_function)
            helper_methods = self.diagnosis_result[self.prompter.HELPER_METHODS]
            helper_methods = [textwrap.dedent(x) for x in helper_methods]

            # try to retrieve planning horizon
            if (
                self.config.repair_sampling_parameters
                and self.prompter.PLANNING_HORIZON in self.diagnosis_result.keys()
            ):
                max_time_steps = self.diagnosis_result[self.prompter.PLANNING_HORIZON]
                max_time_steps = int(max_time_steps * 10)  # convert to time-steps
            else:
                max_time_steps = self.motion_planner.max_time_steps

            # try to retrieve allowed deviation from reference path
            if (
                self.config.repair_sampling_parameters
                and self.prompter.SAMPLING_D in self.diagnosis_result.keys()
            ):
                sampling_d = self.diagnosis_result[self.prompter.SAMPLING_D]
            else:
                sampling_d = self.motion_planner.d

        except Exception as _:
            raise MissingParameterException(self.prompter.COST_FUNCTION)

        self.last_motion_planner = ReactiveMotionPlannerWrapper(
            cost_function_string=self.motion_planner.cost_function_string,
            helper_methods=self.motion_planner.helper_methods,
            max_time_steps=self.motion_planner.max_time_steps,
            d=self.motion_planner.d,
        )
        self.motion_planner = ReactiveMotionPlannerWrapper(
            cost_function_string=updated_cost_function,
            helper_methods=helper_methods,
            max_time_steps=max_time_steps,
            d=sampling_d,
        )

    def describe_planner(self):
        """
        Describes the current state of the planner to the LLM.
        """
        if not self.last_motion_planner:
            last_cf = None
        else:
            last_cf = self.last_motion_planner.cost_function_string

        self.prompter.update_planner_prompt(
            self.motion_planner.cost_function_string,
            last_cf,
        )

        if self.config.repair_sampling_parameters:
            self.prompter.update_config_prompt(self.motion_planner.max_time_steps)

    def plan(self, nr_iter: int) -> Union[PlanningProblemCostResult, Exception]:
        """
        Wrapper method to run the sampling-based motion planner.
        """
        try:
            solution, missing_hf = self.motion_planner.evaluate_on_scenario(
                self.absolute_scenario_path, absolute_save_path=self.config.path_to_plot
            )
            self.statistic.flawed_helper_methods_count += missing_hf
        except Exception as e:
            solution = e
            # generate alternative plot if trajectory can not be obtained
            plot_planner(
                self.scenario,
                self.planning_problem_set,
                None,
                None,
                None,
                self.config.path_to_plot,
            )

        return solution
