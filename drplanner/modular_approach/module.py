import os
import traceback
from abc import abstractmethod, ABC
from typing import Tuple

from commonroad.common.solution import CostFunction
from commonroad_dc.costs.evaluation import PlanningProblemCostResult

from drplanner.describer.trajectory_description import (
    get_infinite_cost_result,
    TrajectoryCostDescription,
)
from drplanner.prompter.base import PrompterBase

from drplanner.planners.reactive_planner_wrapper import (
    get_basic_configuration_path,
    ReactiveMotionPlannerWrapper,
)
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.utils.general import Statistics


class Reflection:
    """
    Class representing a reflection on the current iteration's repair process.
    """

    def __init__(
        self,
        summary: str,
        diagnosis_reflection: str = "",
        repair_reflection: str = "",
    ):
        self.summary = summary
        self.diagnosis_reflection = diagnosis_reflection
        self.repair_reflection = repair_reflection


class Diagnosis:
    """
    Class collecting all information generated by the diagnosis module,
    including the specific prescriptions for the repair module.
    """

    def __init__(
        self,
        scenario_analysis: str,
        evaluation_analysis: str,
        cost_function_analysis: str,
        problem: str,
        prescriptions: list[str],
    ):
        self.scenario_analysis = scenario_analysis
        self.evaluation_analysis = evaluation_analysis
        self.cost_function_analysis = cost_function_analysis
        self.problem = problem
        self.prescriptions = prescriptions
        self.separator = '"""\n'

    def __str__(self):
        description = "These are the solution steps proposed to repair the planner:\n"
        for step in self.prescriptions:
            description += f"{step}\n"
        return f"{self.separator}{description}{self.separator}"

    def to_few_shot(self, max_time_steps: int, d: float) -> str:
        """
        __str__ alternative to store the diagnosis as a single string in a database.
        """
        description = ""
        if self.evaluation_analysis:
            description += f"EVALUATION: {self.evaluation_analysis}\n"
        if self.cost_function_analysis:
            description += f"COST FUNCTION: {self.cost_function_analysis}\n"
        if self.problem:
            description += f"PROBLEM: {self.problem}\n"
        if len(self.__str__()) > 8:
            description += self.__str__()[4:-4]
        planning_horizon = float(max_time_steps / 10)
        description += f"Concerning the planning horizon, {planning_horizon} seconds is advisable. "
        description += f"The lateral deviation maximum should be {d} meters."
        return description


class Module(ABC):
    """
    Abstract base class for all modules.
    """

    def __init__(self, config: DrPlannerConfiguration, statistic: Statistics):
        self.config = config
        self.statistic = statistic
        self.separator = '"""\n'
        self.path_to_prompts = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prompts"
        )

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class EvaluationModule(Module):
    """
    Module responsible for error handling and evaluating a motion planner
    on a given benchmark.
    """

    def __init__(
        self,
        config: DrPlannerConfiguration,
        statistic: Statistics,
        cost_function_type: CostFunction = CostFunction.SM1,
    ):
        super().__init__(config, statistic)
        self.absolute_config_path = get_basic_configuration_path()
        self.cost_function_type = cost_function_type

    def run(
        self,
        absolute_scenario_path: str,
        motion_planner: ReactiveMotionPlannerWrapper,
        plot=True,
    ) -> Tuple[str, PlanningProblemCostResult, str]:
        exception_description = ""
        try:
            path_to_plot = self.config.path_to_plot
            if not plot:
                path_to_plot = None
            cost_result, missing_hm = motion_planner.evaluate_on_scenario(
                absolute_scenario_path,
                self.absolute_config_path,
                cost_type=self.cost_function_type,
                absolute_save_path=path_to_plot,
            )
            self.statistic.flawed_helper_methods_count += missing_hm
            evaluation = TrajectoryCostDescription().generate(cost_result)
        except Exception as e:
            print(traceback.format_exc())
            evaluation = PrompterBase.generate_exception_description(e)
            cost_result = get_infinite_cost_result(self.cost_function_type)
            exception_description = e.__class__.__name__
        return evaluation, cost_result, exception_description
