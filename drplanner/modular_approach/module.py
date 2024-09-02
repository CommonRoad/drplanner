import os
from abc import abstractmethod, ABC
from typing import Tuple

from commonroad.common.solution import CostFunction
from commonroad_dc.costs.evaluation import PlanningProblemCostResult

from describer.trajectory_description import get_infinite_cost_result
from drplanner.prompter.base import PrompterBase

from drplanner.describer.trajectory_description import TrajectoryCostDescription
from planners.reactive_planner import (
    ReactiveMotionPlanner,
    get_basic_configuration_path,
)
from utils.config import DrPlannerConfiguration


class Reflection:
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

    def to_few_shot(self) -> str:
        description = ""
        if self.scenario_analysis:
            description += f"SCENARIO: {self.scenario_analysis}\n"
        if self.evaluation_analysis:
            description += f"EVALUATION: {self.evaluation_analysis}\n"
        if self.cost_function_analysis:
            description += f"COST FUNCTION: {self.cost_function_analysis}\n"
        if self.problem:
            description += f"PROBLEM: {self.problem}\n"

        description += self.__str__()
        return description


class Module(ABC):
    def __init__(self, config: DrPlannerConfiguration):
        self.config = config
        self.separator = '"""\n'
        self.path_to_prompts = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prompts"
        )
        self.path_to_plot = os.path.join(
            os.path.dirname(self.config.project_path), "plots", "img.png"
        )

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


# module that evaluates reactive planner on a scenario
class EvaluationModule(Module):
    def __init__(self, config: DrPlannerConfiguration, cost_function_type: CostFunction = CostFunction.SM1):
        super().__init__(config)
        self.absolute_config_path = get_basic_configuration_path()
        self.cost_function_type = cost_function_type

    def run(
        self, absolute_scenario_path: str, motion_planner: ReactiveMotionPlanner, plot=True
    ) -> Tuple[str, PlanningProblemCostResult]:
        try:
            path_to_plot = self.path_to_plot
            if not plot:
                path_to_plot = None
            cost_result = motion_planner.evaluate_on_scenario(
                absolute_scenario_path,
                self.absolute_config_path,
                cost_type=self.cost_function_type,
                absolute_save_path=path_to_plot,
            )
            evaluation = TrajectoryCostDescription().generate(
                cost_result, self.config.desired_cost
            )
        except Exception as e:
            evaluation = PrompterBase.generate_exception_description(e)
            cost_result = get_infinite_cost_result(self.cost_function_type)
        return evaluation, cost_result
