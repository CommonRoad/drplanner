# for dynamically construct the import statement
import os
from abc import ABC, abstractmethod

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from describer.base import ExceptionDescription
from drplanner.prompter.llm import LLM, LLMFunction
from drplanner.describer.trajectory_description import TrajectoryCostDescription

from commonroad_dc.costs.evaluation import PlanningProblemCostResult


class PrompterBase(ABC):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        api_key: str,
        gpt_version: str = "gpt-3.5-turbo",  # gpt-3.5-turbo, text-davinci-002, gpt-4-1106-preview
        prompts_folder_name: str = "astar/",
        mockup=False,
    ):
        self.api_key = api_key
        self.gpt_version = gpt_version

        self.scenario = scenario
        self.planning_problem = planning_problem

        self.iteration_count = 0  # no iteration is used for the default one

        self.mockup = mockup
        self.llm_function = self.init_LLM()
        self.LLM = LLM(
            self.gpt_version, self.api_key, self.llm_function, mockup=self.mockup
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "system.txt"), "r") as file:
            self.prompt_system = file.read()

        with open(
            os.path.join(script_dir, prompts_folder_name + "template.txt"), "r"
        ) as file:
            self.algorithm_template = file.read()

        with open(
            os.path.join(script_dir, prompts_folder_name + "constraints.txt"), "r"
        ) as file:
            self.astar_constraints = file.read()

        with open(
            os.path.join(script_dir, prompts_folder_name + "few_shots.txt"), "r"
        ) as file:
            self.astar_few_shots = file.read()

        with open(
            os.path.join(script_dir, prompts_folder_name + "algorithm.txt"), "r"
        ) as file:
            self.astar_base = file.read()

        # replace the unchanged parts
        self.algorithm_template = self.algorithm_template.replace(
            "[CONSTRAINTS]", self.astar_constraints
        ).replace("[FEW_SHOTS]", self.astar_few_shots)

        self.trajectory_description = None

    def reload_LLM(self):
        print("*\t <LLM> The LLM is reloaded")
        self.LLM = LLM(
            self.gpt_version, self.api_key, self.llm_function, mockup=self.mockup
        )

    @abstractmethod
    def init_LLM(self) -> LLMFunction:
        pass

    @abstractmethod
    def generate_planner_description(self, *args, **kwargs) -> str:
        pass

    def generate_cost_description(
        self, cost_evaluation: PlanningProblemCostResult, desired_cost: float
    ):
        if not self.trajectory_description:
            self.trajectory_description = TrajectoryCostDescription(cost_evaluation)
        return self.trajectory_description.generate(desired_value=desired_cost)

    @staticmethod
    def update_cost_description(cost_evaluation: PlanningProblemCostResult):
        traj_des = TrajectoryCostDescription(cost_evaluation)
        return traj_des.update()

    @staticmethod
    def generate_exception_description(traceback: str):
        print("*\t !! Errors: ", traceback)
        exp_des = ExceptionDescription()
        return f" The planner failed: {exp_des.generate(traceback)}"
