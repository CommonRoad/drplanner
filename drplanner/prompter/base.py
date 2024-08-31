import os
from abc import ABC, abstractmethod

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from drplanner.describer.base import (
    ExceptionDescription,
    DrPlannerException,
)
from drplanner.prompter.llm import LLM, LLMFunction
from drplanner.describer.trajectory_description import TrajectoryCostDescription

from commonroad_dc.costs.evaluation import PlanningProblemCostResult


class Prompt:
    def __init__(self, template: list):
        # standard structure of a prompt
        self._content: dict[str, str] = {}
        for key in template:
            self._content[key] = ""

    def __str__(self) -> str:
        prompt = ""
        for value in self._content.values():
            if not value:
                continue
            prompt += value + "\n"
        return prompt

    def set(self, key, value):
        if key not in self._content.keys():
            return
        self._content[key] = value


class PrompterBase(ABC):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        user_prompt_template: list[str],
        api_key: str,
        gpt_version: str = "gpt-3.5-turbo",  # gpt-3.5-turbo, text-davinci-002, gpt-4-1106-preview
        prompts_folder_name: str = "astar/",
        temperature=0.2,
        mockup=False,
    ):
        self.api_key = api_key
        self.gpt_version = gpt_version
        self.temperature = temperature

        self.scenario = scenario
        self.planning_problem = planning_problem

        self.iteration_count = 0  # no iteration is used for the default one

        self.mockup = mockup
        self.llm_function = self.init_LLM()
        self.LLM = LLM(
            self.gpt_version,
            self.api_key,
            self.llm_function,
            mockup=self.mockup,
            temperature=self.temperature,
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.system_prompt = Prompt(["base"])
        with open(os.path.join(script_dir, "system.txt"), "r") as file:
            self.system_prompt.set("base", file.read())

        self.user_prompt = Prompt(user_prompt_template)

        for part in user_prompt_template:
            template_path = os.path.join(
                script_dir, prompts_folder_name + f"{part}.txt"
            )
            if os.path.exists(template_path):
                with open(template_path, "r") as file:
                    self.user_prompt.set(part, file.read())

        self.trajectory_description = TrajectoryCostDescription()

    def reload_LLM(self):
        print("*\t <LLM> The LLM is reloaded")
        self.LLM = LLM(
            self.gpt_version,
            self.api_key,
            self.llm_function,
            mockup=self.mockup,
            temperature=self.temperature,
        )

    @abstractmethod
    def init_LLM(self) -> LLMFunction:
        pass

    @abstractmethod
    def update_planner_prompt(self, *args, **kwargs) -> str:
        pass

    def generate_cost_description(
        self, initial: PlanningProblemCostResult, desired_cost: int
    ) -> str:
        return self.trajectory_description.generate(initial, desired_cost)

    def update_cost_description(
        self,
        a: PlanningProblemCostResult,
        b: PlanningProblemCostResult,
        desired_cost: int,
        a_version: str = "initial",
    ) -> str:
        return self.trajectory_description.compare(a, a_version, b, desired_cost)

    @staticmethod
    def generate_exception_description(e: Exception):
        description = "\n"
        description += "!AN EXCEPTION OCCURRED!\n"

        if isinstance(e, DrPlannerException):
            description += e.describe()
        else:
            exp_des = ExceptionDescription(e)
            description += str(e) + "\n"
            description += exp_des.generate()

        print(description)
        return description
