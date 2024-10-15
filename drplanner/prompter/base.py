import os
from abc import ABC, abstractmethod

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from drplanner.describer.trajectory_description import (
    TrajectoryCostComparison,
    TrajectoryCostDescription,
)
from drplanner.describer.base import (
    ExceptionDescription,
    DrPlannerException,
)
from drplanner.prompter.llm import LLM, LLMFunction

from drplanner.utils.config import DrPlannerConfiguration


class Prompt:
    """
    Class representing a prompt consisting of multiple string sections.
    This allows for easy reordering of different sections.
    """

    def __init__(self, template: list):
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

    def append(self, key, value):
        if key not in self._content.keys():
            return
        self._content[key] += value


class PrompterBase(ABC):
    """
    Prompter base class responsible for managing the input prompts for an LLM.
    """

    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        user_prompt_template: list[str],
        config: DrPlannerConfiguration,
        prompts_folder_name: str = "astar",
    ):
        self.config = config

        self.scenario = scenario
        self.planning_problem = planning_problem

        self.llm_function = self.init_LLM()  # initialize in sub-classes
        self.LLM = LLM(
            self.config.gpt_version,
            self.config.openai_api_key,
            self.llm_function,
            mockup=self.config.mockup_openAI,
            temperature=self.config.temperature,
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.system_prompt = Prompt(["base"])
        with open(
            os.path.join(script_dir, prompts_folder_name, "system.txt"), "r"
        ) as file:
            self.system_prompt.set("base", file.read())

        self.user_prompt = Prompt(user_prompt_template)

        # check if there exists a pre-defined text for a specific user prompt section
        for part in user_prompt_template:
            template_path = os.path.join(script_dir, prompts_folder_name, f"{part}.txt")
            if os.path.exists(template_path):
                with open(template_path, "r") as file:
                    self.user_prompt.set(part, file.read())

        #
        self.trajectory_description = TrajectoryCostDescription(
            self.config.desired_cost
        )
        self.trajectory_comparison = TrajectoryCostComparison(self.config.desired_cost)

    def reload_LLM(self):
        print("*\t <LLM> The LLM is reloaded")
        self.LLM = LLM(
            self.config.gpt_version,
            self.config.openai_api_key,
            self.llm_function,
            mockup=self.config.mockup_openAI,
            temperature=self.config.temperature,
        )

    @abstractmethod
    def init_LLM(self) -> LLMFunction:
        """
        Method responsible for determining the output structure of the LLM.
        """
        pass

    @abstractmethod
    def update_planner_prompt(self, *args, **kwargs) -> str:
        """
        Updates the planner description section in the user prompt.
        """
        pass

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

        return description
