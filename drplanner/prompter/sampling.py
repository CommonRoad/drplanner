# for dynamically construct the import statement
import os
from typing import Union
import inspect

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad_rp.cost_function import DefaultCostFunction, CostFunction

from drplanner.describer.planner_description import (
    CostFunctionDescription,
)
from drplanner.prompter.llm import LLM
from drplanner.describer.trajectory_description import TrajectoryCostDescription

from commonroad_dc.costs.evaluation import PlanningProblemCostResult


class PrompterSampling:
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        api_key: str,
        gpt_version: str = "gpt-4-1106-preview",  # gpt-3.5-turbo, text-davinci-002, gpt-4-1106-preview
        prompts_folder_name: str = "reactive-planner/",
    ):
        self.api_key = api_key
        self.gpt_version = gpt_version

        self.scenario = scenario
        self.planning_problem = planning_problem

        self.iteration_count = 0  # no iteration is used for the default one

        self.LLM = LLM(self.gpt_version, self.api_key)

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

    def reload_LLM(self):
        print("*\t <LLM> The LLM is reloaded")
        self.LLM = LLM(self.gpt_version, self.api_key)

    def generate_planner_description(
        self, cost_function_obj: Union[object, CostFunction]
    ) -> str:
        hf_code = (
            "This is the code of the cost function: ```"
            + inspect.getsource(cost_function_obj.evaluate)
            + "```"
        )

        # generate heuristic function's description
        hf_obj = CostFunctionDescription(cost_function_obj.evaluate)
        heuristic_function_des = hf_obj.generate(cost_function_obj)

        return self.astar_base + hf_code + heuristic_function_des

    def generate_reactive_planner_description(
        self, motion_planner_obj: Union[object, DefaultCostFunction]
    ) -> str:
        pass

    @staticmethod
    def generate_cost_description(
        cost_evaluation: PlanningProblemCostResult, desired_cost: float
    ):
        traj_des = TrajectoryCostDescription(cost_evaluation)
        return traj_des.generate(desired_value=desired_cost)

    @staticmethod
    def update_cost_description(cost_evaluation: PlanningProblemCostResult):
        traj_des = TrajectoryCostDescription(cost_evaluation)
        return traj_des.update()
