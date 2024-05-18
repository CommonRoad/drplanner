# for dynamically construct the import statement
from typing import Union
import inspect

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad_rp.cost_function import CostFunction

from drplanner.describer.planner_description import (
    CostFunctionDescription,
)
from drplanner.prompter.base import PrompterBase
from drplanner.prompter.llm import LLMFunction


class PrompterSampling(PrompterBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        api_key: str,
        mockup: bool,
        gpt_version: str = "gpt-4-1106-preview",
        prompts_folder_name: str = "reactive-planner/",
    ):
        self.COST_FUNCTION = "improved_cost_function"
        self.EXTRA_INFORMATION = "extra_information"

        super().__init__(
            scenario,
            planning_problem,
            api_key,
            gpt_version,
            prompts_folder_name,
            mockup=mockup,
        )

    def init_LLM(self) -> LLMFunction:
        llm_function = LLMFunction()
        llm_function.add_code_parameter(self.COST_FUNCTION, "updated cost function")
        llm_function.add_string_parameter(self.EXTRA_INFORMATION, "extra information")
        return llm_function

    def generate_planner_description(
        self, cost_function_obj: Union[object, CostFunction], hf_code: str
    ) -> str:

        if cost_function_obj is None:
            return self.astar_base + "\n" + hf_code
        else:
            hf_code = (
                "This is the code of the cost function: ```"
                + inspect.getsource(cost_function_obj.evaluate)
                + "```"
            )

            # generate heuristic function's description
            hf_obj = CostFunctionDescription(cost_function_obj.evaluate)
            heuristic_function_des = hf_obj.generate(cost_function_obj)

            return self.astar_base + "\n" + hf_code + "\n" + heuristic_function_des
