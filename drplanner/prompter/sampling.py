import inspect

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from drplanner.prompter.base import PrompterBase
from drplanner.prompter.llm import LLMFunction


class PrompterSampling(PrompterBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        api_key: str,
        gpt_version: str = "gpt-4-1106-preview",
        prompts_folder_name: str = "reactive-planner/",
        mockup: bool = False,
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

    def generate_planner_description(self, cost_function) -> str:
        # todo: evaluate whether any more detailed description is necessary
        # if code is directly provided
        if isinstance(cost_function, str):
            return self.astar_base + "\n" + cost_function
        # otherwise access it using "inspect"
        else:
            cf_code = (
                "This is the code of the cost function: ```"
                + inspect.getsource(cost_function.evaluate)
                + "```"
            )
            return self.astar_base + "\n" + cf_code + "\n"
