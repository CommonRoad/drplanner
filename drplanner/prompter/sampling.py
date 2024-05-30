import inspect

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad_rp.utility.config import ReactivePlannerConfiguration

from drplanner.describer.planner_description import CostFunctionDescription

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
        self.PLANNER_CONFIG = "planner_config"
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
        llm_function.add_number_parameter(self.PLANNER_CONFIG, "time step amount")
        llm_function.add_string_parameter(self.EXTRA_INFORMATION, "extra information")
        return llm_function

    def generate_planner_description(
        self, cost_function, config: ReactivePlannerConfiguration
    ):
        # describe the current planning horizon
        config_description = f"The current planning horizon length in time-steps is {config.planning.time_steps_computation}"

        # if code is directly provided
        if isinstance(cost_function, str):
            return self.astar_base + "\n" + cost_function
        # otherwise access it using "inspect" and describe its used methods
        else:
            cf_code = (
                "This is the code of the cost function: ```"
                + inspect.getsource(cost_function.evaluate)
                + "```"
            )
            # generate heuristic function's description
            hf_obj = CostFunctionDescription(cost_function.evaluate)
            heuristic_function_des = hf_obj.generate(cost_function)
            return (
                self.astar_base
                + "\n"
                + cf_code
                + "\n"
                + heuristic_function_des
                # + "\n"
                # + config_description
            )
