# for dynamically construct the import statement
from typing import Union
import inspect

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from drplanner.describer.planner_description import (
    HeuristicDescription,
    MotionPrimitiveDescription,
)
from drplanner.prompter.base import PrompterBase
from drplanner.prompter.llm import LLMFunction
from drplanner.utils.config import DrPlannerConfiguration


class PrompterSearch(PrompterBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        config: DrPlannerConfiguration,
        prompts_folder_name: str = "astar/",
    ):
        # the parameters which the llm has to provide in a response
        self.HEURISTIC_FUNCTION = "improved_heuristic_function"
        self.MOTION_PRIMITIVES = "motion_primitives"
        self.EXTRA_INFORMATION = "extra_information"
        template = [
            "constraints",
            "algorithm",
            "planner",
            "trajectory",
            "few_shots",
            "feedback",
        ]
        super().__init__(
            scenario,
            planning_problem,
            template,
            config,
            prompts_folder_name,
        )

        self.mp_obj = MotionPrimitiveDescription()

    def init_LLM(self) -> LLMFunction:
        llm_function = LLMFunction()
        llm_function.add_code_parameter(
            self.HEURISTIC_FUNCTION, "updated heuristic function"
        )
        llm_function.add_string_parameter(
            self.MOTION_PRIMITIVES, "name of the new motion primitives"
        )
        llm_function.add_string_parameter(self.EXTRA_INFORMATION, "extra information")
        return llm_function

    def update_planner_prompt(
        self,
        motion_planner_obj: Union[object, "AStarSearch"],
        motion_primitives_id: str,
    ):
        # Local import inside the function
        hf_code = (
            "This is the code of the heuristic function: ```"
            + inspect.getsource(motion_planner_obj.heuristic_function)
            + "```"
        )

        # generate heuristic function's description
        hf_obj = HeuristicDescription(motion_planner_obj.heuristic_function)
        heuristic_function_des = hf_obj.generate(motion_planner_obj)

        # generate motion primitives' description
        motion_primitives_des = self.mp_obj.generate(motion_primitives_id)
        self.user_prompt.set(
            "planner", hf_code + heuristic_function_des + motion_primitives_des
        )
