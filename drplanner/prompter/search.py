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

# make sure the SMP has been installed successfully
try:
    import SMP

    print("[DrPlanner] Installed SMP module is called.")
except ImportError as e:
    import sys
    import os

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    smp_path = os.path.join(current_file_dir, "../../commonroad-search/")
    sys.path.append(smp_path)
    print(f"[DrPlanner] Use the external submodule SMP under {smp_path}.")

from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch


class PrompterSearch(PrompterBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        api_key: str,
        gpt_version: str = "gpt-4-1106-preview",
        prompts_folder_name: str = "astar/",
        mockup: bool = False,
    ):
        # the parameters which the llm has to provide in a response
        self.HEURISTIC_FUNCTION = "improved_heuristic_function"
        self.MOTION_PRIMITIVES = "motion_primitives"
        self.EXTRA_INFORMATION = "extra_information"
        super().__init__(
            scenario,
            planning_problem,
            api_key,
            gpt_version,
            prompts_folder_name,
            mockup=mockup,
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

    def generate_planner_description(
        self, motion_planner_obj: Union[object, AStarSearch], motion_primitives_id: str
    ) -> str:
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
        return (
            self.astar_base + hf_code + heuristic_function_des + motion_primitives_des
        )
