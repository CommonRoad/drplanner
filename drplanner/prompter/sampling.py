import inspect
import textwrap

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
        temperature: float,
        gpt_version: str = "gpt-4-1106-preview",
        prompts_folder_name: str = "reactive-planner/",
        mockup: bool = False,
    ):
        self.COST_FUNCTION = "improved_cost_function"
        self.PLANNER_CONFIG = [
            ("t_min", "minimal time horizon in [s]"),
            ("t_max", "maximal time horizon in [s]"),
            ("d_bound", "range bound for distance to reference in [m]"),
        ]

        super().__init__(
            scenario,
            planning_problem,
            api_key,
            gpt_version,
            prompts_folder_name,
            mockup=mockup,
            temperature=temperature,
        )

    def init_LLM(self) -> LLMFunction:
        llm_function = LLMFunction()
        llm_function.add_code_parameter(self.COST_FUNCTION, "updated cost function")
        # add sampling configuration parameters
        for key, descr in self.PLANNER_CONFIG:
            llm_function.add_number_parameter(key, descr)
        return llm_function

    def generate_planner_description(
        self, cost_function, config: ReactivePlannerConfiguration
    ):
        # describe the current planning horizon
        t_max = float(config.planning.dt * config.planning.time_steps_computation)
        config_description = "These are the current intervals used for sampling: "
        config_description += f"Time horizon starts at {config.sampling.t_min} seconds and ends at {t_max} seconds. "
        if config.sampling.t_min <= 2 * config.planning.dt:
            config_description += (
                "In this case, t_min can longer be decreased since it is at a minimum. "
            )
        config_description += f"The car can be between [{config.sampling.d_min}, {config.sampling.d_max}] meters away from the reference path. "
        # config_description += f"The car's velocity can deviate from the target velocity in this range: [{config.sampling.v_min}, {config.sampling.v_max}]."

        # if code is directly provided
        if isinstance(cost_function, str):
            return self.astar_base + "\n" + cost_function
        # otherwise access it using "inspect" and describe its used methods
        else:
            cf_code = (
                "This is the code of the cost function:\n```\n"
                + textwrap.dedent(inspect.getsource(cost_function.evaluate))
                + "```"
            )
            # generate heuristic function's description
            # hf_obj = CostFunctionDescription(cost_function.evaluate)
            # heuristic_function_des = hf_obj.generate(cost_function)
            return (
                self.astar_base
                + "\n"
                + cf_code
                + "\n"
                + config_description
                # + heuristic_function_des
            )
