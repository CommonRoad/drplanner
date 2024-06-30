import inspect
import textwrap

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad_rp.utility.config import ReactivePlannerConfiguration

from drplanner.prompter.base import PrompterBase
from drplanner.prompter.llm import LLMFunction
from drplanner.utils.config import DrPlannerConfiguration


class PrompterSampling(PrompterBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        config: DrPlannerConfiguration,
        prompts_folder_name: str = "reactive-planner/",
    ):
        self.config = config
        self.COST_FUNCTION = "improved_cost_function"
        self.PLANNER_CONFIG = [
            ("t_min", "minimal time horizon in [s]"),
            ("t_max", "maximal time horizon in [s]"),
            ("d_bound", "range bound for distance to reference in [m]"),
        ]
        template = [
            "constraints",
            "algorithm",
            "planner",
            "trajectory",
            "few_shots",
            "sampling",
            "feedback",
        ]

        super().__init__(
            scenario,
            planning_problem,
            template,
            config.openai_api_key,
            config.gpt_version,
            prompts_folder_name,
            mockup=config.mockup_openAI,
            temperature=config.temperature,
        )

    def init_LLM(self) -> LLMFunction:
        llm_function = LLMFunction()
        llm_function.add_code_parameter(self.COST_FUNCTION, "updated cost function")
        if self.config.repair_sampling_parameters:
            # add sampling configuration parameters
            for key, descr in self.PLANNER_CONFIG:
                llm_function.add_number_parameter(key, descr)
        return llm_function

    def update_planner_prompt(self, cost_function, cost_function_previous: str, feedback_mode: int):
        # if code is directly provided
        if isinstance(cost_function, str):
            if feedback_mode < 3 or cost_function == cost_function_previous:
                if feedback_mode < 2:
                    version = "current"
                else:
                    version = "currently best performing"
                cf_code = (
                        f"This is the code of the {version} cost function:\n```\n"
                        + cost_function
                        + "```\n"
                        + "Adjust it to decrease costs."
                )
            else:
                cf_code = (
                        f"This is the current version of the cost function:\n```\n"
                        + cost_function
                        + "```\n"
                )
                cf_code += (
                        f"Now for comparison, this is the code of the currently best performing cost function:\n```\n"
                        + cost_function_previous
                        + "```\n"
                        + "Compare the two version to identify which partial costs are most important and which changes were beneficial!"
                )

        # otherwise access it using "inspect" and describe its used methods
        else:
            cf_code = (
                "This was the code of the initial cost function:\n```\n"
                + textwrap.dedent(inspect.getsource(cost_function.evaluate))
                + "```\n"
                + "Adjust it to decrease costs."
            )
        self.user_prompt.set("planner", cf_code)

    def update_config_prompt(self, config: ReactivePlannerConfiguration):
        # standard prompt
        config_description = (
            "You can also modify the current sampling intervals. "
            "These intervals determine the properties of sampled trajectories and are "
            "therefore responsible for choosing the initial pool of trajectories "
            "(which is then ranked by the cost_function).\n"
            "These are the intervals:\n"
            "1) time horizon [t_min, t_max] in seconds. Increasing t_max will allow for "
            "long, smooth, low-curvature trajectories which are good for following straight paths. "
            "Decreasing t_min will allow for the opposite which is beneficial for maneuvering "
            "through dense traffic etc.\n"
            "2) distance to reference path [-d_max, d_max] in meters. Increasing d_max will give "
            "the car more freedom since it can now roam far away from the reference path. But this "
            "might also lead to the car missing the goal region or driving physically impossible trajectories.\n"
            "Feel free to increase these parameters if everything works well"
        )
        # describe the current planning horizon
        t_max = float(config.planning.dt * config.planning.time_steps_computation)
        config_description += "These are the current intervals used for sampling: "
        config_description += f"Time horizon starts at {config.sampling.t_min} seconds and ends at {t_max} seconds. "
        if config.sampling.t_min <= 2 * config.planning.dt:
            config_description += (
                "In this case, t_min can longer be decreased since it is at a minimum. "
            )
        config_description += f"The car can be between [{config.sampling.d_min}, {config.sampling.d_max}] meters away from the reference path. "
        self.user_prompt.set("sampling", config_description)
