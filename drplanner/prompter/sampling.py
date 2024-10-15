from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem

from drplanner.prompter.base import PrompterBase
from drplanner.prompter.llm import LLMFunction
from drplanner.utils.config import DrPlannerConfiguration


class PrompterSampling(PrompterBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem: PlanningProblem,
        config: DrPlannerConfiguration,
        prompts_folder_name: str = "reactive-planner",
    ):
        self.config = config
        self.COST_FUNCTION = "improved_cost_function"
        self.HELPER_METHODS = "helper_methods"
        self.PLANNING_HORIZON = "planning_horizon"
        self.SAMPLING_D = "lateral_distance"
        template = [
            "algorithm",
            "trajectory",
            "sampling",
            "documentation",
            "planner",
            "feedback",
        ]

        super().__init__(
            scenario,
            planning_problem,
            template,
            config,
            prompts_folder_name,
        )

    def init_LLM(self) -> LLMFunction:
        llm_function = LLMFunction()
        llm_function.add_code_parameter(self.COST_FUNCTION, "updated cost function")
        llm_function.add_array_parameter(
            self.HELPER_METHODS,
            "array to collect helper methods",
            llm_function.get_code_parameter("code of a custom helper method"),
        )
        if self.config.repair_sampling_parameters:
            llm_function.add_number_parameter(
                self.PLANNING_HORIZON, "planning horizon in [sec]"
            )
            llm_function.add_number_parameter(
                self.SAMPLING_D,
                "lateral distance to reference in interval [0;5] meters",
            )
        return llm_function

    def update_planner_prompt(self, cost_function, cost_function_previous):
        if not cost_function_previous:
            cf_code = (
                "This is the code of the current cost function:\n```\n"
                + cost_function
                + "```\n"
            )
        else:
            cf_code = "What follows is a comparison of two recent repairs you made:\n"
            cf_code += f"This is the current version:\n```\n" + cost_function + "```\n"
            cf_code += (
                f"This is the last version:\n```\n" + cost_function_previous + "```\n"
            )

        self.user_prompt.set("planner", cf_code)

    def update_config_prompt(self, time_steps_computation: int):
        """
        Describes the current state of sampling intervals of the reactive planner.
        """
        # standard prompt
        config_description = "You can also modify the length of the planning horizon. There are two options:\n"
        config_description += "If the planner failed, but you can not identify any specific reason for that, it might help to reset planning horizon to 3 seconds. "
        config_description += "If planning performance is stagnating even though major changes where made to the cost function, slightly increasing the planning horizon by 1 second might prove to be helpful.\n"
        config_description += "Furthermore, you can modify the lateral distance which the planner can move away from the goal."
        # describe the current planning horizon
        t_max = float(0.1 * time_steps_computation)
        config_description += f"The current planning horizon is {t_max} seconds."
        self.user_prompt.set("sampling", config_description)
