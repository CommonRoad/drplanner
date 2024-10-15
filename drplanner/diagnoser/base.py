import copy
import os
import time
from typing import Union

import numpy as np
from abc import ABC, abstractmethod

from commonroad.common.solution import VehicleType, VehicleModel, CostFunction
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad_dc.costs.evaluation import (
    CostFunctionEvaluator,
    PlanningProblemCostResult,
)

from drplanner.describer.base import MissingParameterException
from drplanner.describer.trajectory_description import get_infinite_cost_result
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.prompter.search import PrompterSearch
from drplanner.prompter.llm import LLM
from drplanner.utils.gpt import num_tokens_from_messages
from drplanner.memory.memory import FewShotMemory
from drplanner.utils.general import Statistics


class DrPlannerBase(ABC):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem_set: PlanningProblemSet,
        config: DrPlannerConfiguration,
    ):
        self.memory = FewShotMemory()
        self.statistic = Statistics()
        self.scenario = scenario
        self.planning_problem_set = planning_problem_set
        # otherwise the planning problem might be changed during the initialization of the planner
        self.planning_problem = copy.deepcopy(
            list(self.planning_problem_set.planning_problem_dict.values())[0]
        )
        self.config = config
        self.include_plot = False

        self._visualize = self.config.visualize
        self._save_solution = self.config.save_solution

        self.desired_cost = self.config.desired_cost
        self.current_cost = None
        self.initial_cost = None

        self.token_count = 0
        self.cost_list = []

        self.dir_output = os.path.join(os.path.dirname(__file__), "../../outputs/")
        os.makedirs(
            os.path.dirname(self.dir_output), exist_ok=True
        )  # Ensure the directory exists

        self.prompter = PrompterSearch(
            self.scenario,
            self.planning_problem,
            self.config,
        )

        # standard parameters to evaluate a planning result
        self.cost_type = CostFunction.SM1
        self.vehicle_model = VehicleModel.KS
        self.vehicle_type = VehicleType.BMW_320i
        self.cost_evaluator = CostFunctionEvaluator(
            self.cost_type, VehicleType.BMW_320i
        )
        # noinspection PyTypeChecker
        self.cost_result_current: PlanningProblemCostResult = None
        # noinspection PyTypeChecker
        self.cost_result_previous: PlanningProblemCostResult = None
        # stores the last llm response
        self.diagnosis_result = None

    @abstractmethod
    def repair(self):
        """
        Updates last and current motion planner according to LLM output.
        raise: MissingParameterException
        """
        pass

    @abstractmethod
    def describe_planner(self):
        """
        Describes the current state of the planner to the LLM.
        """
        pass

    @abstractmethod
    def plan(self, nr_iter: int) -> Union[PlanningProblemCostResult, Exception]:
        """
        Wrapper method to run the motion planner.
        """
        pass

    def describe_trajectory(self, planned_trajectory: Union[None, Exception]):
        """
        Describes a single cost result to the LLM.
        Should be used once and at the start.
        """
        if not planned_trajectory:
            description = self.prompter.trajectory_description.generate(
                self.cost_result_current
            )
            if self.include_plot:
                description += "To give you a broad understanding of the scenario which is currently used for motion planning, a plot is provided showing all lanes (grey), the planned trajectory (black line) and the goal area (light orange)"
        else:
            description = "Usually here would be an evaluation of the initial motion planning result, but..."
            description += self.prompter.generate_exception_description(
                planned_trajectory
            )

        self.prompter.user_prompt.set("feedback", description)

    def add_feedback(self, cost_result: PlanningProblemCostResult):
        """
        Evaluates the result of the repair process.
        """
        self.cost_result_current = cost_result

        feedback = self.prompter.trajectory_comparison.generate(
            self.cost_result_previous,
            self.cost_result_current,
        )
        indented_feedback = "\n\t\t".join(feedback.splitlines())
        # Print the feedback with indentation
        print(f"*\t Feedback:\n\t\t{indented_feedback}")
        return feedback

    def diagnose_repair(self):
        """
        Full DrPlanner session using repair-cycle:
        Repair. Plan. Evaluate.
        Runs until the patient is cured, or the doctor runs out of tokens/time.
        """
        nr_iteration = 0
        result = None

        path_to_plot = self.config.path_to_plot
        if not self.include_plot:
            path_to_plot = None

        # test the initial motion planner once
        planning_result = self.plan(nr_iteration)
        if isinstance(planning_result, Exception):
            self.cost_result_current = get_infinite_cost_result(self.cost_type)
            self.cost_result_previous = self.cost_result_current
            self.describe_planner()
            self.describe_trajectory(planning_result)
            self.current_cost = np.inf
        else:
            self.cost_result_current = planning_result
            self.cost_result_previous = self.cost_result_current
            self.describe_planner()
            self.describe_trajectory(None)
            self.current_cost = self.cost_result_current.total_costs

        start_time = time.time()
        self.initial_cost = self.current_cost

        # start the repairing process
        while (
            nr_iteration < self.config.iteration_max
            and self.token_count < self.config.token_limit
        ):
            print(f"*\t -----------iteration {nr_iteration}-----------")
            print(
                f"*\t <{nr_iteration}>: total cost {self.current_cost} (desired: {self.desired_cost})\n"
                f"*\t used tokens {self.token_count} (limit: {self.config.token_limit})"
            )
            # prepare messages for API-request
            messages = LLM.get_messages(
                self.prompter.system_prompt.__str__(),
                self.prompter.user_prompt.__str__(),
                path_to_plot,
            )
            self.token_count += num_tokens_from_messages(
                LLM.extract_text_from_messages(messages),
                self.prompter.LLM.gpt_version,
            )

            # send request and receive response
            scenario_id = str(self.scenario.scenario_id)
            save_dir = os.path.join(
                self.config.save_dir, scenario_id, self.config.gpt_version
            )
            mock_up_path = ""
            if self.config.mockup_openAI:
                mock_up_path = save_dir

            result = self.prompter.LLM.query(
                messages,
                save_dir=save_dir,
                nr_iter=nr_iteration,
                path_to_plot=path_to_plot,
                mockup_path=mock_up_path,
            )
            self.diagnosis_result = result

            # in case the LLM is not able to provide a response
            # self.prompter.reload_LLM()
            nr_iteration += 1

            # apply the response and evaluate it
            try:
                self.repair()
                self.describe_planner()

                cost_result = self.plan(nr_iteration)
                # if exception occurred during planning, fail
                if isinstance(cost_result, Exception):
                    raise cost_result

                prompt_feedback = self.add_feedback(cost_result)
                self.current_cost = self.cost_result_current.total_costs
                self.cost_result_previous = self.cost_result_current

                self.statistic.update_iteration(cost_result.total_costs)

            except Exception as e:
                self.describe_planner()

                prompt_feedback = (
                    "Usually here would be an evaluation of the repair, but..."
                )
                exception_description = self.prompter.generate_exception_description(e)
                prompt_feedback += f"{exception_description}\n"
                self.cost_result_current = get_infinite_cost_result(self.cost_type)
                self.current_cost = np.inf

                if isinstance(e, MissingParameterException):
                    self.statistic.missing_parameter_count += 1
                self.statistic.update_iteration(e.__class__.__name__)

            self.prompter.user_prompt.set("feedback", prompt_feedback)
            self.cost_list.append(self.current_cost)
            print(f"*\t ---------------------------------")

        print("[DrPlanner] Ends.")
        end_time = time.time()
        duration = end_time - start_time
        self.statistic.duration = duration
        self.statistic.token_count = self.token_count
        return result
