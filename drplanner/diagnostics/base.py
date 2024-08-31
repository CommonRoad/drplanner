import math
import copy
import os
from datetime import datetime
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

from drplanner.describer.trajectory_description import get_infinite_cost_result
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.prompter.search import PrompterSearch
from drplanner.prompter.llm import LLM
from drplanner.utils.gpt import num_tokens_from_messages
from drplanner.memory.memory import FewShotMemory


class DrPlannerBase(ABC):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem_set: PlanningProblemSet,
        config: DrPlannerConfiguration,
        planner_id: str,
    ):
        self.memory = FewShotMemory()
        self.scenario = scenario
        self.planning_problem_set = planning_problem_set
        # otherwise the planning problem might be changed during the initialization of the planner
        self.planning_problem = copy.deepcopy(
            list(self.planning_problem_set.planning_problem_dict.values())[0]
        )
        self.config = config

        self.planner_id = planner_id

        self._visualize = self.config.visualize
        self._save_solution = self.config.save_solution

        self.THRESHOLD = config.cost_threshold
        self.TOKEN_LIMIT = config.token_limit
        self.ITERATION_MAX = config.iteration_max

        # todo: load from solution file
        self.desired_cost = self.config.desired_cost
        self.initial_cost = math.inf
        self.current_cost = None
        self.lowest_cost = math.inf

        self.token_count = 0
        self.cost_list = []

        self.dir_output = os.path.join(os.path.dirname(__file__), "../../outputs/")
        os.makedirs(
            os.path.dirname(self.dir_output), exist_ok=True
        )  # Ensure the directory exists

        # initialize prompter
        self.prompter = PrompterSearch(
            self.scenario,
            self.planning_problem,
            self.config.openai_api_key,
            self.config.temperature,
            self.config.gpt_version,
            mockup=self.config.mockup_openAI,
        )

        # standard parameters to evaluate a planning result
        self.cost_type = CostFunction.SM1
        self.vehicle_model = VehicleModel.KS
        self.vehicle_type = VehicleType.BMW_320i
        self.cost_evaluator = CostFunctionEvaluator(
            self.cost_type, VehicleType.BMW_320i
        )
        # stores the cost evaluation of the last generated solution
        # noinspection PyTypeChecker
        self.cost_result_current: PlanningProblemCostResult = None
        # stores some benchmark cost result to compare against
        # noinspection PyTypeChecker
        self.cost_result_previous: PlanningProblemCostResult = None
        # stores the last generated llm response
        self.diagnosis_result = None

    @abstractmethod
    def repair(self):
        """
        Tries to implement the recommendations by DrPlanner
        """
        pass

    @abstractmethod
    def describe_planner(
        self,
        update: bool = False,
        improved: bool = False,
    ):
        """
        Describes the current state of the planner to DrPlanner
        """
        pass

    @abstractmethod
    def plan(self, nr_iter: int) -> Union[PlanningProblemCostResult, Exception]:
        """
        Wrapper method to run the motion planner
        """
        pass

    def describe_trajectory(self, planned_trajectory: Union[None, Exception]):
        """
        Describes the state of the initial planner. Should only be used once at the start.
        """
        if not planned_trajectory:
            description = self.prompter.generate_cost_description(
                self.cost_result_current, self.desired_cost
            )
            if self.config.repair_with_plot:
                description += "To give you a broad understanding of the scenario which is currently used for motion planning, a plot is provided showing all lanes (grey), the planned trajectory (black line) and the goal area (light orange)"
        else:
            description = "Usually here would be an evaluation of the initial motion planning result, but..."
            description += self.prompter.generate_exception_description(
                planned_trajectory
            )

        self.prompter.user_prompt.set("feedback", description)

    def add_feedback(self, cost_result: PlanningProblemCostResult):
        """
        Evaluates the result of the repair process
        """
        # retrieve current cost result
        self.cost_result_current = cost_result
        if self.config.feedback_mode > 0:
            version = "last"
        else:
            version = "initial"
        feedback = self.prompter.update_cost_description(
            self.cost_result_previous,
            self.cost_result_current,
            self.desired_cost,
            a_version=version,
        )
        if self.config.feedback_mode > 2:
            feedback += "Based on all provided information about the last and current repair, try to identify which weights are responsible for a better performance!"
        print(f"*\t Feedback: {feedback}")
        # update the current cost
        self.current_cost = self.cost_result_current.total_costs
        return feedback

    def generate_emergency_prescription(self) -> str:
        pass

    def diagnose_repair(self):
        """
        Full DrPlanner session:
        It first describes the current state of the patient.
        After that it runs a prompts repairing cycle:
        Plan. Repair. Evaluate.
        until the patient is cured, or the doctor runs out of tokens/time
        """
        nr_iteration = 0
        run_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(
            f"[DrPlanner] Starts the diagnosis and repair process at {run_start_time}."
        )
        result = None
        emergency_flag = False
        update_memory = False
        new_emergency_prescriptions: list = []
        last_exception_description = ""

        # test the initial motion planner once
        planning_result = self.plan(nr_iteration)
        if isinstance(planning_result, Exception):
            self.cost_result_current = get_infinite_cost_result(self.cost_type)
            self.cost_result_previous = self.cost_result_current
            self.describe_planner(update=True, improved=True)
            self.describe_trajectory(planning_result)
            self.current_cost = np.inf
        else:
            self.cost_result_current = planning_result
            self.cost_result_previous = self.cost_result_current
            self.describe_planner(update=True, improved=True)
            self.describe_trajectory(None)
            self.current_cost = self.cost_result_current.total_costs

        self.initial_cost = self.current_cost
        self.lowest_cost = self.current_cost
        # start the repairing process
        while (
            abs(self.current_cost - self.desired_cost) > self.THRESHOLD
            and self.token_count < self.TOKEN_LIMIT
            and nr_iteration < self.ITERATION_MAX
        ):
            print(f"*\t -----------iteration {nr_iteration}-----------")
            print(
                f"*\t <{nr_iteration}>: total cost {self.current_cost} (desired: {self.desired_cost})\n"
                f"*\t used tokens {self.token_count} (limit: {self.TOKEN_LIMIT})"
            )

            # prepare the API request
            path_to_plots = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            if self.config.repair_with_plot:
                path_to_plots += "/plots/img.png"
            else:
                path_to_plots = None

            messages = LLM.get_messages(
                self.prompter.system_prompt.__str__(),
                self.prompter.user_prompt.__str__(),
                path_to_plots,
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
                path_to_plot=path_to_plots,
                mockup_path=mock_up_path,
            )
            self.diagnosis_result = result
            # reset some variables
            self.prompter.reload_LLM()
            nr_iteration += 1

            # repair the motion planner and test the result
            try:
                self.repair()
                cost_result = self.plan(nr_iteration)
                if isinstance(cost_result, Exception):
                    raise cost_result
                # add feedback
                prompt_feedback = self.add_feedback(cost_result)
                # determine whether the current cost function prompt needs an update
                improved = self.current_cost <= self.lowest_cost
                update = self.config.feedback_mode % 2 == 1 or (
                    improved and self.config.feedback_mode == 2
                )
                if improved:
                    self.lowest_cost = self.current_cost
                if update:
                    self.cost_result_previous = self.cost_result_current

                self.describe_planner(update=update, improved=improved)
                if update_memory is not None and update_memory and emergency_flag:
                    k, _, _ = new_emergency_prescriptions[-1]
                    new_emergency_prescriptions.append((k, self.generate_emergency_prescription(), path_to_plots))
                    update_memory = None
                emergency_flag = False

            except Exception as e:
                prompt_feedback = (
                    "Usually here would be an evaluation of the repair, but..."
                )
                exception_description = self.prompter.generate_exception_description(e)
                prompt_feedback += f"{exception_description}\n"
                emergency_prescription = self.memory.retrieve(exception_description, image_file_path=path_to_plots)
                prompt_feedback += f"Here is some advice on how to deal with this exception:\n{emergency_prescription}"

                self.cost_result_current = get_infinite_cost_result(self.cost_type)
                self.current_cost = np.inf
                update = self.config.feedback_mode == 1
                self.describe_planner(update=update)
                if update_memory is not None and not update_memory and emergency_flag and exception_description == last_exception_description:
                    update_memory = True
                    new_emergency_prescriptions.append((exception_description, None, None))
                emergency_flag = True
                last_exception_description = exception_description

            self.prompter.user_prompt.set("feedback", prompt_feedback)
            self.cost_list.append(self.current_cost)

        if update_memory and new_emergency_prescriptions:
            for k, v, p in new_emergency_prescriptions:
                self.memory.insert(k, v, image_file_path=p)

        print("[DrPlanner] Ends.")
        return result
