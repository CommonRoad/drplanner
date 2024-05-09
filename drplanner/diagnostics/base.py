import math
import copy
import os
import traceback
from typing import Union

import numpy as np
from abc import ABC, abstractmethod

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.evaluation import PlanningProblemCostResult

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.prompter.prompter import Prompter
from drplanner.utils.gpt import num_tokens_from_messages


class DrPlannerBase(ABC):
    def __init__(
            self,
            scenario: Scenario,
            planning_problem_set: PlanningProblemSet,
            config: DrPlannerConfiguration,
            planner_id: str,
    ):

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

        self.token_count = 0
        self.cost_list = []

        self.dir_output = os.path.join(os.path.dirname(__file__), "../../outputs/")
        os.makedirs(
            os.path.dirname(self.dir_output), exist_ok=True
        )  # Ensure the directory exists

        self.prompter = Prompter(
            self.scenario,
            self.planning_problem,
            self.config.openai_api_key,
            self.config.gpt_version,
        )
        self.prompter.LLM.temperature = self.config.temperature

    @abstractmethod
    def repair(self, diagnosis_result: Union[str, None]):
        pass

    @abstractmethod
    def describe(
            self, planned_trajectory: Union[Trajectory, None]
    ) -> (str, PlanningProblemCostResult):
        pass

    @abstractmethod
    def add_feedback(self, updated_trajectory: Trajectory, iteration: int):
        pass

    @abstractmethod
    def plan(self, nr_iter: int) -> Trajectory:
        pass

    @abstractmethod
    def evaluate_trajectory(self, trajectory: Trajectory) -> PlanningProblemCostResult:
        pass

    def diagnose_repair(self):
        nr_iteration = 0
        print("[DrPlanner] Starts the diagnosis and repair process.")
        try:
            planned_trajectory = self.plan(nr_iteration)
            prompt_planner, evaluation_trajectory = self.describe(planned_trajectory)
            self.current_cost = evaluation_trajectory.total_costs
        except:
            prompt_planner, _ = self.describe(None)
            self.current_cost = np.inf
        result = None
        self.initial_cost = self.current_cost
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
            message = [
                {"role": "system", "content": self.prompter.prompt_system},
                {"role": "user", "content": prompt_planner},
            ]
            # count the used token
            # todo: get the token from the openai interface
            self.token_count += num_tokens_from_messages(
                message, self.prompter.LLM.gpt_version
            )
            result = self.prompter.LLM.query(
                str(self.scenario.scenario_id),
                str(self.planner_id),
                message,
                nr_iter=nr_iteration,
                save_dir=self.dir_output + "prompts/",
                # mockup=nr_iteration
            )
            self.prompter.reload_LLM()
            # add nr of iteration
            nr_iteration += 1
            prompt_planner += (
                f"*\t Diagnoses and prescriptions from the iteration {nr_iteration}:"
            )
            try:
                prompt_planner += f" {result['summary']}"
                self.repair(result)
                planned_trajectory = self.plan(nr_iteration)
                # add feedback
                prompt_planner += (
                        self.add_feedback(planned_trajectory, nr_iteration) + "\n"
                )
            except Exception as e:
                error_traceback = (
                    traceback.format_exc()
                )  # This gets the traceback as a string
                print("*\t !! Errors: ", error_traceback)
                # Catching the exception and extracting error information
                prompt_planner += (
                    f" But they cause the error message: {error_traceback}"
                )
                self.current_cost = np.inf
            self.cost_list.append(self.current_cost)
        print("[DrPlanner] Ends.")
        return result
