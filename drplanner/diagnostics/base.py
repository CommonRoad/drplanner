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
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.evaluation import (
    PlanningProblemCostResult,
    CostFunctionEvaluator,
)

from describer.trajectory_description import get_infinite_cost_result
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.prompter.search import PrompterSearch
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

        # initialize prompter
        self.prompter = PrompterSearch(
            self.scenario,
            self.planning_problem,
            self.config.openai_api_key,
            self.config.gpt_version,
            mockup=self.config.mockup_openAI,
        )
        self.prompter.LLM.temperature = self.config.temperature

        # initialize meta parameters
        self.cost_type = CostFunction.SM1
        self.vehicle_model = VehicleModel.KS
        self.vehicle_type = VehicleType.BMW_320i
        self.cost_evaluator = CostFunctionEvaluator(
            self.cost_type, VehicleType.BMW_320i
        )
        self.evaluation_trajectory = None

    @abstractmethod
    def repair(self, diagnosis_result: Union[str, None]):
        """
        Tries to implement the recommendations by DrPlanner
        """
        pass

    @abstractmethod
    def describe_planner(
        self,
        diagnosis_result: Union[str, None],
    ) -> str:
        """
        Describes the current state of the planner to DrPlanner
        """
        pass

    @abstractmethod
    def plan(self, nr_iter: int) -> Trajectory:
        """
        Wrapper method to run the motion planner
        """
        pass

    # def describe(
    #        self,
    #        planned_trajectory: Union[Trajectory, Exception],
    #        diagnosis_result: Union[str, None],
    # ) -> str:
    #    template = self.prompter.algorithm_template
    #    prompt_planner = self.describe_planner(diagnosis_result)
    #    prompt_trajectory, evaluation_trajectory = self.describe_trajectory(
    #        planned_trajectory
    #    )
    #    template = template.replace("[PLANNER]", prompt_planner)
    #    template = template.replace("[PLANNED_TRAJECTORY]", prompt_trajectory)
    #    self.evaluation_trajectory = evaluation_trajectory
    #    return template

    def instantiate_template(
        self, prompt_planner: str, prompt_trajectory: str, prompt_feedback: str
    ):
        template = self.prompter.algorithm_template
        template = template.replace("[PLANNER]", prompt_planner)
        template = template.replace("[PLANNED_TRAJECTORY]", prompt_trajectory)
        return template.replace("[FEEDBACK]", prompt_feedback)

    def describe_trajectory(
        self, planned_trajectory: Union[Trajectory, Exception]
    ) -> (str, PlanningProblemCostResult):
        if isinstance(planned_trajectory, Trajectory):
            evaluation_trajectory = self.cost_evaluator.evaluate_pp_solution(
                self.scenario, self.planning_problem, planned_trajectory
            )
            description = (
                f"The objective is to adjust the total cost of the planned trajectory to closely "
                f"align with the desired value {self.desired_cost}. "
            )
            description += self.prompter.generate_cost_description(
                evaluation_trajectory
            )
            return description, evaluation_trajectory
        else:
            evaluation_trajectory = get_infinite_cost_result(self.cost_type)
            self.prompter.generate_cost_description(evaluation_trajectory)
            description = "Usually here would be an evaluation of the motion planning result, but..."
            description += self.prompter.generate_exception_description(
                planned_trajectory
            )
            return description, evaluation_trajectory

    def add_feedback(self, updated_trajectory: Trajectory):
        """
        Evaluates the result of the repair process
        """
        # set last cost result
        self.prompter.update_cost_description(self.evaluation_trajectory)
        # retrieve current cost result
        self.evaluation_trajectory = self.cost_evaluator.evaluate_pp_solution(
            self.scenario, self.planning_problem, updated_trajectory
        )
        feedback = self.prompter.generate_cost_description(self.evaluation_trajectory)
        print(f"*\t Feedback: {feedback}")
        # update the current cost
        self.current_cost = self.evaluation_trajectory.total_costs
        return feedback

    def diagnose_repair(self):
        """
        Full DrPlanner session:
        It first describes the current state of the patient.
        After that it runs an iterative repairing cycle:
        Plan. Repair. Evaluate.
        until the patient is cured, or the doctor runs out of tokens/time
        """
        nr_iteration = 0
        run_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(
            f"[DrPlanner] Starts the diagnosis and repair process at {run_start_time}."
        )
        result = None

        # test the initial motion planner once
        try:
            planned_trajectory = self.plan(nr_iteration)
            prompt_planner = self.describe_planner(None)
            prompt_trajectory, evaluation_trajectory = self.describe_trajectory(
                planned_trajectory
            )
            self.evaluation_trajectory = evaluation_trajectory
            self.current_cost = self.evaluation_trajectory.total_costs
        except Exception as e:
            prompt_planner = self.describe_planner(None)
            prompt_trajectory, evaluation_trajectory = self.describe_trajectory(e)
            self.evaluation_trajectory = evaluation_trajectory
            self.current_cost = np.inf

        self.initial_cost = self.current_cost
        # history = self.prompter.generate_cost_description(self.evaluation_trajectory)
        prompt_feedback = ""
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
            message = [
                {"role": "system", "content": self.prompter.prompt_system},
                {
                    "role": "user",
                    "content": self.instantiate_template(
                        prompt_planner, prompt_trajectory, prompt_feedback
                    ),
                },
            ]
            # todo: get the token from the openai interface
            self.token_count += num_tokens_from_messages(
                message,
                self.prompter.LLM.gpt_version,
            )
            mockup_nr_iteration = -1
            if self.config.mockup_openAI:
                mockup_nr_iteration = nr_iteration

            # send request and receive response
            result = self.prompter.LLM.query(
                message,
                scenario_id=str(self.scenario.scenario_id),
                # planner_id=str(self.planner_id),
                # start_time=run_start_time,
                save_dir=self.config.save_dir,
                nr_iter=nr_iteration,
                mockup_nr_iter=mockup_nr_iteration,
            )

            # reset some variables
            self.prompter.reload_LLM()
            prompt_feedback = "Diagnoses and prescriptions from the last iteration:\n"
            nr_iteration += 1

            # repair the motion planner and test the result
            try:
                prompt_feedback += f" {result['summary']}\n"
                self.repair(result)
                planned_trajectory = self.plan(nr_iteration)
                # add feedback
                previous_cost = self.current_cost
                prompt_feedback += self.add_feedback(planned_trajectory)
                # if this solution was better than before, set it as new standard
                if self.current_cost <= previous_cost and self.config.version2:
                    prompt_planner = self.describe_planner(result)
                    prompt_trajectory = ""

            except Exception as e:
                prompt_feedback += (
                    "Usually here would be an evaluation of the repair, but..."
                )
                prompt_feedback += self.prompter.generate_exception_description(e)
                self.current_cost = np.inf
            self.cost_list.append(self.current_cost)

        print("[DrPlanner] Ends.")
        return result
