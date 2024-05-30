import math
import copy
import os
import traceback
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

    def describe(
        self,
        planned_trajectory: Union[Trajectory, Exception],
        diagnosis_result: Union[str, None],
    ) -> (str, PlanningProblemCostResult):
        template = self.prompter.algorithm_template
        prompt_planner = self.describe_planner(diagnosis_result)
        prompt_trajectory, evaluation_trajectory = self.describe_trajectory(
            planned_trajectory
        )
        template = template.replace("[PLANNER]", prompt_planner)
        template = template.replace("[PLANNED_TRAJECTORY]", prompt_trajectory)
        return template, evaluation_trajectory

    def describe_trajectory(
        self, planned_trajectory: Union[Trajectory, Exception]
    ) -> (str, PlanningProblemCostResult):
        if isinstance(planned_trajectory, Trajectory):
            evaluation_trajectory = self.cost_evaluator.evaluate_pp_solution(
                self.scenario, self.planning_problem, planned_trajectory
            )

            description = self.prompter.generate_cost_description(
                evaluation_trajectory, self.desired_cost
            )
            return description, evaluation_trajectory
        else:
            description = "Usually here would be an evaluation of the motion planning result, but..."
            description += self.prompter.generate_exception_description(planned_trajectory)
            return description, None

    def evaluate_trajectory(self, trajectory: Trajectory) -> PlanningProblemCostResult:
        return self.cost_evaluator.evaluate_pp_solution(
            self.scenario, self.planning_problem, trajectory
        )

    def add_feedback(self, updated_trajectory: Trajectory, iteration: int):
        """
        Evaluates the result of the repair process
        """
        feedback = "After applying this diagnostic result,"
        evaluation_trajectory = self.evaluate_trajectory(updated_trajectory)
        feedback += self.prompter.update_cost_description(evaluation_trajectory)
        if evaluation_trajectory.total_costs > self.current_cost:
            feedback += (
                f" the performance of the motion planner ({evaluation_trajectory.total_costs})"
                f" is getting worse than iteration {iteration - 1} ({self.current_cost}). "
                f"Please continue output the improved heuristic function and motion primitives."
            )
        else:
            feedback += (
                f" the performance of the motion planner ({evaluation_trajectory.total_costs})"
                f" is getting better than iteration {iteration - 1} ({self.current_cost})."
                " Please continue output the improved heuristic function and motion primitives."
            )
        print(f"*\t Feedback: {feedback}")
        # update the current cost
        self.current_cost = evaluation_trajectory.total_costs
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
        history = ""
        result = None

        # test the initial motion planner once
        try:
            planned_trajectory = self.plan(nr_iteration)
            prompt_planner, evaluation_trajectory = self.describe(
                planned_trajectory, None
            )
            self.current_cost = evaluation_trajectory.total_costs
        except Exception as e:
            prompt_planner, _ = self.describe(e, None)
            self.current_cost = np.inf
        self.initial_cost = self.current_cost

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
                {"role": "user", "content": prompt_planner + history},
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
                str(self.scenario.scenario_id),
                str(self.planner_id),
                message,
                run_start_time,
                nr_iter=nr_iteration,
                save_dir=self.config.save_dir,
                # save_dir=self.dir_output + "prompts/",
                mockup_nr_iter=mockup_nr_iteration,
            )

            # reset some variables
            self.prompter.reload_LLM()
            history = "Diagnoses and prescriptions from the last iteration:"
            nr_iteration += 1

            # repair the motion planner and test the result
            try:
                history += f" {result['summary']}"
                self.repair(result)
                planned_trajectory = self.plan(nr_iteration)
                # add feedback
                history += self.add_feedback(planned_trajectory, nr_iteration) + "\n"
            except Exception as e:
                history += "Usually here would be an evaluation of the repair, but..."
                history += self.prompter.generate_exception_description(e)
                self.current_cost = np.inf
            self.cost_list.append(self.current_cost)

        print("[DrPlanner] Ends.")
        return result


#    def diagnose_repair_version2(self):
#        """
#        Full DrPlanner session:
#        It first describes the current state of the patient.
#        After that it runs an iterative repairing cycle:
#        Plan. Describe. Repair. Evaluate.
#        until the patient is cured, or the doctor runs out of tokens/time
#        """
#        nr_iteration = 0
#        run_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
#        save_dir = "/home/sebastian/Documents/Uni/Bachelorarbeit/DrPlanner_Data/"
#        self.current_cost = math.inf
#        repair_error_message = None
#        result = None  #
#        print(
#            f"[DrPlanner] Starts the diagnosis and repair process at {run_start_time}."
#        )  #
#        while (
#            abs(self.current_cost - self.desired_cost) > self.THRESHOLD
#            and self.token_count < self.TOKEN_LIMIT
#            and nr_iteration < self.ITERATION_MAX
#        ):
#            # --- log current session data ---
#            print(f"*\t -----------iteration {nr_iteration}-----------")
#            print(
#                f"*\t <{nr_iteration}>: total cost {self.current_cost} (desired: {self.desired_cost})\n"
#                f"*\t used tokens {self.token_count} (limit: {self.TOKEN_LIMIT})"
#            )  #
#            # --- add a short summary of the last session ---
#            prompt_evaluation = ""
#            prompt_evaluation += (
#                f"*\t Diagnoses and prescriptions from the previous iteration:\n"
#            )
#            if result is None:
#                prompt_evaluation += "There was no previous iteration...\n"
#            else:
#                prompt_evaluation += f" {result['summary']}\n"  #
#            # --- try to run the planner with the current cost function and describe the result ---
#            prompt_evaluation += "Currently this happens if you run the planner: "
#            try:
#                planned_trajectory = self.plan(nr_iteration)
#                prompt_system, evaluation_trajectory = self.describe(
#                    planned_trajectory, result
#                )
#                self.current_cost = evaluation_trajectory.total_costs
#                # add feedback
#                prompt_evaluation += (
#                    self.add_feedback(planned_trajectory, nr_iteration) + "\n"
#                )
#            except Exception as e:
#                prompt_system, _ = self.describe(None, result)
#                self.current_cost = np.inf
#                # This gets the traceback as a string
#                error_traceback = traceback.format_exc()
#                print("*\t !! Errors during planning: ", error_traceback)  #
#                if repair_error_message is None:
#                    prompt_evaluation += f"The cost function compiles but throws this exception when used in the planning process: {repair_error_message}"
#                else:
#                    prompt_evaluation += f"Unfortunately the repaired cost function did not compile: {error_traceback}"  #
#            # --- create the message for the LLM and count its tokens ---
#            message = [
#                {"role": "system", "content": self.prompter.prompt_system},
#                {"role": "user", "content": prompt_system + prompt_evaluation},
#            ]
#            self.token_count += num_tokens_from_messages(
#                message,
#                self.prompter.LLM.gpt_version,
#            )  #
#            # --- in case the LLM should not actually be contacted for debugging purposes ---
#            mockup_nr_iteration = -1
#            if self.config.mockup_openAI:
#                mockup_nr_iteration = nr_iteration  #
#            # --- run a LLM query with all gathered information (and save its message, results) ---
#            result = self.prompter.LLM.query(
#                str(self.scenario.scenario_id),
#                str(self.planner_id),
#                message,
#                run_start_time,
#                nr_iter=nr_iteration,
#                save_dir=save_dir,
#                mockup_nr_iter=mockup_nr_iteration,
#            )
#            # todo: why is this needed?
#            self.prompter.reload_LLM()
#            nr_iteration += 1  #
#            # --- try to exec the repaired cost function ---
#            try:
#                self.repair(result)
#                repair_error_message = None
#            except Exception as e:
#                error_traceback = (
#                    traceback.format_exc()
#                )  # This gets the traceback as a string
#                print("*\t !! Errors while repairing: ", error_traceback)
#                repair_error_message = error_traceback  #
#            self.cost_list.append(self.current_cost)
#        print("[DrPlanner] Ends.")
#        return result  #
