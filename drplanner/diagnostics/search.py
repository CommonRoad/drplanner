import math
from typing import Union
import importlib
import traceback
import copy
from types import MethodType
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.trajectory import Trajectory
from commonroad.common.solution import (
    CommonRoadSolutionWriter,
    CostFunction,
    VehicleType,
    Solution,
    PlanningProblemSolution,
    VehicleModel,
)

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

from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.utility import create_trajectory_from_list_states
from SMP.motion_planner.utility import visualize_solution
import SMP.batch_processing.helper_functions as hf
from SMP.motion_planner.queue import PriorityQueue
from SMP.motion_planner.utility import plot_primitives
from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch

from commonroad_dc.costs.evaluation import (
    CostFunctionEvaluator,
    PlanningProblemCostResult,
)

from drplanner.utils.gpt import num_tokens_from_messages
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.diagnostics.base import DrPlannerBase
from drplanner.prompter.prompter import Prompter

import numpy as np


class DrSearchPlanner(DrPlannerBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem_set: PlanningProblemSet,
        config: DrPlannerConfiguration,
        motion_primitives_id: str,
        planner_id: str,
    ):
        super().__init__(scenario, planning_problem_set, config)

        # initialize the motion primitives
        self.motion_primitives_id = motion_primitives_id

        # initialize the motion planner
        self.planner_id = planner_id
        # import the planner
        planner_name = f"drplanner.planners.student_{self.planner_id}"
        planner_module = importlib.import_module(planner_name)
        automaton = ManeuverAutomaton.generate_automaton(motion_primitives_id)
        # use StudentMotionPlanner from the dynamically imported module
        self.StudentMotionPlanner = getattr(planner_module, "StudentMotionPlanner")
        self.motion_planner = self.StudentMotionPlanner(
            self.scenario, self.planning_problem, automaton, DefaultPlotConfig
        )

        # initialize the vehicle parameters and the cost function
        self.cost_type = CostFunction.SM1
        self.vehicle_type = VehicleType.BMW_320i
        self.vehicle_model = VehicleModel.KS
        self.cost_evaluator = CostFunctionEvaluator(
            self.cost_type, VehicleType.BMW_320i
        )

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

    def repair(self, diagnosis_result: Union[str, None]):
        # ----- heuristic function -----
        updated_heuristic_function = diagnosis_result[
            self.prompter.LLM.HEURISTIC_FUNCTION
        ]
        # Create a namespace dictionary to hold the compiled function
        function_namespace = {}
        function_namespace.update(self.motion_planner.__dict__)
        function_namespace["np"] = np
        function_namespace["PriorityNode"] = PriorityNode
        function_namespace["DefaultPlotConfig"] = DefaultPlotConfig
        function_namespace["GreedyBestFirstSearch"] = GreedyBestFirstSearch

        # Execute the updated heuristic function string
        try:
            exec(updated_heuristic_function, globals(), function_namespace)
        except Exception as e:
            # Handle exceptions (e.g., compilation errors)
            raise RuntimeError(f"Error compiling heuristic function: {e}")

        # Extract the new function
        new_heuristic = function_namespace["heuristic_function"]
        if not callable(new_heuristic):
            raise ValueError("No valid 'heuristic_function' found after execution")

        # Bind the function to the StudentMotionPlanner instance
        self.motion_planner.heuristic_function = new_heuristic.__get__(
            self.motion_planner
        )

        # ----- motion primitives -----
        updated_motion_primitives_id = diagnosis_result[
            self.prompter.LLM.MOTION_PRIMITIVES
        ]
        if not updated_motion_primitives_id.endswith(".xml"):
            updated_motion_primitives_id += ".xml"
        if updated_motion_primitives_id != self.motion_primitives_id:
            print(f"*\t New primitives {updated_motion_primitives_id} are loaded")
            updated_automaton = ManeuverAutomaton.generate_automaton(
                updated_motion_primitives_id
            )

            if self._visualize:
                plot_primitives(updated_automaton.list_primitives)
        else:
            print("*\t Same primitives are used")
            updated_automaton = self.motion_planner.automaton

        planning_problem = copy.deepcopy(
            list(self.planning_problem_set.planning_problem_dict.values())[0]
        )
        self.motion_planner = self.StudentMotionPlanner(
            self.scenario, planning_problem, updated_automaton, DefaultPlotConfig
        )
        self.motion_planner.heuristic_function = MethodType(
            new_heuristic, self.motion_planner
        )
        self.motion_planner.frontier = PriorityQueue()

    def describe(
        self, planned_trajectory: Union[Trajectory, None]
    ) -> (str, PlanningProblemCostResult):
        template = self.prompter.astar_template

        planner_description = self.prompter.generate_planner_description(
            self.StudentMotionPlanner, self.motion_primitives_id
        )
        template = template.replace("[PLANNER]", planner_description)

        if planned_trajectory:
            evaluation_trajectory = self.evaluate_trajectory(planned_trajectory)

            traj_description = self.prompter.generate_cost_description(
                evaluation_trajectory, self.desired_cost
            )
        else:
            traj_description = "*\t no trajectory is generated"
            evaluation_trajectory = None
        template = template.replace("[PLANNED_TRAJECTORY]", traj_description)
        return template, evaluation_trajectory

    def add_feedback(self, updated_trajectory: Trajectory, iteration: int):
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

    def plan(self, nr_iter: int) -> Trajectory:
        list_paths_primitives, _, _ = self.motion_planner.execute_search()
        trajectory_solution = create_trajectory_from_list_states(
            list_paths_primitives, self.motion_planner.rear_ax_dist
        )
        kwarg = {
            "planning_problem_id": self.planning_problem.planning_problem_id,
            "vehicle_model": self.vehicle_model,
            "vehicle_type": self.vehicle_type,
            "cost_function": self.cost_type,
            "trajectory": trajectory_solution,
        }

        planning_problem_solution = PlanningProblemSolution(**kwarg)
        if self._visualize:
            visualize_solution(
                self.scenario, self.planning_problem_set, trajectory_solution
            )
            target_folder = self.dir_output + "search/"
            os.makedirs(
                os.path.dirname(target_folder), exist_ok=True
            )  # Ensure the directory exists
            # create PlanningProblemSolution object
            hf.save_gif2(
                self.scenario,
                self.planning_problem_set.find_planning_problem_by_id(
                    self.planning_problem.planning_problem_id
                ),
                planning_problem_solution.trajectory,
                output_path=target_folder,
            )
        if self._save_solution:
            # create solution object
            kwarg = {
                "scenario_id": self.scenario.scenario_id,
                "planning_problem_solutions": [planning_problem_solution],
            }

            solution = Solution(**kwarg)
            # write solution to a CommonRoad XML file
            csw = CommonRoadSolutionWriter(solution)
            target_folder = self.dir_output + "search/solutions/"
            os.makedirs(
                os.path.dirname(target_folder), exist_ok=True
            )  # Ensure the directory exists
            csw.write_to_file(
                output_path=target_folder,
                filename=f"solution_{solution.benchmark_id}_iter_{nr_iter}.xml",
                overwrite=True,
            )
        return trajectory_solution

    def evaluate_trajectory(self, trajectory: Trajectory) -> PlanningProblemCostResult:
        return self.cost_evaluator.evaluate_pp_solution(
            self.scenario, self.planning_problem, trajectory
        )
