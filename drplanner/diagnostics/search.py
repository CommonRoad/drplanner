import textwrap
import importlib
import copy
import sys
import os
from types import MethodType
from typing import Union

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.solution import (
    CommonRoadSolutionWriter,
    Solution,
    PlanningProblemSolution,
)
from commonroad_dc.costs.evaluation import PlanningProblemCostResult

# make sure the SMP has been installed successfully
try:
    import SMP

    print("[DrPlanner] Installed SMP module is called.")
except ImportError as e:

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    smp_path = os.path.join(current_file_dir, "../../planners/commonroad-search/")
    sys.path.append(smp_path)
    print(f"[DrPlanner] Use the SMP under {smp_path}.")

from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.utility import create_trajectory_from_list_states
from SMP.motion_planner.utility import visualize_solution
import SMP.batch_processing.helper_functions as hf
from SMP.motion_planner.queue import PriorityQueue
from SMP.motion_planner.utility import plot_primitives
from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.diagnostics.base import DrPlannerBase

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

        # import the planner
        planner_name = f"drplanner.planners.student_{planner_id}"
        planner_module = importlib.import_module(planner_name)
        automaton = ManeuverAutomaton.generate_automaton(motion_primitives_id)
        # use StudentMotionPlanner from the dynamically imported module
        self.StudentMotionPlanner = getattr(planner_module, "StudentMotionPlanner")
        self.motion_planner = self.StudentMotionPlanner(
            self.scenario, self.planning_problem, automaton, DefaultPlotConfig
        )

    def repair(self):
        # ----- heuristic function -----
        updated_heuristic_function = self.diagnosis_result[
            self.prompter.HEURISTIC_FUNCTION
        ]
        updated_heuristic_function = textwrap.dedent(updated_heuristic_function)
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
        updated_motion_primitives_id = self.diagnosis_result[
            self.prompter.MOTION_PRIMITIVES
        ]
        if updated_motion_primitives_id.startswith(
            "'"
        ) and updated_motion_primitives_id.endswith("'"):
            updated_motion_primitives_id = updated_motion_primitives_id[1:-1]
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

    def describe_planner(
        self,
        update: bool = False,
        improved: bool = False,
    ):
        self.prompter.update_planner_prompt(
            self.StudentMotionPlanner, self.motion_primitives_id
        )

    def plan(self, nr_iter: int) -> Union[PlanningProblemCostResult, Exception]:
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
        return self.cost_evaluator.evaluate_pp_solution(
            self.scenario, self.planning_problem, trajectory_solution
        )
