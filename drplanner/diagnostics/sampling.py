import importlib
import math
import os
import textwrap
import traceback
from datetime import datetime
from types import MethodType
from typing import Union, Optional, Tuple

import numpy as np
from commonroad.common.solution import CommonRoadSolutionWriter
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.evaluation import PlanningProblemCostResult
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_rp.cost_function import CostFunction
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.utility.config import ReactivePlannerConfiguration
from commonroad_rp.utility.evaluation import run_evaluation

from drplanner.diagnostics.base import DrPlannerBase
from drplanner.prompter.sampling import PrompterSampling
from drplanner.utils.config import DrPlannerConfiguration
from utils.gpt import num_tokens_from_messages


def get_planner(filename) -> Tuple[ReactivePlannerConfiguration, ReactivePlanner]:
    # Build config object
    config = ReactivePlannerConfiguration.load(
        f"drplanner/planners/standard-config.yaml", filename
    )
    config.update()
    # run route planner and add reference path to config
    route_planner = RoutePlanner(config.scenario, config.planning_problem)
    route = route_planner.plan_routes().retrieve_first_route()

    # initialize reactive planner
    planner = ReactivePlanner(config)

    # set reference path for curvilinear coordinate system
    planner.set_reference_path(route.reference_path)
    return config, planner


def run_planner(
    planner: ReactivePlanner,
    config: ReactivePlannerConfiguration,
    cost_function: CostFunction,
):
    # update cost function
    planner.set_cost_function(cost_function)

    # Get the source code of the function
    # source_code = inspect.getsource(planner.cost_function.evaluate)
    # Add first state to recorded state and input list
    planner.record_state_and_input(planner.x_0)

    while not planner.goal_reached():
        current_count = len(planner.record_state_list) - 1

        # check if planning cycle or not
        plan_new_trajectory = current_count % config.planning.replanning_frequency == 0
        if plan_new_trajectory:
            # new planning cycle -> plan a new optimal trajectory
            planner.set_desired_velocity(current_speed=planner.x_0.velocity)
            optimal = planner.plan()
            if not optimal:
                break

            planner.record_state_and_input(optimal[0].state_list[1])
            planner.reset(
                initial_state_cart=planner.record_state_list[-1],
                initial_state_curv=(optimal[2][1], optimal[3][1]),
                collision_checker=planner.collision_checker,
                coordinate_system=planner.coordinate_system,
            )
        else:
            # continue on optimal trajectory
            temp = current_count % config.planning.replanning_frequency

            planner.record_state_and_input(optimal[0].state_list[1 + temp])
            planner.reset(
                initial_state_cart=planner.record_state_list[-1],
                initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
                collision_checker=planner.collision_checker,
                coordinate_system=planner.coordinate_system,
            )
    solution, _ = run_evaluation(
        planner.config, planner.record_state_list, planner.record_input_list
    )
    return solution


class DrSamplingPlanner(DrPlannerBase):
    def __init__(
        self,
        scenario: Scenario,
        scenario_path: str,
        planning_problem_set: PlanningProblemSet,
        config: DrPlannerConfiguration,
        cost_function_id: str,
    ):
        super().__init__(scenario, planning_problem_set, config, cost_function_id)
        print(scenario_path)
        # initialize the motion planner
        self.motion_planner_config, self.motion_planner = get_planner(scenario_path)

        # initialize prompter
        self.prompter = PrompterSampling(
            self.scenario,
            self.planning_problem,
            self.config.openai_api_key,
            self.config.mockup_openAI,
            self.config.gpt_version,
        )
        self.prompter.LLM.temperature = self.config.temperature

        # import the cost function
        cost_function_name = f"drplanner.planners.student_{cost_function_id}"
        cost_function_module = importlib.import_module(cost_function_name)
        self.DefaultCostFunction = getattr(cost_function_module, "DefaultCostFunction")
        self.cost_function = self.DefaultCostFunction(
            self.motion_planner.x_0.velocity, desired_d=0.0, desired_s=None
        )

    def diagnose_repair_version2(self):
        """
        Full DrPlanner session:
        It first describes the current state of the patient.
        After that it runs an iterative repairing cycle:
        Plan. Describe. Repair. Evaluate.
        until the patient is cured, or the doctor runs out of tokens/time
        """
        nr_iteration = 0
        run_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = "/home/sebastian/Documents/Uni/Bachelorarbeit/DrPlanner_Data/"
        self.current_cost = math.inf
        repair_error_message = None
        result = None

        print(
            f"[DrPlanner] Starts the diagnosis and repair process at {run_start_time}."
        )

        while (
            abs(self.current_cost - self.desired_cost) > self.THRESHOLD
            and self.token_count < self.TOKEN_LIMIT
            and nr_iteration < self.ITERATION_MAX
        ):
            # --- log current session data ---
            print(f"*\t -----------iteration {nr_iteration}-----------")
            print(
                f"*\t <{nr_iteration}>: total cost {self.current_cost} (desired: {self.desired_cost})\n"
                f"*\t used tokens {self.token_count} (limit: {self.TOKEN_LIMIT})"
            )

            # --- add a short summary of the last session ---
            prompt_evaluation = ""
            prompt_evaluation += (
                f"*\t Diagnoses and prescriptions from the previous iteration:\n"
            )
            if result is None:
                prompt_evaluation += "There was no previous iteration...\n"
            else:
                prompt_evaluation += f" {result['summary']}\n"

            # --- try to run the planner with the current cost function and describe the result ---
            prompt_evaluation += "Currently this happens if you run the planner: "
            try:
                planned_trajectory = self.plan(nr_iteration)
                prompt_system, evaluation_trajectory = self.describe(
                    planned_trajectory, result
                )
                self.current_cost = evaluation_trajectory.total_costs
                # add feedback
                prompt_evaluation += (
                    self.add_feedback(planned_trajectory, nr_iteration) + "\n"
                )
            except Exception as e:
                prompt_system, _ = self.describe(None, result)
                self.current_cost = np.inf
                # This gets the traceback as a string
                error_traceback = traceback.format_exc()
                print("*\t !! Errors during planning: ", error_traceback)

                if repair_error_message is None:
                    prompt_evaluation += f"The cost function compiles but throws this exception when used in the planning process: {repair_error_message}"
                else:
                    prompt_evaluation += f"Unfortunately the repaired cost function did not compile: {error_traceback}"

            # --- create the message for the LLM and count its tokens ---
            message = [
                {"role": "system", "content": self.prompter.prompt_system},
                {"role": "user", "content": prompt_system + prompt_evaluation},
            ]
            self.token_count += num_tokens_from_messages(
                message,
                self.prompter.LLM.gpt_version,
                mockup=self.config.mockup_tiktoken,
            )

            # --- in case the LLM should not actually be contacted for debugging purposes ---
            mockup_nr_iteration = -1
            if self.config.mockup_openAI:
                mockup_nr_iteration = nr_iteration

            # --- run a LLM query with all gathered information (and save its message, results) ---
            result = self.prompter.LLM.query(
                str(self.scenario.scenario_id),
                str(self.planner_id),
                message,
                run_start_time,
                nr_iter=nr_iteration,
                save_dir=save_dir,
                mockup_nr_iter=mockup_nr_iteration,
            )
            # todo: why is this needed?
            self.prompter.reload_LLM()
            nr_iteration += 1

            # --- try to exec the repaired cost function ---
            try:
                self.repair(result)
                repair_error_message = None
            except Exception as e:
                error_traceback = (
                    traceback.format_exc()
                )  # This gets the traceback as a string
                print("*\t !! Errors while repairing: ", error_traceback)
                repair_error_message = error_traceback

            self.cost_list.append(self.current_cost)
        print("[DrPlanner] Ends.")
        return result

    def repair(self, diagnosis_result: Union[str, None]):
        # ----- heuristic function -----
        updated_cost_function = diagnosis_result[self.prompter.COST_FUNCTION]
        updated_cost_function = textwrap.dedent(updated_cost_function)
        # Create a namespace dictionary to hold the compiled function
        function_namespace = {}
        function_namespace.update(self.motion_planner.__dict__)
        # initialize imports:
        function_namespace["np"] = np
        function_namespace["Optional"] = Optional
        function_namespace["TrajectorySample"] = TrajectorySample

        # Execute the updated heuristic function string
        try:
            exec(updated_cost_function, globals(), function_namespace)
        except Exception as e:
            # Handle exceptions (e.g., compilation errors)
            raise RuntimeError(f"Error compiling heuristic function: {e}")

        # Extract the new function
        new_cost_function = function_namespace["evaluate"]

        if not callable(new_cost_function):
            raise ValueError("No valid 'heuristic_function' found after execution")

        # Bind the function to the StudentMotionPlanner instance
        self.cost_function.evaluate = new_cost_function.__get__(self.cost_function)

        self.cost_function = self.DefaultCostFunction(
            self.motion_planner.x_0.velocity, desired_d=0.0, desired_s=None
        )

        self.cost_function.evaluate = MethodType(new_cost_function, self.cost_function)

    def describe(
        self,
        planned_trajectory: Union[Trajectory, None],
        diagnosis_result: Union[str, None],
    ) -> (str, PlanningProblemCostResult):

        template = self.prompter.algorithm_template

        if diagnosis_result is None:
            planner_description = self.prompter.generate_planner_description(
                self.cost_function, None
            )
        else:
            updated_cost_function = diagnosis_result[self.prompter.COST_FUNCTION]
            updated_cost_function = textwrap.dedent(updated_cost_function)
            planner_description = self.prompter.generate_planner_description(
                None, updated_cost_function
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

    def plan(self, nr_iter: int) -> Trajectory:
        solution = run_planner(
            self.motion_planner, self.motion_planner_config, self.cost_function
        )
        planning_problem_solution = solution.planning_problem_solutions[0]
        trajectory_solution = planning_problem_solution.trajectory

        # todo: find a good way to visualize solution

        if self._save_solution:
            # write solution to a CommonRoad XML file
            csw = CommonRoadSolutionWriter(solution)
            target_folder = self.dir_output + "sampling/solutions/"
            os.makedirs(
                os.path.dirname(target_folder), exist_ok=True
            )  # Ensure the directory exists
            csw.write_to_file(
                output_path=target_folder,
                filename=f"solution_{solution.benchmark_id}_iter_{nr_iter}.xml",
                overwrite=True,
            )
        return trajectory_solution
