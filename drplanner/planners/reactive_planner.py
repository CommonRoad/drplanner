import inspect
import textwrap
import time
import copy
from types import MethodType
from typing import Type, Tuple

from commonroad.common.solution import VehicleType, CostFunction
from commonroad_dc.costs.evaluation import (
    CostFunctionEvaluator,
    PlanningProblemCostResult,
)
from commonroad_route_planner.route_planner import RoutePlanner

from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_rp.trajectories import TrajectorySample

from commonroad_rp.utility.evaluation import create_full_solution_trajectory
from commonroad_rp.cost_function import (
    CostFunction as ReactiveCostFunction,
    DefaultCostFunction,
)
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.config import ReactivePlannerConfiguration

from drplanner.describer.base import PlanningException


def get_planner(config: ReactivePlannerConfiguration) -> ReactivePlanner:
    # run route planner and add reference path to config
    route_planner = RoutePlanner(
        config.scenario.lanelet_network, config.planning_problem
    )
    route = route_planner.plan_routes().retrieve_first_route()
    # initialize reactive planner
    planner = ReactivePlanner(config)
    # set reference path for curvilinear coordinate system
    planner.set_reference_path(route.reference_path)
    return planner


# helper function to run a ReactivePlanner with a given CostFunction
def run_planner(
    planner: ReactivePlanner,
    config: ReactivePlannerConfiguration,
    cost_function: Type[ReactiveCostFunction | None],
):
    # update cost function
    planner.set_cost_function(cost_function)

    print(f"The current planning horizon is {planner.horizon}!")
    max_planning_time = 40
    current_time = time.time()
    # Add first state to recorded state and input list
    planner.record_state_and_input(planner.x_0)
    optimal = None
    while not planner.goal_reached():
        if time.time() - current_time > max_planning_time:
            cause = "Planning took too much time and was terminated!"
            solution = (
                "The vehicle might be driving to slow or is stuck without moving."
            )
            raise PlanningException(cause, solution)
        current_count = len(planner.record_state_list) - 1

        # check if planning cycle or not
        plan_new_trajectory = current_count % config.planning.replanning_frequency == 0

        if plan_new_trajectory:
            # new planning cycle -> plan a new optimal trajectory
            planner.set_desired_velocity(current_speed=planner.x_0.velocity)
            optimal = planner.plan()

            if not optimal:
                cause = "No optimal trajectory could be found!"
                solution = ""
                raise PlanningException(cause, solution)

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

            if not optimal:
                cause = "No optimal trajectory could be found!"
                solution = ""
                raise PlanningException(cause, solution)

            planner.record_state_and_input(optimal[0].state_list[1 + temp])
            planner.reset(
                initial_state_cart=planner.record_state_list[-1],
                initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
                collision_checker=planner.collision_checker,
                coordinate_system=planner.coordinate_system,
            )
    return create_full_solution_trajectory(config, planner.record_state_list)


class ReactiveMotionPlanner:
    def __init__(self, cost_function_string: str):
        if not cost_function_string:
            cost_function = DefaultCostFunction(None, 0.0, None)
            self.cost_function_string = inspect.getsource(cost_function.evaluate)
            self.cost_function_string = textwrap.dedent(self.cost_function_string)
        else:
            self.cost_function_string = cost_function_string

    def apply(
        self, planner_config: ReactivePlannerConfiguration
    ) -> Tuple[ReactivePlanner, Type[ReactiveCostFunction | None]]:
        # todo adjust confog parameters before get_planner()

        planner = get_planner(planner_config)
        # Create a namespace dictionary to hold the compiled function
        function_namespace = {}
        function_namespace.update(planner.__dict__)
        # initialize imports: todo: check how many are needed
        # function_namespace["np"] = np
        # function_namespace["Optional"] = Optional
        function_namespace["TrajectorySample"] = TrajectorySample
        g = globals()
        exec(self.cost_function_string, globals(), function_namespace)
        # Extract the new function
        cost_function_code = function_namespace["evaluate"]
        if not callable(cost_function_code):
            raise ValueError("No valid 'heuristic_function' found after execution")
        planner.cost_function.evaluate = cost_function_code.__get__(
            planner.cost_function
        )
        cost_function = DefaultCostFunction(
            planner.x_0.velocity, desired_d=0.0, desired_s=None
        )
        cost_function.evaluate = MethodType(cost_function_code, cost_function)

        return planner, cost_function

    def evaluate_on_scenario(
        self,
        absolute_scenario_path: str,
        absolute_config_path: str,
        cost_type: CostFunction = CostFunction.SM1,
    ) -> PlanningProblemCostResult:
        scenario, planning_problem_set = CommonRoadFileReader(
            absolute_scenario_path
        ).open(True)
        planning_problem = copy.deepcopy(
            list(planning_problem_set.planning_problem_dict.values())[0]
        )
        planner_config = ReactivePlannerConfiguration.load(
            absolute_config_path, absolute_scenario_path
        )
        planner_config.update()
        planner, planner_cost_function = self.apply(planner_config)

        cost_evaluator = CostFunctionEvaluator(cost_type, VehicleType.BMW_320i)
        trajectory = run_planner(planner, planner_config, planner_cost_function)
        return cost_evaluator.evaluate_pp_solution(
            scenario, planning_problem, trajectory
        )
