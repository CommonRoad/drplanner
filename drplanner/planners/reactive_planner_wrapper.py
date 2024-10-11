import os
import re
import time
import copy
import traceback
from types import MethodType
from typing import Type, Tuple, Union

import numpy as np
from scipy.integrate import simps

from commonroad.common.solution import VehicleType, CostFunction
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.draw_params import ShapeParams
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_dc.costs.evaluation import (
    CostFunctionEvaluator,
    PlanningProblemCostResult,
)
from commonroad_route_planner.route import Route
from commonroad_route_planner.route_planner import RoutePlanner

from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_rp.trajectories import TrajectorySample

from commonroad_rp.utility.evaluation import create_full_solution_trajectory
from commonroad_rp.cost_function import (
    CostFunction as ReactiveCostFunction,
    DefaultCostFunction,
)
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.config import (
    ReactivePlannerConfiguration,
    VehicleConfiguration,
)

from drplanner.describer.base import PlanningException


def get_planner(config: ReactivePlannerConfiguration) -> Tuple[ReactivePlanner, Route]:
    """
    Generates a new instance of the ReactivePlanner based in the given configuration.
    """
    # run route planner and add reference path to config
    route_planner = RoutePlanner(
        config.scenario.lanelet_network, config.planning_problem
    )
    route = route_planner.plan_routes().retrieve_first_route()

    planner = ReactivePlanner(config)
    # set reference path for curvilinear coordinate system
    planner.set_reference_path(route.reference_path)
    return planner, route


def run_planner(
    planner: ReactivePlanner,
    config: ReactivePlannerConfiguration,
    cost_function: Type[ReactiveCostFunction | None],
) -> Trajectory:
    """
    Attempts to calculate a trajectory through iterative re-planning.
    raise: PlanningException
    """
    print(
        f"Start planning with a horizon of {config.planning.time_steps_computation/10} sec"
    )
    planner.set_cost_function(cost_function)
    # set limit in seconds to restrict the length of the planning process
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


def plot_planner(
    scenario: Scenario,
    planning_problem: PlanningProblemSet,
    route: Union[Route, None],
    trajectory: Union[Trajectory, None],
    config: Union[VehicleConfiguration, None],
    abs_save_path: str,
):
    """
    Creates a visual depiction of the planning result
    and saves it as png to abs_save_path.
    """
    if not route:
        planning_problem = copy.deepcopy(
            list(planning_problem.planning_problem_dict.values())[0]
        )
        route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
        route = route_planner.plan_routes().retrieve_first_route()
    lanelets_to_be_shown = route.lanelet_ids

    # remove the vehicles not in the related lanelets
    for veh in scenario.obstacles:
        if veh.initial_shape_lanelet_ids.isdisjoint(lanelets_to_be_shown):
            scenario.remove_obstacle(veh)

    if not trajectory:
        rnd = MPRenderer(figsize=(20, 10))
        rnd.draw_params.lanelet_network.draw_ids = lanelets_to_be_shown
        planning_problem.draw(rnd)
        scenario.draw(rnd)
        rnd.render(show=False, filename=abs_save_path)
        return

    state_list = trajectory.state_list
    # get plot limits from trajectory
    position_array = np.array([state.position for state in state_list])
    x_min = np.min(position_array[:, 0]) - 20
    x_max = np.max(position_array[:, 0]) + 20
    y_min = np.min(position_array[:, 1]) - 20
    y_max = np.max(position_array[:, 1]) + 20
    plot_limits = [x_min, x_max, y_min, y_max]

    # create renderer object (if no existing renderer is passed)
    rnd = MPRenderer(figsize=(20, 10), plot_limits=plot_limits)
    rnd.draw_params.lanelet_network.draw_ids = lanelets_to_be_shown

    # set renderer draw params
    rnd.draw_params.time_begin = 0
    rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
    rnd.draw_params.planning_problem.initial_state.state.radius = 0.5

    # set occupancy shape params
    occ_params = ShapeParams()
    occ_params.facecolor = "#E37222"
    occ_params.edgecolor = "#9C4100"
    occ_params.opacity = 1.0
    occ_params.zorder = 51

    scenario.draw(rnd)
    planning_problem.draw(rnd)

    # visualize occupancies of trajectory
    for i in range(len(state_list)):
        state = state_list[i]
        occ_pos = Rectangle(
            length=config.length,
            width=config.width,
            center=state.position,
            orientation=state.orientation,
        )
        if i >= 1:
            occ_params.opacity = 0.3
            occ_params.zorder = 50
        occ_pos.draw(rnd, draw_params=occ_params)

    # visualize trajectory
    pos = np.asarray([state.position for state in state_list])
    rnd.ax.plot(
        pos[:, 0],
        pos[:, 1],
        color="k",
        marker="x",
        markersize=3.0,
        markeredgewidth=0.4,
        zorder=21,
        linewidth=0.8,
    )

    # render scenario and occupancies
    rnd.render(show=False, filename=abs_save_path)


def get_method_name_from_exception(e: Exception) -> Union[str, None]:
    tb = traceback.extract_tb(e.__traceback__)
    if tb:
        # The last entry in the traceback is where the exception occurred?
        last_entry = tb[-1]
        return last_entry.name
    return None


def get_basic_cost_function() -> str:
    """
    Basic cost function according to SM1.
    """
    return """def evaluate(self, trajectory: TrajectorySample) -> float:
    cost = 0.0
    cost += 50.0 * self.acceleration_costs(trajectory)
    cost += 50.0 * self.orientation_offset_costs(trajectory)
    cost += 50.0 * self.steering_angle_costs(trajectory)
    cost += 50.0 * self.steering_velocity_costs(trajectory)
    cost += 1.0 * self.path_length_costs(trajectory)
    return cost"""


def get_basic_configuration_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "standard-config.yaml"
    )


def get_basic_method(name: str):
    print(f"Added missing helper method {name}")
    return f"""def {name}(trajectory: TrajectorySample) -> float:
    return 0.0"""


class ReactiveMotionPlannerWrapper:
    """
    Wrapper class representing a ReactivePlanner.
    It handles automatic integration of cost function strings.
    """

    def __init__(
        self,
        cost_function_string: str = None,
        helper_methods: list[str] = None,
        max_time_steps: int = None,
        d: float = None,
    ):
        if not cost_function_string and helper_methods is None:
            self.cost_function_string = get_basic_cost_function()
            self.helper_methods = []
        elif not cost_function_string:
            self.cost_function_string = get_basic_cost_function()
        elif helper_methods is None:
            self.helper_methods = []
        else:
            self.cost_function_string = cost_function_string
            self.helper_methods = helper_methods
        if not max_time_steps:
            self.max_time_steps = 30
        else:
            self.max_time_steps = max_time_steps

        if not d:
            self.d = 3
        else:
            self.d = d

        self.missing_helper_method = None
        self.last_missing_helper_method = None

    def evaluate_on_scenario(
        self,
        absolute_scenario_path: str,
        absolute_config_path: str = None,
        cost_type: CostFunction = CostFunction.SM1,
        absolute_save_path: str = None,
        counter: int = 0,
    ) -> Tuple[PlanningProblemCostResult, int]:
        """
        Run the motion planner on the given benchmark.
        Vehicle model and type are fixed.
        Recursively handles exceptions which occur in helper methods.
        """
        if not absolute_config_path:
            absolute_config_path = get_basic_configuration_path()
        # if a helper method used in the cost function does not exist, add a mockup instead
        if self.missing_helper_method:
            self.helper_methods.append(get_basic_method(self.missing_helper_method))

        # --- initialization ---
        cost_evaluator = CostFunctionEvaluator(cost_type, VehicleType.BMW_320i)
        scenario, planning_problem_set = CommonRoadFileReader(
            absolute_scenario_path
        ).open(True)
        planning_problem = copy.deepcopy(
            list(planning_problem_set.planning_problem_dict.values())[0]
        )
        planner_config = ReactivePlannerConfiguration.load(
            absolute_config_path, absolute_scenario_path
        )
        planner_config.planning.time_steps_computation = self.max_time_steps
        planner_config.sampling.d_max = self.d
        planner_config.sampling.d_min = -self.d
        planner_config.update()

        # --- integrate all components to obtain the final motion planner ---
        planner, route, planner_cost_function = ReactiveMotionPlannerWrapper.apply(
            self.cost_function_string, self.helper_methods, planner_config
        )

        # --- start the planning process ---
        try:
            trajectory = run_planner(planner, planner_config, planner_cost_function)
        except AttributeError as e:
            self.last_missing_helper_method = self.missing_helper_method  # cache
            # determine the name of the helper method causing the exception
            self.missing_helper_method = get_method_name_from_exception(e)
            if self.missing_helper_method == "evaluate":
                self.missing_helper_method = e.name
            # if exception handling was futile, fail the planning
            if self.missing_helper_method == self.last_missing_helper_method:
                raise e
            # otherwise try to restart the planning
            return self.evaluate_on_scenario(
                absolute_scenario_path,
                absolute_config_path=absolute_config_path,
                cost_type=cost_type,
                absolute_save_path=absolute_save_path,
                counter=(counter + 1),
            )

        # --- plot the planning result ---
        if absolute_save_path:
            plot_planner(
                scenario,
                planning_problem_set,
                route,
                trajectory,
                planner_config.vehicle,
                absolute_save_path,
            )

        return (
            cost_evaluator.evaluate_pp_solution(scenario, planning_problem, trajectory),
            counter,
        )

    def __str__(self):
        description = (
            f"The planner uses a planning horizon of {self.max_time_steps/10} seconds. "
        )
        description += f"This is its cost function:\n {self.cost_function_string}"
        return description

    @staticmethod
    def extract_method_names(helper_methods: list[str]) -> list[str]:
        """
        Maps helper methods to their method names.
        """
        names = []
        for method in helper_methods:
            match = re.search(r"def\s+(\w+)\s*\(", method)
            if match:
                names.append(match.group(1))
            else:
                names.append(None)
        return names

    @staticmethod
    def join_method_lines(lines):
        """
        Combines a list of strings to a single helper method.
        """
        methods = []
        current_method = ""

        for line in lines:
            if line.strip().startswith("def") and current_method:
                methods.append(current_method)
                current_method = ""
            current_method += line + "\n"

        if current_method:
            methods.append(current_method)

        return methods

    @staticmethod
    def preprocess_helper_methods(
        helper_methods: list[str],
    ) -> list[Tuple[str, str, bool]]:
        """
        Handles typical LLM mistakes/hallucinations.
        Maps helper method string to method name and code.
        Determines whether the method can be static.
        """
        result = []
        if not helper_methods:
            return result

        # check if the LLM provided a split-up method instead of mutiple complete ones
        split_up_method = all(["\n" not in x for x in helper_methods])
        if split_up_method:
            helper_methods = ReactiveMotionPlannerWrapper.join_method_lines(helper_methods)

        for method in helper_methods:
            # handle the case that the LLM forgot to include a return statement
            method = re.sub(r"\bpass\b", "return 0.0", method)
            match = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", method)
            if match:
                static = True
                name = match.group(1)
                params = match.group(2).split(",")
                if params and params[0].strip() == "self":
                    static = False
                result.append((name, method, static))
            else:
                result.append((None, None, None))
        return result

    @staticmethod
    def defuse(code: str) -> str:
        """
        Makes flawed code compilable.
        """
        signature = code.split("\n")[0]
        print(f"Had to defuse {signature}")
        return f"{signature}    return 0.0"

    @staticmethod
    def apply(
        cost_function_string: str,
        helper_methods: list[str],
        planner_config: ReactivePlannerConfiguration,
    ) -> Tuple[ReactivePlanner, Route, Type[ReactiveCostFunction | None]]:
        """
        Compile and integrate the strings representing cost function and helper methods.
        """
        planner, route = get_planner(planner_config)
        # Create a namespace dictionary to hold the compiled function
        function_namespace = {}
        function_namespace.update(planner.__dict__)
        function_namespace["np"] = np
        function_namespace["simps"] = simps
        function_namespace["TrajectorySample"] = TrajectorySample

        # add helper methods to name space piece by piece
        preprocess_result = ReactiveMotionPlannerWrapper.preprocess_helper_methods(
            helper_methods
        )
        for name, code, static in preprocess_result:
            if not name:
                continue
            try:
                exec(code, globals(), function_namespace)
            except Exception as _:
                exec(ReactiveMotionPlannerWrapper.defuse(code), globals(), function_namespace)
            method = function_namespace[name]
            if callable(method):
                if static:
                    setattr(DefaultCostFunction, name, staticmethod(method))
                else:
                    setattr(DefaultCostFunction, name, method)

        try:
            exec(cost_function_string, globals(), function_namespace)
        except Exception as e:
            print(e)
        # Extract the new function
        cost_function_code = function_namespace["evaluate"]
        if not callable(cost_function_code):
            raise ValueError("No valid 'cost_function' found after execution")
        # noinspection PyUnresolvedReferences
        planner.cost_function.evaluate = cost_function_code.__get__(
            planner.cost_function
        )
        cost_function = DefaultCostFunction(
            planner.x_0.velocity, desired_d=0.0, desired_s=None
        )
        cost_function.evaluate = MethodType(cost_function_code, cost_function)

        return planner, route, cost_function
