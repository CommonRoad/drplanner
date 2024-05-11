# standard imports
from typing import Tuple

from commonroad.common.solution import Solution

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner

# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.evaluation import run_evaluation
from commonroad_rp.utility.config import ReactivePlannerConfiguration


# def get_planner(filename) -> Tuple[ReactivePlannerConfiguration, ReactivePlanner]:
#    # Build config object
#    config = ReactivePlannerConfiguration.load(f"standard-config.yaml", filename)
#    config.update()
#    # run route planner and add reference path to config
#    route_planner = RoutePlanner(config.scenario, config.planning_problem)
#    route = route_planner.plan_routes().retrieve_first_route()
#
#    # initialize reactive planner
#    planner = ReactivePlanner(config)
#
#    # set reference path for curvilinear coordinate system
#    planner.set_reference_path(route.reference_path)
#    return config, planner
#
#
# def run_planner(
#    planner: ReactivePlanner, config: ReactivePlannerConfiguration
# ) -> Tuple[Solution, list[bool]]:
#    # Add first state to recorded state and input list
#    planner.record_state_and_input(planner.x_0)
#
#    while not planner.goal_reached():
#        current_count = len(planner.record_state_list) - 1
#
#        # check if planning cycle or not
#        plan_new_trajectory = current_count % config.planning.replanning_frequency == 0
#        if plan_new_trajectory:
#            # new planning cycle -> plan a new optimal trajectory
#            planner.set_desired_velocity(current_speed=planner.x_0.velocity)
#            optimal = planner.plan()
#            if not optimal:
#                break
#
#            planner.record_state_and_input(optimal[0].state_list[1])
#            planner.reset(
#                initial_state_cart=planner.record_state_list[-1],
#                initial_state_curv=(optimal[2][1], optimal[3][1]),
#                collision_checker=planner.collision_checker,
#                coordinate_system=planner.coordinate_system,
#            )
#        else:
#            # continue on optimal trajectory
#            temp = current_count % config.planning.replanning_frequency
#
#            planner.record_state_and_input(optimal[0].state_list[1 + temp])
#            planner.reset(
#                initial_state_cart=planner.record_state_list[-1],
#                initial_state_curv=(optimal[2][1 + temp], optimal[3][1 + temp]),
#                collision_checker=planner.collision_checker,
#                coordinate_system=planner.coordinate_system,
#            )
#    solution,_ = run_evaluation(
#        planner.config, planner.record_state_list, planner.record_input_list
#    )
#    return solution.planning_problem_solutions[0].trajectory
#
