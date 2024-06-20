from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch, AStarSearch

import numpy as np
import math


class StudentMotionPlanner(AStarSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)

    def heuristic_function(self, node_current: PriorityNode) -> float:
        recent_path = node_current.list_paths[-1]
        recent_node = recent_path[-1]
        if self.reached_goal(recent_path):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - recent_node.time_step

        velocity = recent_node.velocity
        distance = self.calc_euclidean_distance(current_node=node_current)

        if np.isclose(velocity, 0) or \
                self.calc_heuristic_distance(recent_node) is None or \
                np.isclose(self.calc_path_efficiency(recent_path), 0):
            return math.inf

        angle_to_goal = self.calc_angle_to_goal(recent_node)
        change_in_angel = self.calc_orientation_diff(angle_to_goal, recent_node.orientation)
        so_far_length = self.calc_travelled_distance(recent_path)

        cost = (100 - so_far_length) + 2 * distance + abs(change_in_angel)

        return max(cost, 0)
