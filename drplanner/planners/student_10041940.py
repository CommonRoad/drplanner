import numpy as np
import math
from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch


class StudentMotionPlanner(GreedyBestFirstSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)

    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of GBFS is f(n) = h(n)
        """

        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:

        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0
        mean_time = (self.time_desired.start + self.time_desired.end) / 2
        if self.position_desired is None:
            return mean_time - node_current.list_paths[-1][-1].time_step

        position_difference = self.calc_euclidean_distance(current_node=node_current)

        velocity = node_current.list_paths[-1][-1].velocity
        mean_end_velocity = (self.velocity_desired.start + self.velocity_desired.end) / 2
        velocity_difference = abs(velocity - mean_end_velocity)

        angle = self.calc_angle_to_goal(node_current.list_paths[-1][-1])
        orientation_difference = abs(self.calc_orientation_diff(angle, node_current.list_paths[-1][-1].orientation))

        time_difference = mean_time - node_current.list_paths[-1][-1].time_step

        weights = np.zeros(6)
        weights[0] = 5
        weights[1] = 6
        weights[2] = 15
        weights[3] = 0
        cost = weights[0] * position_difference + weights[1] * position_difference / velocity * math.sin(
            orientation_difference) - weights[2] * velocity_difference

        factor = 1 / self.calc_path_efficiency(node_current.list_paths[-1])

        return cost

