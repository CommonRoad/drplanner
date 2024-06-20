from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import BestFirstSearch
from commonroad.scenario.state import InitialState, KSState
import math
import numpy as np

class StudentMotionPlanner(BestFirstSearch):
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
        Evaluation function of A* is f(n) = g(n) + h(n)
        """
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(
                node_current.list_paths)
        # calculate g(n)
        node_current.priority += (
            len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(node_current=node_current)

    def heuristic_function(self, node_current: PriorityNode) -> float:

        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step

        else:
            path_last = node_current.list_paths[-1]
            velocity = path_last[-1].velocity
            path_angle = path_last[-1].orientation
            cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path_last)

            if cost_lanelet is None or final_lanelet_id[0] is None:
                return np.inf
            # numObs = self.num_obstacles_in_lanelet_at_time_step(path_last[-1].time_step, final_lanelet_id[0])
            if np.isclose(velocity, 0):
                return np.inf

            else:
                return self.calc_euclidean_distance(current_node=node_current) + self.calc_angle_to_goal(path_last[-1]) + cost_lanelet # + numObs