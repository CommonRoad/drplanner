from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch


import numpy as np


# class StudentMotionPlanner(GreedyBestFirstSearch):
class StudentMotionPlanner(AStarSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(
        self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig
    ):
        super().__init__(
            scenario=scenario,
            planningProblem=planningProblem,
            automaton=automata,
            plot_config=plot_config,
        )

    def evaluation_function(self, node_current: PriorityNode) -> float:
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(
                node_current.list_paths
            )
        # calculate g(n)
        node_current.priority += (
            len(node_current.list_paths[-1]) - 1
        ) * self.scenario.dt
        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(
            node_current=node_current
        )

    def heuristic_function(self, node_current: PriorityNode) -> float:

        node_last = node_current.list_paths[-1][-1]
        # lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([node_current.list_paths[0][0].position])[0][0]

        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0
        # survival mode
        if self.position_desired is None:
            # print('Case position_desired is None')
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step

        else:
            # a = self.planningProblem.goal
            angle_diff = self.calc_angle_to_goal(node_last)
            orientationToGoalDiff = self.calc_orientation_diff(
                angle_diff, node_last.orientation
            )

            velocity = node_current.list_paths[-1][-1].velocity
            dist_to_goal = self.calc_euclidean_distance(current_node=node_current)
            dist_to_goal_disc = min(5 * dist_to_goal, 100)
            orient_disc = min(orientationToGoalDiff, 1)

            if np.isclose(velocity, 0):
                return np.inf
            else:
                return dist_to_goal / velocity + orient_disc + dist_to_goal_disc
