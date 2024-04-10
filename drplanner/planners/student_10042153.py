import numpy as np
from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch


class StudentMotionPlanner(AStarSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)

    def evaluation_function(self, node_current: PriorityNode) -> float:
        ########################################################################
        # todo: Implement your own evaluation function here.                   #
        ########################################################################
        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:
        path_last = node_current.list_paths[-1]

        angleToGoal = self.calc_angle_to_goal(path_last[-1])

        orientationToGoalDiff = self.calc_orientation_diff(angleToGoal, path_last[-1].orientation)

        cost_time = self.calc_time_cost(path_last)

        if self.reached_goal(node_current.list_paths[-1]):
            heur_time = 0.0

        if self.position_desired is None:
            heur_time = self.time_desired.start - node_current.list_paths[-1][-1].time_step

        else:
            velocity = node_current.list_paths[-1][-1].velocity

            if np.isclose(velocity, 0):
                heur_time = np.inf

            else:
                heur_time = self.calc_euclidean_distance(current_node=node_current) / velocity

        cost = 20 * orientationToGoalDiff + 0.5 * cost_time + heur_time
        if cost < 0:
            cost = 0
        return cost
