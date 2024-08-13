import numpy as np
from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch
import time


class StudentMotionPlanner(GreedyBestFirstSearch):
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
        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:

        ##########################
        path_last = node_current.list_paths[-1]

        goaldistLastState = self.calc_heuristic_distance(path_last[-1])

        current_lanelet = self.scenario.lanelet_network.find_lanelet_by_position(
            [path_last[-1].position]
        )[0]
        current_lanelet = current_lanelet[0]
        dist_closest_obstacle = self.calc_dist_to_closest_obstacle(
            current_lanelet, path_last[-1].position, path_last[-1].time_step
        )
        number_of_obstacles = self.num_obstacles_in_lanelet_at_time_step(
            path_last[-1].time_step, current_lanelet
        )

        ##########################

        if self.reached_goal(path_last):
            return 0.0

            #####################################
        elif self.position_desired is None:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step

        elif node_current.list_paths[-1][-1].time_step >= max(self.time_desired):
            return np.inf

        else:

            goal = self.is_goal_in_lane(current_lanelet, traversed_lanelets=None)
            dist_heu_lanelet, _, _ = self.calc_heuristic_lanelet(path_last)
            velocity = node_current.list_paths[-1][-1].velocity

            if goaldistLastState > 0.1:
                heu_dist = goaldistLastState / self.calc_heuristic_distance(
                    node_current.list_paths[0][0]
                )

            if goaldistLastState < 0.1:
                heu_dist = (
                    goaldistLastState
                    / self.calc_heuristic_distance(node_current.list_paths[0][0])
                    * 0.5
                )
            if dist_heu_lanelet != None:
                heu_lanelet = dist_heu_lanelet

            if dist_heu_lanelet == None:
                heu_lanelet = 0

            heu_eff = 1 / self.calc_path_efficiency(path_last)

            if np.isclose(velocity, 0):
                heu_time = 10

            if velocity > 0.1:
                heu_time = (
                    self.calc_euclidean_distance(current_node=node_current) / velocity
                )

            if goal == True:
                heu_goal = 1

            if goal == False:
                heu_goal = 2

            if (self.calc_euclidean_distance(current_node=node_current) / 20) > (
                max(self.time_desired) - node_current.list_paths[-1][-1].time_step
            ):
                return np.inf

            # heu_orient = abs((self.calc_angle_to_goal(path_last[-1]) - path_last[-1].orientation / self.calc_angle_to_goal(path_last[-1])))

            weights = np.array([0.1, 0.25, 0.5, 0.2, 1])
            heuristic = np.array([heu_time, heu_goal, heu_eff, heu_lanelet, heu_dist])
            weighted_heu = weights * heuristic
            sum_weighted_heu = sum(weighted_heu)
            return sum_weighted_heu
