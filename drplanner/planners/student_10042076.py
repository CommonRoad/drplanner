import numpy as np

from SMP.motion_planner.node import PriorityNode
import SMP.batch_processing.helper_functions as helper
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch

def get_last_time_step_in_scenario(scenario):
    time_steps = [
        len(obs.prediction.occupancy_set) for obs in scenario.dynamic_obstacles
    ]
    return max(time_steps)

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
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)
        # calculate g(n)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(node_current=node_current)

    def heuristic_function(self, node_current: PriorityNode) -> float:
        survival = False
        multiplier = 1
        alignment = 0
        num_obst = 0


        path_last = node_current.list_paths[-1]
        cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path_last)
        distStartState = self.calc_heuristic_distance(path_last[0])
        distLastStateManhattan = self.calc_heuristic_distance(path_last[-1], distance_type=1)
        distLastStateEuclidean = self.calc_heuristic_distance(path_last[-1])

        if len(path_last) > 5:
            vel_avg = self.calc_path_efficiency(path_last[-5:]) * 10
        else:
            vel_avg = self.calc_path_efficiency(path_last) * 10

        cur_lanelet = self.scenario.lanelet_network.find_lanelet_by_position([path_last[-1].position])[0][0]

        if cur_lanelet is not None:
            try:
                num_obst = self.num_obstacles_in_lanelet_at_time_step(path_last[-1].time_step, cur_lanelet)
                laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(cur_lanelet)
                llAngle = laneletObj.orientation_by_position(path_last[-1].position)
                myAngle = path_last[-1].orientation
                alignment = self.calc_orientation_diff(llAngle, myAngle)
            except AssertionError:
                alignment = 0

        if self.position_desired is None:
            survival = True

        if survival:
            # SURVIVAL PROBLEM, NO POSITION GOAL
            if vel_avg > 40:
                multiplier *= 100
            elif survival and vel_avg < 1:
                return np.inf
            elif survival and vel_avg < 5:
                multiplier *= 1000

            if abs(alignment) > np.radians(45):
                multiplier *= 500

            weights_scores = np.array([[20, num_obst],
                                       [10, max(0., 15. - vel_avg)],
                                       [10, 10 * abs(alignment)]])
        else:
            # NOT SURVIVAL, GOAL-ORIENTED
            time_total = 10 * helper.get_last_time_step_in_scenario(self.scenario)
            reqd_avg = distStartState / time_total

            if distStartState < distLastStateEuclidean and not self.reached_goal(path_last):
                multiplier *= 1e7

            if vel_avg > 40 and not reqd_avg > 40:
                multiplier *= 100
            elif np.isclose(vel_avg, 0) and not np.isclose(reqd_avg, 0):
                return np.inf

            if self.reached_goal(path_last):
                pc_cov = 1
            else:
                pc_cov = 1 - (distLastStateEuclidean / distStartState)

            defEuclidean = self.calc_euclidean_distance(current_node=node_current)

            if self.reached_goal(path_last):
                return 0
            elif defEuclidean < 0.5:
                multiplier *= 1e-3
                if cur_lanelet is not None:
                    if self.is_goal_in_lane(cur_lanelet):
                        multiplier *= 1e-2

            if abs(alignment) > np.radians(45):
                multiplier *= 1e4

            weights_scores = np.array([[80, defEuclidean / path_last[-1].velocity],
                                       [10, num_obst],
                                       [10, defEuclidean],
                                       [3, 100 * (1 - pc_cov)],
                                       [1, max(0., reqd_avg - vel_avg)],
                                       [1, 10 * abs(alignment)]])

        # print(weights_scores)
        # print(defEuclidean - distLastStateEuclidean)
        # print(distLastStateEuclidean)
        return max(np.dot(weights_scores[:, 0], weights_scores[:, 1]) * multiplier, 0)