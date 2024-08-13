import numpy as np

from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch


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
        ########################################################################
        # todo: Implement your own evaluation function here.                   #
        ########################################################################
        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:

        path_last = node_current.list_paths[-1]
        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        distStartState = self.calc_heuristic_distance(path_last[0])
        distLastState = self.calc_heuristic_distance(path_last[-1])

        if distLastState is None:
            return np.inf
        if distStartState < distLastState:
            return np.inf

        final_lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position(
            [path_last[-1].position]
        )[0]
        cost_lanelet, _, _ = self.calc_heuristic_lanelet(path_last)

        if cost_lanelet is None:
            cost_lanelet = 0

        if final_lanelet_id and final_lanelet_id[0]:
            num_obst = self.num_obstacles_in_lanelet_at_time_step(
                path_last[-1].time_step, final_lanelet_id[0]
            )

            dist_closest_obst = self.calc_dist_to_closest_obstacle(
                final_lanelet_id[0], path_last[-1].position, path_last[-1].time_step
            )
            lanelet_orient = self.calc_lanelet_orientation(
                final_lanelet_id[0], path_last[-1].position
            )
            if dist_closest_obst == np.inf:
                dist_closest_obst = 0
            if num_obst == np.inf:
                num_obst = 0
            if lanelet_orient == np.inf:
                lanelet_orient = 0

        path_eff = self.calc_path_efficiency(path_last)
        angleToGoal = self.calc_angle_to_goal(path_last[-1])
        orientationToGoalDiff = self.calc_orientation_diff(
            angleToGoal, path_last[-1].orientation
        )
        pathLength = self.calc_travelled_distance(path_last)
        cost_time = self.calc_time_cost(path_last)

        if hasattr(self.planningProblem.goal.state_list[0], "velocity"):
            v_mean_goal = (
                self.planningProblem.goal.state_list[0].velocity.start
                + self.planningProblem.goal.state_list[0].velocity.end
            ) / 2
            dist_vel = abs(path_last[-1].velocity - v_mean_goal)
        else:
            dist_vel = 0

        ############################################################################
        if self.position_desired is None:
            time_to_goal = (
                self.time_desired.start - node_current.list_paths[-1][-1].time_step
            )
        else:
            velocity = node_current.list_paths[-1][-1].velocity
            if np.isclose(velocity, 0):
                return np.inf
            else:
                time_to_goal = (
                    self.calc_euclidean_distance(current_node=node_current) / velocity
                )

        if time_to_goal:
            goal_vals = {}
            for node in path_last:
                if hasattr(node, "yaw_rate") and node.yaw_rate:
                    c = 1
                for attr in node.attributes:
                    if attr == "position":
                        continue
                    if attr not in goal_vals.keys():
                        goal_vals[attr] = 0
                    if getattr(node, attr):
                        goal_vals[attr] += getattr(node, attr)

        ############################################################################

        dist0 = self.calc_heuristic_distance(path_last[-1], distance_type=0)
        dist1 = self.calc_heuristic_distance(path_last[-1], distance_type=1)
        dist2 = self.calc_heuristic_distance(path_last[-1], distance_type=2)
        dist3 = self.calc_heuristic_distance(path_last[-1], distance_type=3)
        dist4 = self.calc_heuristic_distance(path_last[-1], distance_type=4)
        dist5 = self.calc_heuristic_distance(path_last[-1], distance_type=5)
        dist6 = self.calc_heuristic_distance(path_last[-1], distance_type=6)
        dist7 = self.calc_heuristic_distance(path_last[-1], distance_type=7)

        weights = {
            "w0": 0.00046194967056635555,
            "w1": 0.9828827813273089,
            "w2": 0.5182104410628761,
            "w3": 0.28949426420057567,
            "w4": 0.0,
            "w5": 0.07552486701020841,
            "w6": 0.3809167876077965,
            "w7": 0.0,
            "w8": 0.964963017720853,
            "w9": 0.49740311687522976,
            "w10": 0.5844063564815822,
            "w11": 0.11975550478664763,
            "w12": 0.8947927982104272,
            "w13": 1.2881120263060735,
            "w14": 0.15253865718828405,
            "w15": 1.4584231000864778,
            "w16": 0.05523276676937928,
            "w17": 0.2636324582012759,
        }
        weights = list(weights.values())

        cost = (
            weights[11] * (cost_lanelet / len(path_last)) / 1.18
            + weights[10] * abs(orientationToGoalDiff) / 3.14
            + weights[2] * cost_time
            + weights[3] * (distLastState) / 214.6
            + weights[4] * abs(100 - pathLength) / 100
            + weights[5] * dist_vel / 9.28
            + weights[6] * (time_to_goal - 0.06) / 139.04
            + weights[7] * path_eff * 0
            + weights[8] * dist1 / 302.89
            + weights[9] * (dist0) / 214.6
            + weights[12] * dist2 / 175.11
            + weights[13] * dist3 / 46060.07
            + weights[14] * dist4 / 151.45
            + weights[15] * dist5 / 23030.04
            + weights[16] * dist6 / 2
            + weights[17] * dist7
            + 8.53 / 10.45
            + weights[18] * list(goal_vals.values())[0]
            + weights[19] * list(goal_vals.values())[1]
            + weights[20] * list(goal_vals.values())[2]
            + weights[21] * list(goal_vals.values())[3]
        )

        return cost
