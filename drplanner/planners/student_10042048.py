from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch


import numpy as np


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

        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step

        else:
            velocity = node_current.list_paths[-1][-1].velocity

            if np.isclose(velocity, 0):
                return np.inf

            euclDistance = (
                self.calc_euclidean_distance(current_node=node_current) / velocity
            )

            if hasattr(self.planningProblem.goal.state_list[0], "velocity"):
                v_mean_goal = (
                    self.planningProblem.goal.state_list[0].velocity.start
                    + self.planningProblem.goal.state_list[0].velocity.end
                ) / 2
                diff_vel = (
                    abs(node_current.list_paths[-1][-1].velocity - v_mean_goal) ** 2
                )
            else:
                diff_vel = 0

            angleToGoal = self.calc_angle_to_goal(node_current.list_paths[-1][-1])
            orientationToGoalDiff = self.calc_orientation_diff(
                angleToGoal, node_current.list_paths[-1][-1].orientation
            )

            weights = np.zeros(3)

            weights[0] = 6
            weights[1] = 4
            weights[2] = 1

            total = np.sum(weights)
            weights /= total

            cost = (
                weights[0] * euclDistance
                + weights[1] * orientationToGoalDiff
                + weights[2] * diff_vel
            )
