from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass

from commonroad.scenario.state import KSState
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
        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:

        last_path = node_current.list_paths[-1]

        cost_path_efficiency = self.calc_path_efficiency(last_path)
        cost_orientation = self.calc_orientation_diff(last_path[-1])
        cost_dist_to_LastState = self.calc_heuristic_distance(last_path[-1])
        cost_dist_to_FirstState = self.calc_heuristic_distance(last_path[0])
        cost_time = abs(self.calc_time_cost(last_path))
        cost_vel = self.calc_velocity_difference(last_path[-1])

        # Preliminary checks
        if cost_dist_to_LastState is None:
            return np.inf

        if cost_dist_to_FirstState < cost_dist_to_LastState:
            return np.inf

        weights = np.zeros(5)

        # Weigth definition
        weights[0] = 5  # Path efficiency [Is alredy between 0 and 1]
        weights[1] = (
            30  # Orientation Radians very low value, so we need a high weight to balance out
        )
        weights[2] = 3  # Distance to last state
        weights[3] = 3
        weights[4] = 2

        cost = (
            weights[0] * 1 / cost_path_efficiency
            + weights[1] * cost_orientation
            + weights[2] * cost_dist_to_LastState
            + weights[3] * cost_time
            + weights[4] * cost_vel
        )

        return cost

    # Defined alternative functions for Velocity and angle calculation

    def calc_velocity_difference(self, state: KSState) -> float:
        """Calculates the velocity diff to the goal."""
        if hasattr(self.planningProblem.goal.state_list[0], "velocity"):
            statevelocity = state.velocity
            targetvelocity = (
                self.planningProblem.goal.state_list[-1].velocity.start
                + self.planningProblem.goal.state_list[-1].velocity.end
            ) / 2
            velocity_difference = abs(statevelocity - targetvelocity)
            return velocity_difference
        else:
            return 0

    def calc_orientation_diff(self, state: KSState) -> float:
        """Calculates the orientation diff to the goal."""
        if hasattr(self.planningProblem.goal.state_list[0], "orientation"):

            stateorientation = state.orientation
            targetorientation = (
                self.planningProblem.goal.state_list[0].orientation.start
                + self.planningProblem.goal.state_list[0].orientation.end
            ) / 2
            orientation_difference = SearchBaseClass.calc_orientation_diff(
                stateorientation, targetorientation
            )
            return orientation_difference
        else:
            return 0
