from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch

import copy
import time
import numpy as np
from abc import abstractmethod, ABC
from typing import Tuple, Dict, Any, List, Union

from commonroad.scenario.state import KSState

from SMP.maneuver_automaton.motion_primitive import MotionPrimitive

# from SMP.motion_planner.node import PriorityNode
# from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.utility import (
    MotionPrimitiveStatus,
    initial_visualization,
    update_visualization,
)
from SMP.motion_planner.queue import PriorityQueue
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
import SMP.batch_processing.timeout_config


class StudentMotionPlanner(AStarSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the AStarSearch planner.
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
        # todo: Implement your own evaluation function here.
        #
        # Copied from AStarSearch
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

        path = node_current.list_paths[-1][-1]

        pathLength = self.calc_travelled_distance(node_current.list_paths[-1])
        length = self.calc_heuristic_distance(path)
        # if Distance couldn't be calculated:
        if length is None:
            return np.inf

        # Velocity (I used the heuristic for best first search as basis):
        if self.position_desired is None:
            velocity = (
                self.time_desired.start - node_current.list_paths[-1][-1].time_step
            )
        else:
            vel = path.velocity
            if np.isclose(vel, 0):
                return np.inf
            else:
                velocity = self.calc_euclidean_distance(current_node=node_current) / vel

        # Orientation angle
        angleToGoal = self.calc_angle_to_goal(path)
        orientationToGoalDiff = abs(
            self.calc_orientation_diff(angleToGoal, path.orientation)
        )

        # Time needed
        cost_time = self.calc_time_cost(node_current.list_paths[-1])

        return orientationToGoalDiff + 8 * velocity + cost_time + 15 * (pathLength)
