from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch
from SMP.motion_planner.plot_config import StudentScriptPlotConfig
from commonroad.geometry.shape import Rectangle, Polygon, ShapeGroup, Circle
import numpy as np
import math


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
        ########################################################################
        # todo: Implement your own evaluation function here.                   #
        ########################################################################
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(
                node_current.list_paths
            )

        node_current.priority += (
            len(node_current.list_paths[-1]) - 1
        ) * self.scenario.dt

        return self.heuristic_function(node_current=node_current)

    def heuristic_function(self, node_current: PriorityNode) -> float:

        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if self.position_desired is None:
            # e.g. self.scenatio.dt = 0.1, self.time_desired.start=33

            return self.time_desired.start - node_current.list_paths[-1][-1].time_step
        else:
            count = 0
            velocity = node_current.list_paths[-1][-1].velocity
            if velocity in self.velocity_desired:
                count += 1
            distance_to_goal = self.calc_euclidean_distance(current_node=node_current)
            if np.isclose(velocity, 0):
                return np.inf

            else:
                time_to_goal = distance_to_goal / velocity

                # returns id of the start lanelet
                cur_lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position(
                    [node_current.list_paths[-1][-1].position]
                )[0]
                if not cur_lanelet_id:
                    return time_to_goal
                else:

                    time_step = node_current.list_paths[-1][-1].time_step
                    lanelet_id = cur_lanelet_id[0]
                    pos = node_current.list_paths[-1][-1].position
                    obstacles_in_lanelet = self.get_obstacles(
                        self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id),
                        time_step,
                    )

                    shortestDist = math.inf
                    for obstacleObj in obstacles_in_lanelet:
                        shape_obs = obstacleObj.occupancy_at_time(time_step).shape
                        if isinstance(shape_obs, Circle):
                            if self.distance(pos, shape_obs.center) < shortestDist:
                                shortestDist = self.distance(pos, shape_obs.center)
                        elif isinstance(shape_obs, Rectangle):
                            if self.distance(pos, shape_obs.center) < shortestDist:
                                shortestDist = self.distance(pos, shape_obs.center)

                    if shortestDist > 10:
                        count += 2
                    if any(id in cur_lanelet_id for id in self.list_ids_lanelets_goal):
                        count += 3
                    orientation = node_current.list_paths[-1][-1].orientation
                    if orientation in self.orientation_desired:
                        count += 1

                score = time_to_goal * (1 - 0.1 * count)
                return score
