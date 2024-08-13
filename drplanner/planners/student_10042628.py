import numpy as np

from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch


class StudentMotionPlanner(AStarSearch):

    def __init__(
        self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig
    ):
        super().__init__(
            scenario=scenario,
            planningProblem=planningProblem,
            automaton=automata,
            plot_config=plot_config,
        )
        self.velocity_desired_mean = None
        if self.velocity_desired is not None:
            self.velocity_desired_mean = (
                self.velocity_desired.start + self.velocity_desired.end
            ) / 2

    def heuristic_function(self, node_current: PriorityNode) -> float:
        path_last = node_current.list_paths[-1]
        velocity = path_last[-1].velocity

        if self.position_desired is None:
            exp_travel_time = self.time_desired.start - path_last[-1].time_step
        else:
            if np.isclose(velocity, 0):
                exp_travel_time = np.inf
            else:
                exp_travel_time = (
                    self.calc_euclidean_distance(current_node=node_current) / velocity
                )

        cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(
            path_last
        )

        obstacles = 0
        if final_lanelet_id is not None:
            obstacles = self.num_obstacles_in_lanelet_at_time_step(
                path_last[-1].time_step, final_lanelet_id[0]
            )

        orientationToGoalDiff = self.calc_orientation_diff(
            self.calc_angle_to_goal(path_last[-1]), path_last[-1].orientation
        )
        orientationToGoalDiff = (orientationToGoalDiff + np.pi) % (2 * np.pi) - np.pi

        cost = 2 * exp_travel_time + 2 * obstacles + 1 * orientationToGoalDiff

        return max(cost, 0)
