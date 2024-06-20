import numpy as np

from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch


class StudentMotionPlanner(GreedyBestFirstSearch):
    def __init__(
            self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig
    ):
        super().__init__(
            scenario=scenario,
            planningProblem=planningProblem,
            automaton=automata,
            plot_config=plot_config,
        )
        if plot_config.SAVE_FIG:
            self.path_fig = "../figures/student/"
        else:
            self.path_fig = None

    def heuristic_function(self, node_current: PriorityNode) -> float:
        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step
        else:
            path_last = node_current.list_paths[-1]
            current_node = path_last[-1]
            (
                off_dist,
                end_lanelet_id,
                start_lanelet_id,
            ) = self.calc_heuristic_lanelet(path_last)
            if end_lanelet_id is None:
                return np.inf

            e = self.calc_path_efficiency(path_last)
            v = current_node.velocity

            if np.isclose(v, 0):
                return np.inf
            else:
                t = self.calc_euclidean_distance(current_node=node_current) / v
                return 12 / e + 10 * t + off_dist

    def evaluation_function(self, node_current: PriorityNode) -> float:
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(
                node_current.list_paths
            )
        node_current.priority = self.heuristic_function(node_current=node_current)

        return node_current.priority
