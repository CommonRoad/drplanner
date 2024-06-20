from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import BestFirstSearch
import numpy as np

class StudentMotionPlanner(BestFirstSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """
    
    def _init_(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super()._init_(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

        if plot_config.SAVE_FIG:
            self.path_fig = '../figures/astar/'
        else:
            self.path_fig = None

    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of A* is f(n) = g(n) + h(n)
        """
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)
        # calculate g(n)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(node_current=node_current)

    def heuristic_function(self, node_current: PriorityNode) -> float:

        path_last = node_current.list_paths[-1]

        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - path_last[-1].time_step

        # cost lanelet
        cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path_last)

        if cost_lanelet is None or final_lanelet_id[0] is None:
            return np.inf

        else:
            pathEffic = self.calc_path_efficiency(path_last)
            velocity = path_last[-1].velocity

            return ((self.calc_euclidean_distance(current_node=node_current)) / (1.3 * velocity)) * (1 / pathEffic) ** 4





        