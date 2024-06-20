import numpy as np
from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import BestFirstSearch


class StudentMotionPlanner(BestFirstSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)
        
        if plot_config.SAVE_FIG:
            self.path_fig = '../figures/student/'
        else:
            self.path_fig = None


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
        last_path = node_current.list_paths[-1]


        if self.reached_goal(node_current.list_paths[-1]):
            return  0.0

        #time difference
        if self.position_desired is None:
            c_time = self.time_desired.start - last_path[-1].time_step

        else:
            velocity = last_path[-1].velocity

            if np.isclose(velocity, 0):
                return np.inf

            else:
                c_time = self.calc_euclidean_distance(current_node=node_current) / velocity

        #positional distance
        c_position = self.calc_heuristic_distance(last_path[-1])

        #orientation difference
        if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):
            orientation_center = (self.orientation_desired.start + self.orientation_desired.end) / 2.0
            c_orientation = abs(self.calc_orientation_diff(last_path[-1].orientation, orientation_center))
        else:
            c_orientation = 0

        #velocity difference
        if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
            c_velocity = abs(last_path[-1].velocity - (self.velocity_desired.start + self.velocity_desired.end) / 2.0)
        else:
            c_velocity = 0


        w_time = 0.5
        w_position = 0.5
        w_orientation = 0.1
        w_velocity = 0.4

        cost = w_time * c_time + w_position * c_position + w_orientation * c_orientation + w_velocity * c_velocity

        return cost