import numpy
import math

from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch


class StudentMotionPlanner(GreedyBestFirstSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)

    def evaluation_function(self, node_current: PriorityNode) -> float:
        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:
        last_path = node_current.list_paths[-1]

        distStartState = self.calc_heuristic_distance(last_path[0])
        distLastState = self.calc_heuristic_distance(last_path[-1])

        if self.reached_goal(last_path):
            return 0.0

        if distLastState is None:
            print(32)
            return numpy.inf

        if distStartState < distLastState:
            return self.calc_euclidean_distance(current_node=node_current)

        cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(last_path)

        if cost_lanelet is None or final_lanelet_id[0] is None:
            if hasattr(self.planningProblem.goal.state_list[0], 'time_step'):
                return abs(last_path[-1].time_step - self.planningProblem.goal.state_list[0].time_step.start)
            return numpy.inf

        # Fastet path and closest to the destination
        velocity = last_path[-1].velocity

        if numpy.isclose(velocity, 0):
            return numpy.inf
        else:
            try:
                # A: time to reach goal
                time_diff = self.calc_euclidean_distance(current_node=node_current) / velocity
                # normalize: time to goal / (time until now + time to goal)
                now_time = self.calc_time_cost(last_path)
                time_diff = time_diff / ( now_time + time_diff )

                # B: orientational diff to goal
                angle = self.calc_angle_to_goal(last_path[-1])
                orientation_diff = abs(self.calc_orientation_diff(angle, last_path[-1].orientation))
                # normalize: orientation diff / 180°
                orientation_diff = orientation_diff / math.pi

                # C: velocity difference to goal
                if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
                    v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start +
                           self.planningProblem.goal.state_list[0].velocity.end) / 2
                    vel_diff = abs(last_path[-1].velocity - v_mean_goal)
                else:
                    vel_diff = 0

                factor = 10
                weights = numpy.zeros(3)
                weights[0] = .49  # A: time to reach goal
                weights[1] = .5   # B: orientation diff to goal
                weights[2] = .01  # C: velocity difference
                cost = weights[0] * time_diff + \
                       weights[1] * orientation_diff  + \
                       weights[2] * vel_diff

                return cost * factor
            except Exception as error:
                print(error)
                return distLastState