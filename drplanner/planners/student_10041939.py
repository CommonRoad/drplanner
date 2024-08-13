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
        # copied the implementation in AStarSearch

        """
        Evaluation function of A* is f(n) = g(n) + h(n)
        """
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

        last_path = node_current.list_paths[-1]
        if self.reached_goal(last_path):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - last_path[-1].time_step

        else:
            velocity = last_path[-1].velocity

            if np.isclose(velocity, 0):
                return np.inf

        cost_lanelet, end_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(
            last_path
        )
        if cost_lanelet is None or end_lanelet_id is None or start_lanelet_id is None:
            return np.inf
        goal_state = self.planningProblem._goal_region.state_list[0]

        # • positional distance: the positional distance (in Euclidean, Manhattan or other distances) between
        #   the (x, y) position of a given state and the goal region (e.g. center of the goal region).
        positional_distance = (
            self.calc_euclidean_distance(current_node=node_current) / velocity
        )
        # print(positional_distance)
        # • velocity difference: the velocity difference between the velocity of a given state and the goal state
        #   (e.g. center of the desired velocity interval).
        if hasattr(goal_state, "velocity"):
            velocity_difference = abs(
                velocity - (goal_state.velocity.start + goal_state.velocity.end) / 2
            )
        else:
            velocity_difference = 0.0

        # • orientation difference: the orientation difference between the orientation of a given state and the
        #   goal state (e.g. center of the desired orientation interval).
        orientation_difference = self.calc_orientation_diff(
            self.calc_angle_to_goal(last_path[-1]), last_path[-1].orientation
        )
        if positional_distance <= 10.0:  # more important when staying closer
            orientation_difference *= 2

        # • time difference: the time difference between the time step of a given state and the goal state (e.g.
        #   center of the desired time step interval).
        # time_difference = abs(last_path[-1].time_step - (goal_state.time_step.end - goal_state.time_step.start) / 2)

        #   Besides these state components, one might also want to consider some other factors such as:
        # • lanelet id: we can retrieve the id of the lanelet on which the state is located. By this we can determine
        #   whether the examined state is located on the lanelet of the goal state, and reward such states.
        if end_lanelet_id is not None:
            goal_in_lane = -float(self.is_goal_in_lane(end_lanelet_id[-1]))
        else:
            goal_in_lane = 0.0
        # • obstacles on lanelet: contrary to the previous metric, if there are obstacles located on the lanelet
        #   of the goal state, we might want to make a lane change, thus penalizing such states.
        if start_lanelet_id is not None:
            print(last_path[-1].time_step, start_lanelet_id[-1])
            obstacles_on_lanelet = self.num_obstacles_in_lanelet_at_time_step(
                last_path[-1].time_step, start_lanelet_id[-1]
            )
        else:
            obstacles_on_lanelet = 0
        # • trajectory efficiency: we can calculate the ratio of the length of the trajectory traveled so far to the
        #   time required to travel such a trajectory as trajectory efficiency, and reward those trajectories with
        #   higher efficiencies.
        trajectory_efficiency = self.calc_path_efficiency(last_path)
        if trajectory_efficiency == np.inf:
            trajectory_efficiency = 0
        metrics = np.array(
            [
                positional_distance,
                velocity_difference,
                orientation_difference,
                # time_difference,
                goal_in_lane,
                obstacles_on_lanelet,
                trajectory_efficiency,
                cost_lanelet,
            ]
        )
        weights = np.array(
            [
                2.8,
                2.0,
                0.5,
                # 0.0,
                1.0,
                0.1,
                -2.0,
                0.1,
            ]
        )
        cost = metrics @ weights
        # print(metrics)
        # print(cost, positional_distance)
        # print('lp:', last_path[-1].position, 'gp:', goal_state.position.center)
        # if cost < 0:
        #     print(metrics)
        return 0 if cost < 0 else cost
