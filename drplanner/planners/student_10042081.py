from SMP.motion_planner.node import PriorityNode
import numpy as np
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
        ########################################################################
        # todo: Implement your own evaluation function here.                   #
        ########################################################################
     
      
        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:

        path_last = node_current.list_paths[-1]
        curr_orientation = path_last[-1].orientation
        if self.is_collision_free(node_current.list_paths[0]):

            if self.reached_goal(node_current.list_paths[-1]):
                return 0.0
            if self.position_desired is None:
                return self.time_desired.start - node_current.list_paths[-1][-1].time_step

            else:
                goal_distance = self.calc_euclidean_distance(node_current)
                orientation_difference = self.calc_orientation_diff(curr_orientation, path_last[-1].orientation)
                time_to_goal = self.calc_time_cost(path_last)
                time_efficiency = self.calc_path_efficiency(path_last)
                velocity = node_current.list_paths[-1][-1].velocity

                if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
                    v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start +
                                   self.planningProblem.goal.state_list[0].velocity.end) / 2

                    dist_vel = abs(path_last[-1].velocity - v_mean_goal)
                else:
                    dist_vel = 0

                goal_distance = self.calc_euclidean_distance(node_current)
                return_value = 0.4 * goal_distance + 1 * time_to_goal + 0.05 * orientation_difference + 0.08 * time_efficiency + 0.1 * dist_vel

                return return_value

        else:
            return np.inf
        
        
       

      
        
            
            
            
            

        
     
            
        
        
        
        
        
  
      
      
