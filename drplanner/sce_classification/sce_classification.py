import numpy as np

from commonroad_route_planner.reference_path import ReferencePath
from commonroad.scenario.scenario import Scenario

class SceneClassifier:

    def __init__(
            self,
            reference_path:ReferencePath,
            scenario:Scenario
    ) -> None:
        """
        :param reference_path: reference path
        """
        self.reference_path = reference_path
        self.scenario = scenario

    def classify_scene(self):
        """
        Classifies the scene based on the path characteristics and prints the type of scene.
        """
        if self.is_straight_line():
            print("The scene is a straight line.")
        else:
            turn_type = self.classify_turn()
            if turn_type == 'left':
                print("The scene is a simple left turn.")
            elif turn_type == 'right':
                print("The scene is a simple right turn.")
            else:
                print("The scene is a complex path or no turn.")

        if self.is_lane_change():
            print("The scene involves a lane change.")

        if self.is_long_curved_section():
            print("The scene is a long curved section.")

        if self.has_few_dynamic_obstacles(self.scenario):
            print("The scene has few dynamic obstacles.")


    def is_straight_line(self) -> bool:
        """
        Determines if the path is straight.
        
        """
        #TODO: Determine the value of tolerance
        tolerance: float = 1e-2

        # Calculating Direction Angle Change
        direction_changes = np.diff(self.reference_path.path_orientation)

        # Check that the change in direction angle is within the tolerance range
        return np.all(np.abs(direction_changes) < tolerance)
    
    def classify_turn(self) -> str:
        """
        Determines if the path is a simple left turn, right turn, or no turn.

        """
        curvature_threshold: float = 0.1
        std_threshold: float = 0.05
        angle_threshold: float =np.pi/2

        # Check that the path direction change is predominantly to the specified direction
        direction_changes = np.diff(self.reference_path.path_orientation)
        left_turns = direction_changes[direction_changes > 0]
        right_turns = direction_changes[direction_changes < 0]

        # Check that the curvature stays within a certain range
        is_curvature_consistent = np.all(np.abs(self.reference_path.path_curvature) < curvature_threshold)
        #TODO: Decide on a metric
        # Check the standard deviation of the curvature
        is_curvature_std_small = np.std(self.reference_path.path_curvature) < std_threshold

        # Check if the sum of left turns is around 90 degrees (in radians)
        left_turn_sum = np.sum(left_turns)
        is_left_turn_sum_near_90 = np.abs(np.degrees(left_turn_sum) - angle_threshold) < 10  # Allow a tolerance of 10 degrees

        # Check if the sum of right turns is around 90 degrees (in radians)
        right_turn_sum = np.sum(right_turns)
        is_right_turn_sum_near_90 = np.abs(np.degrees(right_turn_sum) - angle_threshold) < 10  # Allow a tolerance of 10 degrees

        # Determine the type of turn
        if len(left_turns) > len(direction_changes) / 2 and is_curvature_consistent and is_curvature_std_small and is_left_turn_sum_near_90:
            return 'left'
        elif len(right_turns) > len(direction_changes) / 2 and is_curvature_consistent and is_curvature_std_small and is_right_turn_sum_near_90:
            return 'right'
        else:
            return 'no_turn'

    def is_lane_change(self) -> bool:
        """
        Determine if the path is a lane change scenario.
        """
        # Check the number of lane changes in the path
        lane_change_threshold = 0 
        return self.reference_path.num_lane_change_actions > lane_change_threshold

    def is_long_curved_section(self) -> bool:
        """
        Determine if the path is a long curved section.
        
        """
        curvature_threshold: float = 0.1
        length_threshold: float = 50.0

        # Checking the curvature and length of paths
        is_curvature_high = np.any(np.abs(self.reference_path.path_curvature) > curvature_threshold)
        is_length_long = self.reference_path.length_reference_path > length_threshold

        # If the curvature is high and the length is long, it is considered to be a long curve section
        return is_curvature_high and is_length_long

    def has_few_dynamic_obstacles(self, scenario:Scenario) -> bool:
        """
        Determine if the number of dynamic obstacles in the scene is low.
        """
        #TODO:Combined with straight routes or?
        threshold: int = 5
        return len(scenario._dynamic_obstacles) < threshold


if __name__ == '__main__':
    pass