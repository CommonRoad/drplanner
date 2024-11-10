import numpy as np

from commonroad_route_planner.reference_path import ReferencePath
from commonroad.scenario.scenario import Scenario

class SceneClassifier:

    def __init__(
            self,
            reference_path:ReferencePath
    ) -> None:
        """
        :param reference_path: reference path
        """
        self.reference_path = reference_path

    def is_straight_line(self, ) -> bool:
        """
        Determines if the path is straight.
        
        """
        #TODO: Determine the value of tolerance
        tolerance: float = 1e-2

        # Calculating Direction Angle Change
        direction_changes = np.diff(self.reference_path.path_orientation)

        # Check that the change in direction angle is within the tolerance range
        return np.all(np.abs(direction_changes) < tolerance)
    
    def is_simple_left_turn(self) -> bool:
        """
        Determines if the path is a simple left turn.
        """
        # Check that the path direction change is predominantly to the left
        direction_changes = np.diff(self.reference_path.path_orientation)
        left_turns = direction_changes[direction_changes > 0]

        # Check that the curvature stays within a certain range
        #TODO: Adjust thresholds as needed
        curvature_threshold = 0.1  
        is_curvature_consistent = np.all(np.abs(self.reference_path.path_curvature) < curvature_threshold)

        # If most of the direction change is to the left and the curvature is consistent, it is considered a simple left turn
        if len(left_turns) > len(direction_changes) / 2 and is_curvature_consistent:
            return True
        return False
    
    def is_simple_right_turn(self) -> bool:
        #TODO: Maybe combine with left turn.
        """
        Determines if the path is a simple right turn.
        """
        # Check that the path direction change is predominantly to the right
        direction_changes = np.diff(self.reference_path.path_orientation)
        right_turns = direction_changes[direction_changes < 0]

        # Check that the curvature stays within a certain range
        #TODO: Adjust thresholds as needed
        curvature_threshold = 0.1  
        is_curvature_consistent = np.all(np.abs(self.reference_path.path_curvature) < curvature_threshold)

        # If most of the direction change is to the right and the curvature is consistent, it is considered a simple right turn
        if len(right_turns) > len(direction_changes) / 2 and is_curvature_consistent:
            return True
        return False

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
        threshold: int = 5
        return len(scenario._dynamic_obstacles) < threshold
