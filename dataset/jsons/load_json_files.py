import json

# Replace 'your_file.json' with the path to your JSON file
filename = "10041944.json"

# Read the JSON file and parse its contents
with open(filename, "r") as file:
    data = json.load(file)

# Now 'data' is a Python object (dict or list) containing the data from the JSON file
# You can use it directly in your script
# For example, print the data
print(data["input"]["heuristic_function"])
print(data["output"]["improved_heuristic_function"])
# Example of accessing a specific part of the data
# This depends on the structure of your JSON, so adjust accordingly
# Example: print(data['key']) if your JSON is a dictionary

code = """
"\n    def heuristic_function(self, node_current: PriorityNode) -> float:\n\n        recent_path = node_current.list_paths[-1]\n        recent_node = recent_path[-1]\n        if self.reached_goal(recent_path):\n            return 0.0\n\n        if self.position_desired is None:\n            return self.time_desired.start -recent_node.time_step\n\n        velocity = recent_node.velocity\n        distance = self.calc_euclidean_distance(current_node=node_current)\n\n        if np.isclose(velocity, 0) or             self.calc_heuristic_distance(recent_node) is None or                 np.isclose(self.calc_path_efficiency(recent_path), 0):\n            return math.inf\n\n        acceleration_cost = self.calc_acceleration_cost(recent_path)\n        steering_cost = self.calc_steering_cost(recent_path)\n        path_efficiency = self.calc_path_efficiency(recent_path)\n\n        angle_to_goal = self.calc_angle_to_goal(recent_node)\n        change_in_angel = self.calc_orientation_diff(angle_to_goal, recent_node.orientation)\n        so_far_length = self.calc_travelled_distance(recent_path)\n        \n        cost = (100 - so_far_length) + 2 * distance + abs(change_in_angel) +\n                acceleration_cost + steering_cost + path_efficiency\n\n        return max(cost, 0)\n"""
print(code)
