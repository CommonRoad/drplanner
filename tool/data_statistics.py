import os
import glob
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.reference_path_planner import ReferencePathPlanner
from drplanner.sce_classification.sce_classification import SceneClassifier

def data_statistics(scenarios_dir):
    data = {
    "max_curvature": [],
    "mean_curvature": [],
    "std_curvature": [],
    "left_turn_sum": [],
    "left_turn_sum_degree": []
    }
    # Iterate through the XML files in the scenarios directory
    for xml_file in glob.glob(os.path.join(scenarios_dir, "*.xml")):

        # Reading scenarios and planning problem sets
        scenario, planning_problem_set = CommonRoadFileReader(xml_file).open()
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

        route_planner = RoutePlanner(lanelet_network=scenario.lanelet_network,  planning_problem=planning_problem,scenario=scenario)
        routes= route_planner.plan_routes()

        # Instantiate reference path planner and plan reference path
        ref_path_planner: ReferencePathPlanner = ReferencePathPlanner(
            lanelet_network=scenario.lanelet_network,
            planning_problem=planning_problem,
            routes=routes,
        )

        reference_path = ref_path_planner.plan_shortest_reference_path(
            retrieve_shortest=True, consider_least_lance_changes=True
        )
     
        sceneClassifier=SceneClassifier(reference_path,scenario)
        
        max_curvature, mean_curvature, std_curvature, left_turn_sum, left_turn_sum_degree = sceneClassifier.left_turn_data()
        data["max_curvature"].append(max_curvature)
        data["mean_curvature"].append(mean_curvature)
        data["std_curvature"].append(std_curvature)
        data["left_turn_sum"].append(left_turn_sum)
        data["left_turn_sum_degree"].append(left_turn_sum_degree)
    max_curvature_all = max(data["max_curvature"])
    mean_curvature_all = np.mean(data["mean_curvature"])
    std_curvature_all = np.std(data["std_curvature"])
    max_left_turn_sum = max(data["left_turn_sum"])
    min_left_turn_sum = min(data["left_turn_sum"])
    max_left_turn_sum_degree = max(data["left_turn_sum_degree"])
    min_left_turn_sum_degree = min(data["left_turn_sum_degree"])

    # Print results
    print("maximum curvature values:", max_curvature_all)
    print("average curvature value:", mean_curvature_all)
    print("standard deviation of curvatures:", std_curvature_all)
    print("maximum left turn sum:", max_left_turn_sum)
    print("minimum left turn sum:", min_left_turn_sum)
    print("maximum left turn sum degree:", max_left_turn_sum_degree)
    print("minimum left turn sum degree:", min_left_turn_sum_degree)
        


scenarios_dir = "../scenarios/left_turn"
data_statistics(scenarios_dir)