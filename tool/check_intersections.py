from commonroad.common.file_reader import CommonRoadFileReader

if __name__ == "__main__":
    file_path_root = "../scenarios/inD_repair/"

    # changable items
    name_file_motion_primitives = (
        "V_0.0_20.0_Vstep_2.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml"
    )
    planner_id = "10042153"
    scenario_id = "DEU_AachenAseag-1_1_T-19"

    file_path = file_path_root + scenario_id + ".xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(True)
    intersections=scenario._lanelet_network._intersections
    if intersections is None:
        print("No intersections found")
    else:
        for intersection_id, intersection in intersections.items():
            print(f"Number of Intersection={intersection_id}")
            print(f"Intersection(intersection_id={intersection.intersection_id})")
            for incomings in intersection.incomings:
                print(f"IntersectionIncomingElement(incoming_id={incomings.incoming_id},")
                print(f"incoming_lanelets={incomings.incoming_lanelets},")
                print(f"successors_right={incomings.successors_right}, successors_straight={incomings.successors_straight}, ")
                print(f"successors_left={incomings.successors_left}, left_of={incomings.left_of})")
            
    