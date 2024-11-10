from drplanner.diagnostics.search import DrSearchPlanner
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.memory.vectorStore import PlanningMemory

from commonroad.common.file_reader import CommonRoadFileReader

if __name__ == '__main__':
    file_path_root = "scenarios/"

    #changable items
    name_file_motion_primitives = (
        "V_0.0_20.0_Vstep_2.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml"

    )
    planner_id = "10042153"
    scenario_id = "DEU_Guetersloh-8_1_T-1"

    file_path = file_path_root + scenario_id + ".xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open(True)

    config = DrPlannerConfiguration()
    config.openai_api_key =config.openai_api_key
    reflectin =config.reflection_module
    memory_path = config.memory_path

    # load memory
    agent_memory = PlanningMemory(db_path=memory_path)
    # load updated memory
    if reflectin:      
        updated_memory = PlanningMemory(db_path=memory_path + "_updated")
        updated_memory.combineMemory(agent_memory)
    else:
        updated_memory = None

    #initialize the DrPlanner
    dr_planner = DrSearchPlanner(
        scenario, planning_problem_set, config, name_file_motion_primitives, planner_id, agent_memory, updated_memory
    )

    result = dr_planner.diagnose_repair()
    
    print(dr_planner.cost_list, min(dr_planner.cost_list), dr_planner.initial_cost)
