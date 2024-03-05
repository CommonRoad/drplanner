from drplanner.diagnoser.drplanner import DrPlanner
from drplanner.utils.config import DiagnoserConfiguration

from commonroad.common.file_reader import CommonRoadFileReader


file_path_root = "../scenarios/"

name_file_motion_primitives = (
    "V_0.0_20.0_Vstep_4.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml"
)

planner_id = "10042153"
# load scenario and planning problem
scenario_id = (
    "DEU_Guetersloh-15_2_T-1"
)
# DEU_Backnang-5_1_T-1/ DEU_Guetersloh-15_2_T-1
file_path = file_path_root + scenario_id + ".xml"
scenario, planning_problem_set = CommonRoadFileReader(file_path).open(True)

config = DiagnoserConfiguration()
config.api_key = "sk-..."

diagnoser = DrPlanner(
    scenario, planning_problem_set, name_file_motion_primitives, planner_id, config
)

result = diagnoser.diagnose_repair()
print(diagnoser.cost_list, min(diagnoser.cost_list), diagnoser.initial_cost)
