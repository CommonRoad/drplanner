import csv
import glob
import os

from commonroad.common.file_reader import CommonRoadFileReader
from drplanner.diagnostics.sampling import DrSamplingPlanner

from drplanner.utils.config import DrPlannerConfiguration


def results_to_csv(filename: str, data: list):
    # opening the csv file in 'a+' mode
    file = open(filename, "a+", newline="")
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerow(data)


def run_dr_sampling_planner(scenario_filepath: str):
    scenario, planning_problem_set = CommonRoadFileReader(scenario_filepath).open(True)
    config = DrPlannerConfiguration()
    dr_planner = DrSamplingPlanner(
        scenario, scenario_filepath, planning_problem_set, config, "planner_id"
    )
    dr_planner.diagnose_repair()
    filename = config.save_dir + "results.csv"

    if len(dr_planner.cost_list) <= 0:
        print("This scenario needs no repairs!")
        row = [
            str(scenario.scenario_id),
            str(config.gpt_version),
            dr_planner.initial_cost,
        ]
        results_to_csv(filename, row)

    row = [
        str(scenario.scenario_id),
        str(config.gpt_version),
        dr_planner.initial_cost,
        min(dr_planner.cost_list),
    ]
    for c in dr_planner.cost_list:
        row.append(c)
    results_to_csv(filename, row)
    print(dr_planner.cost_list, min(dr_planner.cost_list), dr_planner.initial_cost)


drplanner_config = DrPlannerConfiguration()

# initialize results.csv
row_data = ["scenario", "gpt-version", "initial_cost", "final_cost"]
for i in range(8):
    row_data.append(f"iter_{i}")
result_filename = drplanner_config.save_dir + "results.csv"
result_file = open(result_filename, "w+", newline="")
with result_file:
    w = csv.writer(result_file)
    w.writerow(row_data)

# collect all scenarios
scenarios_folder = (
    "/home/sebastian/Documents/Uni/Bachelorarbeit/Scenarios/Datasets/Feasible"
)
xml_files = glob.glob(os.path.join(scenarios_folder, "**", "*.xml"), recursive=True)
scenarios = [os.path.abspath(file) for file in xml_files]

for p in scenarios:
    print(p)
    run_dr_sampling_planner(p)
