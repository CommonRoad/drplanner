import csv
import glob
import os

from commonroad.common.file_reader import CommonRoadFileReader

from diagnostics.iterative import run_iteration
from drplanner.diagnostics.sampling import DrSamplingPlanner

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.memory.memory import FewShotMemory


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
        FewShotMemory(), scenario, scenario_filepath, planning_problem_set, config, "planner_id"
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


def run_dr_iteration_planner(scenario_filepath: str):
    cost_results, config = run_iteration(scenario_filepath)
    initial_cost_result = cost_results.pop(0)
    best_cost_result = min(cost_results)
    scenario_id = os.path.basename(scenario_filepath)[:-4]
    row = [
        scenario_id,
        config.gpt_version,
        initial_cost_result,
        best_cost_result,
    ]
    for c in cost_results:
        row.append(c)
    filename = config.save_dir + "results.csv"
    results_to_csv(filename, row)


PATH_TO_SCENARIOS = (
    "/home/sebastian/Documents/Uni/Bachelorarbeit/Scenarios/Datasets/Batch"
)
NR_ITERATIONS = 3
MODE = 1

# initialize results.csv
row_data = ["scenario", "gpt-version", "initial_cost", "final_cost"]
for i in range(NR_ITERATIONS):
    row_data.append(f"iter_{i}")
result_filename = DrPlannerConfiguration().save_dir + "results.csv"
result_file = open(result_filename, "w+", newline="")
with result_file:
    w = csv.writer(result_file)
    w.writerow(row_data)

# collect all scenarios
xml_files = glob.glob(os.path.join(PATH_TO_SCENARIOS, "**", "*.xml"), recursive=True)
scenarios = [os.path.abspath(file) for file in xml_files]

for p in scenarios:
    print(p)
    if MODE == 0:
        run_dr_sampling_planner(p)
    elif MODE == 1:
        run_dr_iteration_planner(p)
