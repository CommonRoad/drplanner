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


def run_dr_sampling_planner(scenario_filepath: str, result_filepath: str):
    scenario, planning_problem_set = CommonRoadFileReader(scenario_filepath).open(True)
    config = DrPlannerConfiguration()
    dr_planner = DrSamplingPlanner(
        FewShotMemory(),
        scenario,
        scenario_filepath,
        planning_problem_set,
        config,
        "planner_id",
    )
    dr_planner.diagnose_repair()

    if len(dr_planner.cost_list) <= 0:
        print("This scenario needs no repairs!")
        row = [
            str(scenario.scenario_id),
            str(config.gpt_version),
            dr_planner.initial_cost,
        ]
        results_to_csv(result_filepath, row)

    row = [
        str(scenario.scenario_id),
        str(config.gpt_version),
        dr_planner.initial_cost,
        min(dr_planner.cost_list),
    ]
    for c in dr_planner.cost_list:
        row.append(c)
    results_to_csv(result_filepath, row)
    print(dr_planner.cost_list, min(dr_planner.cost_list), dr_planner.initial_cost)


def run_dr_iteration_planner(scenario_filepath: str, result_filepath: str):
    config = DrPlannerConfiguration()
    config.save_dir = os.path.dirname(result_filepath)
    cost_results = run_iteration(scenario_filepath, config=config)
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
    results_to_csv(result_filepath, row)


PATH_TO_SCENARIOS = (
    "/home/sebastian/Documents/Uni/Bachelorarbeit/Scenarios/Datasets/***"
)
SAVE_DIR_NAME = "IterationV02"
COMMIT_HASH = "1f0fcff1030c7c4ebde1ec022b350690ba472b9e"
NR_ITERATIONS = 3
MODE = 1

# initialize results.csv
row_data = ["scenario", "gpt-version", "initial_cost", "final_cost"]
for i in range(NR_ITERATIONS):
    row_data.append(f"iter_{i}")
data_set = PATH_TO_SCENARIOS.split("/")[-1].lower()
result_filename = os.path.join(
    DrPlannerConfiguration().save_dir,
    SAVE_DIR_NAME,
    data_set,
    f"results-{data_set}-{COMMIT_HASH}.csv",
)
if not os.path.exists(os.path.dirname(result_filename)):
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
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
        run_dr_sampling_planner(p, result_filename)
    elif MODE == 1:
        run_dr_iteration_planner(p, result_filename)
