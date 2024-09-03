import csv
import glob
import os

from commonroad.common.file_reader import CommonRoadFileReader

from drplanner.diagnostics.sampling import DrSamplingPlanner

from drplanner.utils.config import DrPlannerConfiguration
from modular_approach.iteration import run_iterative_repair


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


def run_dr_iteration_planner(scenario_filepath: str, result_filepath):
    config = DrPlannerConfiguration()
    config.save_dir = os.path.dirname(result_filepath)

    cost_results, statistics = run_iterative_repair(scenario_path=scenario_filepath, config=config)
    cost_results = [x.total_costs for x in cost_results]
    initial_cost_result = cost_results.pop(0)
    best_cost_result = min(cost_results)
    scenario_id = os.path.basename(scenario_filepath)[:-4]
    row = [
        scenario_id,
        initial_cost_result,
        best_cost_result,
    ]
    for c in cost_results:
        row.append(c)
    results_to_csv(result_filepath, row)


def test_modular_version():
    config = DrPlannerConfiguration()
    path_to_repo = os.path.dirname(os.path.abspath(__file__))
    path_to_scenarios = str(os.path.join(path_to_repo, "scenarios", config.dataset))
    experiment_name = f"modular-{config.gpt_version}-{config.temperature}"
    path_to_results = os.path.join(
        DrPlannerConfiguration().save_dir,
        experiment_name,
        config.dataset
    )
    result_csv_path = os.path.join(path_to_results, "results.csv")
    result_config_path = os.path.join(path_to_results, "config.txt")
    index_row = [
        "scenario_id",
        "initial",
        "best",
        "duration",
        "token_count",
        "missing parameters",
        "flawed helper methods",
        "missing few-shots",
        "added few-shots"
    ]

    for i in range(config.iteration_max):
        index_row.append(f"iter_{i}")

    if not os.path.exists(os.path.dirname(result_csv_path)):
        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(result_config_path)):
        os.makedirs(os.path.dirname(result_config_path), exist_ok=True)

    result_csv_file = open(result_csv_path, "w+", newline="")
    with result_csv_file:
        w = csv.writer(result_csv_file)
        w.writerow(index_row)

    with open(result_config_path, "w+") as file:
        file.write(config.__str__())

    # collect all scenarios
    xml_files = glob.glob(os.path.join(path_to_scenarios, "**", "*.xml"), recursive=True)
    scenarios = [os.path.abspath(file) for file in xml_files]
    for p in scenarios:
        print(p)
        run_dr_iteration_planner(p, result_csv_path)


test_modular_version()
