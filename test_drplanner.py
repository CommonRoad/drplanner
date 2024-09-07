import csv
import glob
import os
import shutil

from commonroad.common.file_reader import CommonRoadFileReader

from drplanner.diagnostics.sampling import DrSamplingPlanner

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.modular_approach.iteration import run_iterative_repair
from drplanner.memory.memory import FewShotMemory


def results_to_csv(filename: str, data: list):
    # opening the csv file in 'a+' mode
    file = open(filename, "a+", newline="")
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerow(data)


def run_dr_sampling_planner(scenario_filepath: str, result_filepath, config):
    scenario, planning_problem_set = CommonRoadFileReader(scenario_filepath).open(True)
    config.save_dir = os.path.dirname(result_filepath)
    dr_planner = DrSamplingPlanner(
        scenario,
        scenario_filepath,
        planning_problem_set,
        config,
        "planner_id",
    )
    dr_planner.diagnose_repair()
    row = [
        str(scenario.scenario_id),
        dr_planner.initial_cost,
        min(dr_planner.cost_list),
    ]
    row.extend(dr_planner.statistic.get_iteration_data())
    results_to_csv(result_filepath, row)


def run_dr_iteration_planner(scenario_filepath: str, result_filepath, config):
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
    row.extend(statistics.get_iteration_data())
    results_to_csv(result_filepath, row)


def run_tests(dataset: str, config: DrPlannerConfiguration, modular: bool):
    path_to_repo = os.path.dirname(os.path.abspath(__file__))
    path_to_scenarios = os.path.join(path_to_repo, "scenarios", dataset)
    experiment_name = f"modular-{config.gpt_version}-{config.temperature}"
    path_to_results = os.path.join(
        config.save_dir,
        experiment_name,
        dataset
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
        memory_size = FewShotMemory().get_size()
        file.write(f"{config.__str__()}memory size: {memory_size}")

    # collect all scenarios
    xml_files = glob.glob(os.path.join(path_to_scenarios, "**", "*.xml"), recursive=True)
    scenarios = [os.path.abspath(file) for file in xml_files]
    for p in scenarios:
        print(p)
        if modular:
            run_dr_iteration_planner(p, result_csv_path, config)
        else:
            run_dr_sampling_planner(p, result_csv_path, config)
    return result_csv_path


standard_config = DrPlannerConfiguration()
standard_save_dir = standard_config.save_dir
standard_config.temperature = 0.6
standard_config.include_plot = False
standard_config.feedback_mode = 0

# e1
standard_config.save_dir = os.path.join(standard_save_dir, "ablation_original_no_plot", "basic")
csv_path = run_tests("large", standard_config, False)
new_path = os.path.join(standard_save_dir, "results", "comparison_gpt4o", "basic.csv")
if not os.path.exists(os.path.dirname(new_path)):
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
shutil.copy(csv_path, new_path)

# e2
standard_config.repair_sampling_parameters = False
standard_config.save_dir = os.path.join(standard_save_dir, "ablation_original_no_plot", "basic_no_sampling")
csv_path = run_tests("large", standard_config, False)
new_path = os.path.join(standard_save_dir, "results", "comparison_gpt4o", "basic_no_sampling.csv")
if not os.path.exists(os.path.dirname(new_path)):
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
shutil.copy(csv_path, new_path)
