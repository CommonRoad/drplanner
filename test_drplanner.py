import csv
import glob
import os
import shutil

from commonroad.common.file_reader import CommonRoadFileReader

from drplanner.diagnoser.sampling import DrSamplingPlanner

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.modular.iteration import run_iterative_repair
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
        planning_problem_set,
        config,
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

    cost_results, statistics = run_iterative_repair(
        scenario_path=scenario_filepath, config=config
    )
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


def run_tests(dataset: str, config: DrPlannerConfiguration, modular: bool, skip=None):
    if not skip:
        skip = []
    path_to_repo = os.path.dirname(os.path.abspath(__file__))
    path_to_scenarios = os.path.join(path_to_repo, "scenarios", dataset)
    experiment_name = f"modular-{config.gpt_version}-{config.temperature}"
    path_to_results = os.path.join(config.save_dir, experiment_name, dataset)
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
        "added few-shots",
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
    xml_files = glob.glob(
        os.path.join(path_to_scenarios, "**", "*.xml"), recursive=True
    )
    scenarios = [os.path.abspath(file) for file in xml_files]
    for p in scenarios:
        test_for_skip = [not (x in p) for x in skip]
        if all(test_for_skip):
            print(p)
        else:
            print(f"Skipping {p}")
        if modular:
            run_dr_iteration_planner(p, result_csv_path, config)
        else:
            run_dr_sampling_planner(p, result_csv_path, config)
    return result_csv_path


standard_config = DrPlannerConfiguration()
standard_save_dir = standard_config.save_dir
standard_config.temperature = 0.6
standard_config.include_plot = False
standard_config.repair_sampling_parameters = True
standard_config.iteration_max = 3
standard_config.reflection_module = True

standard_config.update_memory_module = False
standard_config.memory_module = True
standard_config.include_cost_function_few_shot = True
# test

skip = [
    # "ESP_Mad-1_6_I-1-1.cr",
    # "DEU_Frankfurt-191_12_I-1.cr",
    # "DEU_Frankfurt-95_9_I-1.cr",
    # "ESP_Mad-1_8_I-1-1.cr",
    # "DEU_Frankfurt-65_2_I-1.cr",
    # "DEU_Frankfurt-11_10_I-1.cr",
    # "DEU_Frankfurt-11_3_I-1.cr",
    # "CHN_Sha-4_1_I-1-1.cr",
    # "DEU_Frankfurt-65_7_I-1.cr",
    # "DEU_Frankfurt-11_12_I-1.cr",
    # "ESP_Mad-1_7_I-1-1.cr",
    # "DEU_Frankfurt-147_6_I-1.cr",
    # "DEU_Frankfurt-11_2_I-1.cr",
    # "CHN_Sha-16_1_I-1-1.cr",
    # "CHN_Cho-1_1_I-1-1.cr",
]

standard_config.reflection_module = False
standard_config.save_dir = os.path.join(
    standard_save_dir, "performance", "memory", "memory_10_09"
)
csv_path = run_tests("test", standard_config, True)
new_path = os.path.join(standard_save_dir, "results", "memory", "memory_10_09.csv")
if not os.path.exists(os.path.dirname(new_path)):
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
shutil.copy(csv_path, new_path)

# standard_config.reflection_module = False
# standard_config.save_dir = os.path.join(standard_save_dir, "performance", "memory", "memory")
# csv_path = run_tests("test", standard_config, True)
# new_path = os.path.join(standard_save_dir, "results", "memory", "memory.csv")
# if not os.path.exists(os.path.dirname(new_path)):
#     os.makedirs(os.path.dirname(new_path), exist_ok=True)
# shutil.copy(csv_path, new_path)
#
# standard_config.reflection_module = True
# standard_config.save_dir = os.path.join(standard_save_dir, "performance", "memory", "memory_reflection2")
# csv_path = run_tests("test", standard_config, True)
# new_path = os.path.join(standard_save_dir, "results", "memory", "memory_reflection2.csv")
# if not os.path.exists(os.path.dirname(new_path)):
#     os.makedirs(os.path.dirname(new_path), exist_ok=True)
# shutil.copy(csv_path, new_path)
#
# standard_config.reflection_module = False
# standard_config.save_dir = os.path.join(standard_save_dir, "performance", "memory", "memory2")
# csv_path = run_tests("test", standard_config, True)
# new_path = os.path.join(standard_save_dir, "results", "memory", "memory2.csv")
# if not os.path.exists(os.path.dirname(new_path)):
#     os.makedirs(os.path.dirname(new_path), exist_ok=True)
# shutil.copy(csv_path, new_path)
