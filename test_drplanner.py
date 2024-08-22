import csv
import glob
import math
import os

from commonroad.common.file_reader import CommonRoadFileReader

from drplanner.diagnostics.sampling import DrSamplingPlanner

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.memory.memory import FewShotMemory
from modular_approach.iteration import run_iterative_repair
from modular_approach.module import EvaluationModule
from planners.reactive_planner import (
    ReactiveMotionPlanner,
    get_basic_configuration_path,
)


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

    cost_results = run_iterative_repair(scenario_filepath)
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


def run_simple_dr_planner(scenario_filepaths: list[str], result_filepath: str):
    memory = FewShotMemory()
    config = DrPlannerConfiguration()
    evaluation_module = EvaluationModule()
    filenames = [
        "DEU_Frankfurt-191_12_I-1.cr",
        "DEU_Frankfurt-11_8_I-1.cr",
        "DEU_Muc-19_1_I-1-1.cr",
        "DEU_Frankfurt-95_9_I-1.cr",
        "ESP_Mad-1_8_I-1-1.cr",
    ]

    for j in range(20):
        scenario_filepath = scenario_filepaths[j]
        scenario_id = os.path.basename(scenario_filepath)[:-4]
        # plot the scenario
        save_to = os.path.join(
            os.path.dirname(config.project_path), "plots", f"{scenario_id}.png"
        )
        config_path = get_basic_configuration_path()
        ReactiveMotionPlanner.create_plot(scenario_filepath, config_path, save_to)
        # retrieve template
        cf_string, relative = memory.retrieve_with_image(save_to)
        cf_string = cf_string[0]
        relative = int(relative[0][4:])
        # evaluate
        motion_planner = ReactiveMotionPlanner(cf_string)
        print("start eval")
        _, total_cost = evaluation_module.run(scenario_filepath, motion_planner)
        print(total_cost)
        # write to csv
        row = [scenario_id, total_cost, filenames[relative]]
        results_to_csv(result_filepath, row)


def run_brute_force_dr_planner(scenario_filepaths: list[str], result_filepath: str):
    config = DrPlannerConfiguration()
    evaluation_module = EvaluationModule()
    filenames = [
        "DEU_Frankfurt-191_12_I-1.cr.txt",
        "DEU_Frankfurt-11_8_I-1.cr.txt",
        "DEU_Muc-19_1_I-1-1.cr.txt",
        "DEU_Frankfurt-95_9_I-1.cr.txt",
        "ESP_Mad-1_8_I-1-1.cr.txt",
    ]

    templates = []
    for name in filenames:
        path_to_cf = os.path.join(config.project_path, "memory", "plots", name)
        with open(path_to_cf, "r") as file:
            cf_string = file.read()
        templates.append(cf_string)

    for j in range(20):
        scenario_filepath = scenario_filepaths[j]
        row = [scenario_filepath]
        best = math.inf
        best_scenario = ""
        for index in range(len(filenames)):
            template_cf = templates[index]
            template_scenario = filenames[index]
            motion_planner = ReactiveMotionPlanner(template_cf)
            _, total_cost = evaluation_module.run(scenario_filepath, motion_planner)
            print(total_cost)
            row.append(total_cost)
            if best > total_cost:
                best = total_cost
                best_scenario = template_scenario

        row.append(best)
        row.append(best_scenario)
        results_to_csv(result_filepath, row)


PATH_TO_SCENARIOS = (
    "/home/sebastian/Documents/Uni/Bachelorarbeit/Scenarios/Datasets/Batch_Feasible"
)
SAVE_DIR_NAME = "TEST"
COMMIT_HASH = ""
NR_ITERATIONS = 0
MODE = 2

# file1 = os.path.join(
#     DrPlannerConfiguration().save_dir,
#     SAVE_DIR_NAME,
#     "feasible",
#     f"results-feasible-.csv",
# )
# file2 = os.path.join(
#     DrPlannerConfiguration().save_dir,
#     SAVE_DIR_NAME,
#     "feasible",
#     f"results-feasible-brute_force.csv",
# )
# file3 = os.path.join(
#     DrPlannerConfiguration().save_dir,
#     SAVE_DIR_NAME,
#     "feasible",
#     f"results.csv",
# )
# with open(file1, mode='r', newline='') as infile:
#     reader = csv.reader(infile)
#     rows1 = list(reader)
#     head1 = rows1[0]
#     rows1 = rows1[1:]
#
# with open(file2, mode='r', newline='') as infile:
#     reader = csv.reader(infile)
#     rows2 = list(reader)
#     head2 = rows2[0]
#     rows2 = rows2[1:]
#
# with open(file3, mode='r', newline='') as infile:
#     reader = csv.reader(infile)
#     rows3 = list(reader)
#     head3 = rows3[0]
#     rows3 = rows3[1:]
#
# best1 = []
# best2 = []
# best3 = []
#
# for row in rows1:
#     best1.append(row[1])
# for row in rows3:
#     best3.append(row[3])
# for row in rows2:
#     costs = row[1:-1]
#     best2.append(min([float(x) for x in costs]))
#
# counter = 0
# counter2 = 0
# counter3 = 0
# n = min(len(best1), len(best2), len(best3))
# for i in range(n):
#     a = float(best1[i])
#     b = float(best2[i])
#     c = float(best3[i])
#     diff = a - b
#     diff2 = c - b
#     if diff == 0.0:
#         counter += 1
#     if diff2 >= 0:
#         counter2 += 1
#     if diff2 < math.inf:
#         counter3 += diff2
#     print(f"a: {a}, b: {b}, c: {c} diff_ab: {diff} diff_cb: {diff2}")
# print(f"number of samples: {n}")
# print(f"number of correct guesses: {counter}")
# print(f"number of better answers through naive approach: {counter2}")
# print(f"average improvement: {counter3/n}")
#
# exit(0)
# initialize results.csv
row_data = ["scenario", "best", "template"]
# row_data = ["scenario", "a", "b", "c", "d", "e", "best_score", "best_template"]

for i in range(NR_ITERATIONS):
    row_data.append(f"iter_{i}")
data_set = PATH_TO_SCENARIOS.split("/")[-1].lower()
result_filename = os.path.join(
    DrPlannerConfiguration().save_dir,
    SAVE_DIR_NAME,
    data_set,
    f"results-{data_set}-{COMMIT_HASH}.csv",
)

print(f"saving results at {result_filename}")

if not os.path.exists(os.path.dirname(result_filename)):
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
result_file = open(result_filename, "w+", newline="")
with result_file:
    w = csv.writer(result_file)
    w.writerow(row_data)

# collect all scenarios
xml_files = glob.glob(os.path.join(PATH_TO_SCENARIOS, "**", "*.xml"), recursive=True)
scenarios = [os.path.abspath(file) for file in xml_files]

if MODE > 1:
    if MODE == 2:
        run_simple_dr_planner(scenarios, result_filename)
    elif MODE == 3:
        run_brute_force_dr_planner(scenarios, result_filename)
else:
    for p in scenarios:
        print(p)
        if MODE == 0:
            run_dr_sampling_planner(p, result_filename)
        elif MODE == 1:
            run_dr_iteration_planner(p, result_filename)
