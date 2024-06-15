import csv
import json
import math
import os

import Levenshtein

from drplanner.utils.config import DrPlannerConfiguration

config = DrPlannerConfiguration()


class DrPlannerResult:
    def __init__(self, row: list):
        self.scenario_id = row[0]
        self.gpt_version = row[1]
        self.initial_cost = float(row[2])
        if len(row) > 3:
            self.final_cost = float(row[3])
        else:
            self.final_cost = self.initial_cost
        self.iterations = []
        self.cost_functions = []
        if len(row) > 4:
            for i in range(4, len(row)):
                self.iterations.append(float(row[i]))
            jsons_folder = (
                os.path.join(config.save_dir, self.scenario_id)
                + "/gpt-3.5-turbo/jsons/"
            )
            for i in range(len(row) - 4):
                iteration = load_json_file(jsons_folder + f"result_iter-{i}.json")
                if "improved_cost_function" in iteration.keys():
                    self.cost_functions.append(iteration["improved_cost_function"])
                else:
                    self.cost_functions.append(None)


def load_json_file(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def parse_results():
    filename = config.save_dir + "results.csv"
    planner_results = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            planner_results.append(DrPlannerResult(row))
    return planner_results


def absolute_cost_statistic(drplanner_results: list):
    # This only includes runs where the initial planner was successful
    absolute_cost_reduction = [
        r.initial_cost - r.final_cost
        for r in drplanner_results
        if not math.isinf(r.initial_cost) and not math.isinf(r.final_cost)
    ]
    print(f"maximal absolute cost reduction: {max(absolute_cost_reduction)}")
    print(f"minimal absolute cost reduction: {min(absolute_cost_reduction)}")
    average_cost_reduction = sum(absolute_cost_reduction) / len(absolute_cost_reduction)
    print(f"average absolute cost reduction: {average_cost_reduction}")


def relative_cost_statistic(drplanner_results: list):
    # This only includes runs where the initial planner was successful
    relative_cost_reduction = [
        ((r.initial_cost - r.final_cost) / r.initial_cost) * 100
        for r in drplanner_results
        if not math.isinf(r.initial_cost) and not math.isinf(r.final_cost)
    ]
    print(f"maximal relative cost reduction: {max(relative_cost_reduction):.2f}%")
    print(f"minimal relative cost reduction: {min(relative_cost_reduction):.2f}%")
    average_cost_reduction = sum(relative_cost_reduction) / len(relative_cost_reduction)
    print(f"average relative cost reduction: {average_cost_reduction:.2f}%")


def failure_statistics(drplanner_results: list):
    fixed = [
        r
        for r in drplanner_results
        if math.isinf(r.initial_cost) and not math.isinf(r.final_cost)
    ]
    not_fixed = [
        r
        for r in drplanner_results
        if math.isinf(r.initial_cost) and math.isinf(r.final_cost)
    ]
    failed = len(fixed) + len(not_fixed)
    fixed = len(fixed) / failed * 100
    print(
        f"Of {failed} scenarios where the initial planner did not find a trajectory, DrPlanner fixed {fixed:.2f}%"
    )
    # find runs where DrPlanner only resulted in minimal improvements
    threshold = 5.0
    stagnated = [
        r
        for r in drplanner_results
        if (
            abs(r.initial_cost - r.final_cost) < threshold
            or r.initial_cost - r.final_cost < 0
        )
        and not math.isinf(r.initial_cost)
    ]
    not_failed = len(drplanner_results) - failed
    not_fixed = len(stagnated) / not_failed * 100
    print(
        f"Of {not_failed} scenarios where the initial planner found a trajectory, DrPlanner did not improve {not_fixed:.2f}%"
    )


def exception_statistics(drplanner_results: list):
    fails_per_run = []
    for r in drplanner_results:
        num_fails = 0
        for iteration in r.iterations:
            if math.isinf(iteration):
                num_fails += 1
        fails_per_run.append(num_fails)
    amount_of_fails = [0, 1, 2, 3, 4, 5]
    for amount in amount_of_fails:
        specific_runs = [x for x in fails_per_run if x == amount]
        percent = len(specific_runs) / len(fails_per_run) * 100
        print(f"In {percent:.2f}% of runs, there occurred exactly {amount} exceptions")
    # combine lists of iterations into one list
    iterations = [r.iterations for r in drplanner_results]
    iterations = [item for sublist in iterations for item in sublist]
    failed = [i for i in iterations if math.isinf(i)]
    avg_failures = len(failed) / len(iterations) * 100
    print(
        f"Some sort of exception occurred in {avg_failures:.2f}% of all combined iterations"
    )


def content_statistics(drplanner_results: list):
    # evaluate how many unique trajectories were generated each run
    num_exceptions = 0
    unique_results = []
    for r in drplanner_results:
        result_set = set()
        for i in r.iterations:
            if math.isinf(i):
                num_exceptions += 1
            else:
                result_set.add(i)
        percent = len(result_set) / len(r.iterations) * 100
        unique_results.append(percent)

    avg_uniqueness = sum(unique_results) / len(unique_results)
    print(
        f"Per run an average {avg_uniqueness:.2f}% of successful iterations produced unique results"
    )

    # first extract original cost function:
    original = "def evaluate(self, trajectory: TrajectorySample) -> float:\n    cost = 0.0\n    cost += self.acceleration_costs(trajectory)\n   cost += self.desired_velocity_costs(trajectory)\n   cost += self.desired_path_length_costs(trajectory)\n    cost += self.distance_to_reference_path_costs(trajectory)\n    cost += self.orientation_offset_costs(trajectory)\n    return cost\n"

    cost_function_array = [x.cost_functions for x in drplanner_results]
    temp = set()
    [temp.add(len(c)) for c in cost_function_array]
    assert len(temp) == 1

    # evaluate how many times the cost function was not included in the llm response
    flattened_cost_functions = [
        item for sublist in cost_function_array for item in sublist
    ]
    parameter_error = [c for c in flattened_cost_functions if not c]
    percent = len(parameter_error) / num_exceptions * 100
    print(
        f"Of {num_exceptions} exceptions, {len(parameter_error)} where caused by missing cost function parameter ({percent:.2f}%)"
    )

    # evaluate how much the cost function was changed by the llm (on average)
    avg_creativity_array = []
    for cfs in cost_function_array:
        cfs = [cf for cf in cfs if cf]
        all_dist_count = 0
        for i, cf in enumerate(cfs):
            single_dist_count = 0
            for j, cf2 in enumerate(cfs):
                if i != j:
                    single_dist_count += Levenshtein.distance(cf, cf2)
            single_dist_count /= len(cfs) - 1
            all_dist_count += single_dist_count
        avg_creativity_array.append((all_dist_count / len(cfs)))

    avg_creativity = sum(avg_creativity_array) / len(avg_creativity_array)
    print(f"avg creativity {avg_creativity}")


def print_statistics():
    results = parse_results()
    # assert that all runs had the same amount of iterations
    temp = set()
    [temp.add(len(r.iterations)) for r in results]
    assert len(temp) == 1
    absolute_cost_statistic(results)
    relative_cost_statistic(results)
    failure_statistics(results)
    exception_statistics(results)
    content_statistics(results)


print_statistics()
