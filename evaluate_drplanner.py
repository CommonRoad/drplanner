import csv
import math
import os.path
from typing import Tuple

from numpy import ndarray

from drplanner.utils.general import estimate_pass_at_k


def get_index_row() -> list[str]:
    return [
        "scenario_id",  # 0
        "initial",  # 1
        "best",  # 2
        "duration",  # 3
        "token_count",  # 4
        "missing parameters",  # 5
        "flawed helper methods",  # 6
        "missing few-shots",  # 7
        "added few-shots"  # 8
    ]


def parse_csv(filepath: str) -> list:
    data = []
    with open(filepath, mode='r') as file:
        for row in csv.reader(file):
            data.append(row)
    return data[1:]


def strip_scenario_id(scenario_id: str) -> str:
    if scenario_id.endswith(".cr"):
        return scenario_id[:-3]
    else:
        return scenario_id


def pair_up_rows(result_a: str, result_b: str) -> list[Tuple[list, list]]:
    rows_a = parse_csv(result_a)
    rows_b = parse_csv(result_b)
    # pair up equal scenario_ids
    paired_rows: list[Tuple[list, list]] = []
    for row_a in rows_a:
        scenario_id_a = strip_scenario_id(row_a[0])
        for row_b in rows_b:
            scenario_id_b = strip_scenario_id(row_b[0])
            if scenario_id_a == scenario_id_b:
                paired_rows.append((row_a, row_b))
                break
    return paired_rows


def relative_improvement(a: float, b: float) -> float:
    if a == b == math.inf:
        return 0.0
    if a != b and (a == math.inf or b == math.inf):
        return None
    return (a - b) / a


def passed(initial_score, best_score, threshold=0.05) -> bool:
    if best_score == math.inf:
        return False
    elif initial_score == math.inf:
        return True
    elif initial_score <= best_score:
        return False
    else:
        percentage = relative_improvement(initial_score, best_score)
        return percentage > threshold


def extract_final_scores(result_file_path: str) -> list[Tuple[float, float]]:
    rows = parse_csv(result_file_path)
    scores: list[Tuple[float, float]] = []
    for row in rows:
        try:
            initial_score = float(row[1])
        except ValueError as _:
            initial_score = math.inf
        try:
            best_score = float(row[2])
        except ValueError as _:
            best_score = math.inf
        scores.append((initial_score, best_score))
    return scores


def extract_iterations(result_file_path: str) -> list[list[float]]:
    rows = parse_csv(result_file_path)
    iterations = []
    for row in rows:
        iteration_str = row[9:]
        iteration = []
        for i in iteration_str:
            try:
                score = float(i)
            except ValueError as _:
                score = math.inf
            iteration.append(score)
        iterations.append(iteration)

    return iterations


def compute_pass_at_k(results: list[str], k=1, threshold=0.1) -> ndarray:
    num_samples = []
    num_correct = []

    for result_file_path in results:
        scores = extract_final_scores(result_file_path)
        correct_count = 0
        for initial_result, best_result in scores:
            if passed(initial_result, best_result, threshold=threshold):
                correct_count += 1

        num_samples.append(len(scores))
        num_correct.append(correct_count)

    return estimate_pass_at_k(num_samples, num_correct, k)


def compute_avg_relative_improvement(results: list[str]) -> list[float]:
    avg_relative_improvements = []
    for result_file_path in results:
        scores = extract_final_scores(result_file_path)
        improvements = [relative_improvement(a, b) for (a, b) in scores if relative_improvement(a, b)]
        avg_relative_improvements.append(sum(improvements) / float(len(improvements)))

    return avg_relative_improvements


def compute_positive_repairs(results: list[str]) -> list[float]:
    actual_repair_percentages = []
    for result_file_path in results:
        scores = extract_final_scores(result_file_path)
        repairs = [(a, b) for (a, b) in scores if a == math.inf and not b == math.inf]
        actual_repair_percentages.append(float(len(repairs)) / float(len(scores)))

    return actual_repair_percentages


def compute_negative_repairs(results: list[str]) -> list[float]:
    actual_repair_percentages = []
    for result_file_path in results:
        scores = extract_final_scores(result_file_path)
        repairs = [(a, b) for (a, b) in scores if not a == math.inf and b == math.inf]
        actual_repair_percentages.append(float(len(repairs)) / float(len(scores)))

    return actual_repair_percentages


def compute_relative_stability(results: list[str]) -> list[float]:
    relative_stability = []
    for result_file_path in results:
        iteration_scores = extract_iterations(result_file_path)
        exception_count = 0
        sample_count = 0
        for iteration in iteration_scores:
            exceptions = [x for x in iteration if x == math.inf]
            exception_count += len(exceptions)
            sample_count += len(iteration)

        relative_stability.append(1 - float(exception_count) / float(sample_count))

    return relative_stability


def compute_variance(results: list[str]) -> float:
    pass


def print_results(results: list[str]):
    pass_at_1 = compute_pass_at_k(results, k=5)
    avg_improv = compute_avg_relative_improvement(results)
    positive_repairs = compute_positive_repairs(results)
    negative_repairs = compute_negative_repairs(results)
    stability = compute_relative_stability(results)

    print(pass_at_1)
    for i, result in enumerate(results):
        print(result)
        print(f"pass at 1: {pass_at_1[i]} |avg improvement: {avg_improv[i]} |amount of repairs: pos ({positive_repairs[i]}), neg ({negative_repairs[i]})|stability: {stability[i]}")


path_to_results = "/home/sebastian/Documents/Uni/Experiments/results"
result_names = [
    "performance_original.csv",
    "performance_with_feedback.csv",
    "performance_without_feedback.csv"
]

for i, name in enumerate(result_names):
    result_names[i] = os.path.join(path_to_results, name)

print_results(result_names)
