import csv
import math
import os.path
from typing import Tuple

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from drplanner.utils.general import estimate_pass_at_k
from drplanner.utils.gpt import token_cost


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
        "added few-shots",  # 8
    ]


def parse_csv(filepath: str) -> list:
    data = []
    with open(filepath, mode="r") as file:
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


def compute_pass_at_k(results: list[str], k=1, p=0.1) -> ndarray:
    num_samples = []
    num_correct = []

    for result_file_path in results:
        scores = extract_final_scores(result_file_path)
        correct_count = 0
        for initial_result, best_result in scores:
            if passed(initial_result, best_result, threshold=p):
                correct_count += 1

        num_samples.append(len(scores))
        num_correct.append(correct_count)

    return estimate_pass_at_k(num_samples, num_correct, k)


def compute_avg_relative_improvement(
    results: list[str],
) -> Tuple[list[float], list[float]]:
    avg_relative_improvements = []
    standard_deviations = []
    for result_file_path in results:
        scores = extract_final_scores(result_file_path)
        improvements = [
            relative_improvement(a, b)
            for (a, b) in scores
            if not relative_improvement(a, b) is None
        ]
        squared_improvements = [x ** 2 for x in improvements]
        mean = sum(improvements) / float(len(improvements))
        squared_mean = sum(squared_improvements) / float(len(squared_improvements))
        standard_deviations.append(math.sqrt(squared_mean - mean ** 2))
        avg_relative_improvements.append(mean)

    return avg_relative_improvements, standard_deviations


def compute_positive_repairs(results: list[str]) -> list[float]:
    actual_repair_percentages = []
    for result_file_path in results:
        scores = extract_final_scores(result_file_path)
        failures = [(a, b) for (a, b) in scores if a == math.inf]
        repairs = [(a, b) for (a, b) in scores if a == math.inf and not b == math.inf]
        actual_repair_percentages.append(float(len(repairs)) / float(len(failures)))

    return actual_repair_percentages


def compute_negative_repairs(results: list[str]) -> list[float]:
    actual_repair_percentages = []
    for result_file_path in results:
        scores = extract_final_scores(result_file_path)
        not_failures = [(a, b) for (a, b) in scores if a < math.inf]
        repairs = [(a, b) for (a, b) in scores if a < math.inf and b == math.inf]
        actual_repair_percentages.append(float(len(repairs)) / float(len(not_failures)))

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


def compute_run_sigma(results: list[str]) -> list[float]:
    sigma_per_run = []
    for result_file_path in results:
        iteration_scores = extract_iterations(result_file_path)
        final_results = extract_final_scores(result_file_path)
        variances = []
        for (i, _), iteration in zip(final_results, iteration_scores):
            filtered = [
                relative_improvement(i, x)
                for x in iteration
                if relative_improvement(i, x) is not None
            ]
            if filtered:
                n = len(filtered)
                mean = sum(filtered) / n
                squared_mean = sum([x ** 2 for x in filtered]) / n
                variances.append(squared_mean - mean ** 2)

        try:
            sigma_per_run.append(math.sqrt(sum(variances) / len(variances)))
        except ValueError:
            print("here")
    return sigma_per_run


def compute_avg_token_usage(result: str) -> float:
    data = parse_csv(result)
    token_data = [float(row[4]) for row in data]
    return sum(token_data) / len(token_data)


def compute_avg_iteration_duration(
    results: list[str],
) -> Tuple[list[float], list[float]]:
    excluded_row_idxs = set()
    durations_per_result = []
    variance_per_result = []
    for result in results:
        iterations = extract_iterations(result)
        for idx, its in enumerate(iterations):
            if math.inf in its:
                excluded_row_idxs.add(idx)
    print(len(excluded_row_idxs))
    for result in results:
        data = parse_csv(result)
        durations = []
        variances = []
        for i, row in enumerate(data):
            if i not in excluded_row_idxs:
                durations.append(float(row[3]))
                variances.append(float(row[3]) ** 2)
            else:
                print("excluded")
        durations_per_result.append(durations)
        variance_per_result.append(variances)

    means = [sum(durations) / len(durations) for durations in durations_per_result]
    e_squared_means = [
        sum(durations_squared) / len(durations_squared)
        for durations_squared in variance_per_result
    ]
    variances = [a - b ** 2 for (a, b) in zip(e_squared_means, means)]
    return means, variances


def extract_variance(dir_name):
    time_until_best_result: list[list[float]] = []
    best_scores: list[list[float]] = []
    best_iterations: list[list[float]] = []
    filenames = []
    for i in range(25):
        filenames.append(f"run_{i}.csv")

    for filename in filenames:
        path = os.path.join(dir_name, filename)
        if not os.path.splitext(filename)[1]:
            new_file_path = path + ".csv"
            os.rename(path, new_file_path)
            path = new_file_path

        scenario_ids = [str(x[0]) for x in parse_csv(path)]
        final_scores = extract_final_scores(path)
        iterations = extract_iterations(path)
        time_samples = []
        best_samples = []
        avg_iter_samples = []

        for scenario_id, (ini, bs), its in zip(scenario_ids, final_scores, iterations):
            if bs > ini:
                best_samples.append(ini)
            else:
                best_samples.append(bs)
            for i, iteration in enumerate(its):
                if bs == iteration:
                    time_samples.append(float(i + 1))
                    break
            its = [x for x in its if x < math.inf]
            if its:
                avg_iter_samples.append(sum(its) / len(its))
                if 44074.07259973753 == sum(its) / len(its):
                    print("arrived")
            else:
                avg_iter_samples.append(math.inf)

        time_until_best_result.append(time_samples)
        best_scores.append(best_samples)
        best_iterations.append(avg_iter_samples)

    return time_until_best_result, best_scores, best_iterations, scenario_ids


def calculate_variance(samples: list[list[float]]) -> Tuple[list[float], list[float]]:
    variances = []
    means = []
    n = len(samples)
    print(n)
    samples = list(zip(*samples))
    for sample in samples:
        sample = tuple(x for x in sample if x < math.inf)
        E = 1 / n * sum(sample)
        means.append(E)
        squared_sample = tuple(x ** 2 for x in sample)
        E_sqrd = 1 / n * sum(squared_sample)
        variances.append(E_sqrd - E ** 2)
    return means, variances


def reorder(l: list, idxs: list[int]) -> list:
    new_l = []
    for i in range(len(l)):
        idx = idxs[i]
        value = l[idx]
        new_l.append(value)
    return new_l


def plot_variance(results, scenario_ids: list[str], label="test", minimum=math.inf):
    means1, variances1 = results[0]
    means2, variances2 = results[1]

    std_devs1 = [min(math.sqrt(var), minimum) for var in variances1]
    std_devs2 = [min(math.sqrt(var), minimum) for var in variances2]

    plt.figure(figsize=(10, 6))

    # Plot error bars for both sets of data
    plt.errorbar(
        range(len(means1)),
        means1,
        yerr=std_devs1,
        fmt="o",
        color="b",
        label="Modular Version Mean",
        elinewidth=10,
        markersize=20,
        alpha=0.4,
    )
    plt.errorbar(
        range(len(means2)),
        means2,
        yerr=std_devs2,
        fmt="o",
        color="r",
        label="Original Version Mean",
        elinewidth=10,
        markersize=20,
        alpha=0.4,
    )

    # Set custom x-tick labels
    custom_x_labels = [f"Sc. {i + 1}" for i in range(len(scenario_ids))]
    plt.xticks(range(len(scenario_ids)), custom_x_labels)

    plt.xlabel("Scenario Nr.")
    plt.ylabel(f"Mean of {label}")
    plt.legend()

    # Save the plot to a file
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(os.path.join(plot_dir, label.replace(" ", "_")))

    plt.close()


def plot_variance_results():
    dirs = [
        "/home/sebastian/Documents/Uni/Experiments/results/variance/variance_with_reflection",
        "/home/sebastian/Documents/Uni/Experiments/results/variance/variance_not_modular",
    ]

    results1 = []
    results2 = []
    results3 = []
    scenario_ids = []
    idxs = []

    for path_to_results in dirs:

        if not idxs:
            m1, m2, m3, scenario_ids = extract_variance(path_to_results)
            m, v = calculate_variance(m1)
            idxs = list(range(len(m)))
            to_sort = list(zip(m, idxs))
            to_sort.sort(key=lambda x: x[0])
            idxs = [x[1] for x in to_sort]
            scenario_ids = reorder(scenario_ids, idxs)
        else:
            m1, m2, m3, _ = extract_variance(path_to_results)

        m, v = calculate_variance(m1)
        means1, variances1 = zip(*reorder(list(zip(m, v)), idxs))
        results1.append((means1, variances1))

        m, v = calculate_variance(m2)
        means2, variances2 = zip(*reorder(list(zip(m, v)), idxs))
        results2.append((means2, variances2))

        m, v = calculate_variance(m3)
        means3, variances3 = zip(*reorder(list(zip(m, v)), idxs))
        results3.append((means3, variances3))

    print(scenario_ids)
    plot_variance(results1, scenario_ids, label="Iteration Number")
    plot_variance(results2, scenario_ids, label="Final Score")
    plot_variance(results3, scenario_ids, label="Average Score", minimum=2500)


def repair_csv(filename: str):
    new_rows = [get_index_row()]
    rows = parse_csv(filename)
    iterations = extract_iterations(filename)
    for i in range(len(rows)):
        it = iterations[i]
        b = min(it)
        new_row = rows[i]
        new_row[2] = b
        new_rows.append(new_row)

    file = open(filename, "w+", newline="")
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(new_rows)


def plot_temperature_results():
    dir = "/home/sebastian/Documents/Uni/Experiments/temperature/"
    result_names = [
        "zero.csv",
        "zero_point_three.csv",
        "zero_point_six.csv",
        "zero_point_nine.csv",
    ]
    for i, name in enumerate(result_names):
        result_names[i] = os.path.join(dir, name)

    results = compute_pass_at_k(result_names)
    x_values = [0.0, 0.3, 0.6, 0.9]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, results, marker="o", linestyle="-", color="b")
    plt.xlabel("Temperature")
    plt.ylabel("Pass@K")
    plt.title("Pass@K vs Temperature")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dir, "temperature_plot.png"))
    plt.show()


def plot_fee_results():
    dir = "/home/sebastian/Documents/Uni/Experiments/results/performance"
    result_names = [
        (
            "not_modular_reflection.csv",
            "not_modular_reflection_expansive.csv",
            "Original",
        ),
        (
            "modular_without_reflection.csv",
            "modular_without_feedback_expensive.csv",
            "Modular",
        ),
        (
            "modular_with_reflection.csv",
            "modular_with_reflection_expansive.csv",
            "Modular + Feedback",
        ),
    ]

    colors = ["b", "g", "r"]

    for idx, (a, b, c) in enumerate(result_names):
        a_path = os.path.join(dir, a)
        b_path = os.path.join(dir, b)
        # plot_cost_a = 0.003825
        # plot_cost_b = 0.001913
        avg_tokens_a = 3 * compute_avg_token_usage(a_path)
        avg_tokens_b = 3 * compute_avg_token_usage(b_path)
        cost_a = token_cost(avg_tokens_a, "gpt-4o-mini")
        cost_b = token_cost(avg_tokens_b, "gpt-4o")
        plt.plot(
            ["gpt-4o-mini", "gpt-4o"],
            [cost_a, cost_b],
            color=colors[idx],
            marker="o",
            label=c,
        )

    plt.xlabel("GPT Versions")
    plt.ylabel("Fee in $")
    plt.legend()
    plt.title("Average Monetary Cost per 3-Iteration-Run")
    plt.savefig(os.path.join(dir, "monetary_cost_plot.png"))
    plt.show()


def plot_time_results():
    folder = (
        "/home/sebastian/Documents/Uni/Experiments/results/tables/iteration_comparison/"
    )

    result_names = [
        ("original.csv", "Original"),
        ("modular.csv", "Modular"),
        ("modular_reflection.csv", "Modular + Feedback"),
    ]
    path_names = []
    for idx, (a, b) in enumerate(result_names):
        path_names.append(os.path.join(folder, a))

    colors = ["b", "g", "r"]
    duration_means, duration_variances = compute_avg_iteration_duration(path_names)

    # Calculate standard deviations from variances
    std_devs = [math.sqrt(var) for var in duration_variances]

    plt.bar(
        ["Original", "Modular", "Modular + Feedback"],
        duration_means,
        yerr=std_devs,
        color=colors,
        alpha=0.8,
        capsize=5,
    )
    plt.ylabel("Duration in [s]")
    plt.legend()
    plt.title("Average Duration of a Single Iteration")
    plt.savefig(os.path.join(folder, "duration_plot.png"))
    plt.show()


def print_results(results: list[str]):
    pass_at_1_10 = compute_pass_at_k(results, k=1)
    pass_at_5_10 = compute_pass_at_k(results, k=5)
    pass_at_1_20 = compute_pass_at_k(results, k=1, p=0.2)
    pass_at_5_20 = compute_pass_at_k(results, k=5, p=0.2)
    pass_at_1_30 = compute_pass_at_k(results, k=1, p=0.3)
    pass_at_5_30 = compute_pass_at_k(results, k=5, p=0.3)
    avg_improv, sigma = compute_avg_relative_improvement(results)
    positive_repairs = compute_positive_repairs(results)
    negative_repairs = compute_negative_repairs(results)
    stability = compute_relative_stability(results)
    sigmas_per_run = compute_run_sigma(results)

    for i, result in enumerate(results):
        print(result)
        a = pass_at_1_10[i] * 100
        b = pass_at_5_10[i] * 100
        c = pass_at_1_20[i] * 100
        d = pass_at_5_20[i] * 100
        e = pass_at_1_30[i] * 100
        f = pass_at_5_30[i] * 100
        g = avg_improv[i] * 100
        h = sigma[i] * 100
        k = stability[i] * 100
        m = positive_repairs[i] * 100
        n = negative_repairs[i] * 100
        j = sigmas_per_run[i] * 100
        print(
            f"& ${a:.1f}\\%$ & ${b:.1f}\\%$ & ${c:.1f}\\%$ & ${d:.1f}\\%$ & ${e:.1f}\\%$ & ${f:.1f}\\%$ & ${g:.1f}\\%$ & ${h:.1f}\\%$ & ${k:.1f}\\%$ \\\\"
        )
        print(f"& ${k:.1f}\\%$ & ${m:.1f}\\%$ & ${n:.1f}\\%$ \\\\")


def plot_average_improvement(filename1: str, filename2: str):
    scores1 = extract_final_scores(filename1)
    improvements1 = [
        relative_improvement(a, b)
        for (a, b) in scores1
        if not relative_improvement(a, b) is None
    ]
    scores2 = extract_final_scores(filename2)
    improvements2 = [
        relative_improvement(a, b)
        for (a, b) in scores2
        if not relative_improvement(a, b) is None
    ]
    improvements = [
        (a, b)
        for (a, b) in zip(improvements1, improvements2)
        if a is not None and b is not None
    ]
    improvements.sort(key=lambda x: x[0])
    scoreA, scoreB = zip(*improvements)
    x = np.arange(len(scoreA))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width / 2, scoreA, width, label="Decoupled DrPlanner", color="b", alpha=0.6
    )
    rects2 = ax.bar(
        x + width / 2,
        scoreB,
        width,
        label="Decoupled DrPlanner + Memory",
        color="r",
        alpha=0.6,
    )

    ax.set_xlabel("Benchmark Nr.")
    ax.set_ylabel("Relative Improvement")
    ax.set_title("Relative Improvement Comparison on unseen Benchmarks")
    ax.set_ylim(-0.5, 1)  # Set y-axis limits
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    plt.savefig("relative_improvement_plot.png")
    plt.show()
