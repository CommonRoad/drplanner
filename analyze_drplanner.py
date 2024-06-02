import csv
import math

from drplanner.utils.config import DrPlannerConfiguration


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
        if len(row) > 4:
            for i in range(4, len(row)):
                self.iterations.append(float(row[i]))


def parse_results():
    config = DrPlannerConfiguration()
    filename = config.save_dir + "results.csv"
    planner_results = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for (i, row) in enumerate(reader):
            if i == 0:
                continue
            planner_results.append(DrPlannerResult(row))
    return planner_results


results = parse_results()
gain = [
    r.initial_cost - r.final_cost
    for r in results
    if not math.isinf(r.initial_cost) and not math.isinf(r.final_cost)
]
print(f"maximal absolute cost reduction: {max(gain)}")
print(f"minimal absolute cost reduction: {min(gain)}")
avg_gain = sum(gain) / len(gain)
print(f"average absolute cost reduction: {avg_gain}")

gain = [
    ((r.initial_cost - r.final_cost) / r.initial_cost) * 100
    for r in results
    if not math.isinf(r.initial_cost) and not math.isinf(r.final_cost)
]
print(f"maximal relative cost reduction: {max(gain):.2f}%")
print(f"minimal relative cost reduction: {min(gain):.2f}%")
avg_gain = sum(gain) / len(gain)
print(f"average relative cost reduction: {avg_gain:.2f}%")
print(f"Of {len(results)} different scenarios:")
fixed = [
    r for r in results if math.isinf(r.initial_cost) and not math.isinf(r.final_cost)
]
print(f"fixed planning for {len(fixed)} scenarios")
not_fixed = [
    r for r in results if math.isinf(r.initial_cost) and math.isinf(r.final_cost)
]
print(f"could not fix planning for {len(not_fixed)} scenarios")
threshold = 5.0
stagnated = [
    r
    for r in results
    if (
        abs(r.initial_cost - r.final_cost) < threshold
        or r.initial_cost - r.final_cost < 0
    )
    and not math.isinf(r.initial_cost)
    and not math.isinf(r.final_cost)
]
print(f"did not significantly improve {len(stagnated)} scenarios")
iterations = [r.iterations for r in results]
iterations = [item for sublist in iterations for item in sublist]
print(f"Of {len(iterations)} iterations:")
failed = [i for i in iterations if math.isinf(i)]
not_failed = [i for i in iterations if not math.isinf(i)]
avg_failures = len(failed) / len(iterations) * 100
print(f"exceptions occurred in {avg_failures:.2f}%")
distinct = []
[
    distinct.append(num)
    for num in not_failed
    if all(abs(num - x) > 5.0 for x in distinct)
]
percent = len(distinct) / len(not_failed) * 100
print(
    f"{len(not_failed)} iterations where successful but only {len(distinct)} produced distinct results ({percent:.2f}%)"
)
