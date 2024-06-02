import csv
import glob
import os

from drplanner.utils.config import DrPlannerConfiguration
from run_drplanner import run_dr_sampling_planner

config = DrPlannerConfiguration()

# initialize results.csv
data = ["scenario", "gpt-version", "initial_cost", "final_cost"]
for i in range(8):
    data.append(f"iter_{i}")
filename = config.save_dir + "results.csv"
file = open(filename, "w+", newline="")
with file:
    write = csv.writer(file)
    write.writerow(data)

# collect all scenarios
scenarios_folder = (
    "/home/sebastian/Documents/Uni/Bachelorarbeit/Scenarios/Datasets/Feasible"
)
xml_files = glob.glob(os.path.join(scenarios_folder, "**", "*.xml"), recursive=True)
scenarios = [os.path.abspath(file) for file in xml_files]

for p in scenarios:
    print(p)
    run_dr_sampling_planner(p)
