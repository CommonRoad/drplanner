import os
import csv
import glob
import shutil
import unittest
from unittest.mock import patch

from commonroad.common.file_reader import CommonRoadFileReader
from drplanner.diagnoser.sampling import DrSamplingPlanner
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.modular.iteration import run_iterative_repair
from drplanner.memory.memory import FewShotMemory


class SamplingPlannerTests(unittest.TestCase):

    @staticmethod
    def results_to_csv(filename: str, data: list):
        """Write a list of data into a CSV file."""
        with open(filename, "a+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def run_dr_sampling_planner(self, scenario_filepath: str, result_filepath: str, config: DrPlannerConfiguration):
        """Run DrSamplingPlanner on the given scenario and save the results to CSV."""
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
        self.results_to_csv(result_filepath, row)

    def run_dr_iteration_planner(self, scenario_filepath: str, result_filepath: str, config: DrPlannerConfiguration):
        """Run DrPlanner's iterative repair method on the scenario and save the results to CSV."""
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
        self.results_to_csv(result_filepath, row)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "phantom_api_key"})
    def run_tests(self, dataset: str, config: DrPlannerConfiguration, modular: bool, skip=None):
        """Run tests on a set of scenarios, either with modular or sampling planners."""
        if skip is None:
            skip = []

        path_to_repo = os.path.dirname(os.path.abspath(__file__))
        path_to_scenarios = os.path.join(path_to_repo, "scenarios", dataset)
        experiment_name = f"modular-{config.gpt_version}-{config.temperature}"
        path_to_results = os.path.join(config.save_dir, experiment_name, dataset)
        result_csv_path = os.path.join(path_to_results, "results.csv")
        result_config_path = os.path.join(path_to_results, "config.txt")

        index_row = [
            "scenario_id", "initial", "best", "duration", "token_count", "missing parameters",
            "flawed helper methods", "missing few-shots", "added few-shots"
        ]
        index_row.extend([f"iter_{i}" for i in range(config.iteration_max)])

        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)

        with open(result_csv_path, "w+", newline="") as result_csv_file:
            writer = csv.writer(result_csv_file)
            writer.writerow(index_row)

        with open(result_config_path, "w+") as file:
            memory_size = FewShotMemory().get_size()
            file.write(f"{config.__str__()}memory size: {memory_size}")

        # Collect and run tests on all scenarios
        xml_files = glob.glob(os.path.join(path_to_scenarios, "**", "*.xml"), recursive=True)
        scenarios = [os.path.abspath(file) for file in xml_files]

        for p in scenarios:
            if all(not (x in p) for x in skip):
                print(p)
                if modular:
                    self.run_dr_iteration_planner(p, result_csv_path, config)
                else:
                    self.run_dr_sampling_planner(p, result_csv_path, config)
            else:
                print(f"Skipping {p}")

        return result_csv_path

    @patch.dict(os.environ, {"OPENAI_API_KEY": "phantom_api_key"})
    def test_sampling_planner(self):
        """Test running the sampling planner with a phantom API key."""
        with patch.dict('os.environ', {"OPENAI_API_KEY": "phantom_api_key"}):
            standard_config = DrPlannerConfiguration(mockup_openAI=True)
            standard_save_dir = standard_config.save_dir
            standard_config.temperature = 0.6
            standard_config.include_plot = False
            standard_config.repair_sampling_parameters = True
            standard_config.iteration_max = 3
            standard_config.reflection_module = False
            standard_config.update_memory_module = False
            standard_config.memory_module = True
            standard_config.include_cost_function_few_shot = True

            standard_config.save_dir = os.path.join(standard_save_dir, "performance", "memory", "memory_10_09")
            csv_path = self.run_tests("test", standard_config, True)

            # Assert that the csv_path file is created
            self.assertTrue(os.path.exists(csv_path))

            # Copy to final destination
            new_path = os.path.join(standard_save_dir, "results", "memory", "memory_10_09.csv")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copy(csv_path, new_path)

            # Assert that the new path file is also created
            self.assertTrue(os.path.exists(new_path))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "phantom_api_key"})
    def test_iteration_planner(self):
        """Test running the iteration planner with a phantom API key."""
        with patch.dict('os.environ', {"OPENAI_API_KEY": "phantom_api_key"}):
            config = DrPlannerConfiguration(mockup_openAI=True)
            config.temperature = 0.8
            config.iteration_max = 5
            config.include_cost_function_few_shot = False
            config.reflection_module = True

            # Run the test and assert results
            csv_path = self.run_tests("test", config, False)
            self.assertTrue(os.path.exists(csv_path))


if __name__ == "__main__":
    unittest.main()
