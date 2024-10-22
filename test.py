from drplanner.diagnostics.search import DrSearchPlanner
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.memory.vectorStore import PlanningMemory
from crgeo.commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import DepopulateScenarioPreprocessor
from drplanner.sce_description.collect_data_scenario_over import collect_data_from_scenarios_over
from pathlib import Path


from commonroad.common.file_reader import CommonRoadFileReader

if __name__ == "__main__":
    dataset_ = collect_data_from_scenarios_over(
        scenario_dir=Path('scenarios/DEU_Guetersloh-4_5_T-1.xml'),
        samples_per_scenario=100,
        total_samples=10,
        preprocessor=DepopulateScenarioPreprocessor(depopulator=5)
    )
    print(f"Collected {len(dataset_)} samples")