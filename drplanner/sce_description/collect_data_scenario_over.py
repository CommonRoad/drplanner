import sys; import os; sys.path.insert(0, os.getcwd())
import torch

from pathlib import Path

from crgeo.commonroad_geometric.dataset.collection.dataset_collector import DatasetCollector
from crgeo.commonroad_geometric.dataset.commonroad_data import CommonRoadData
from crgeo.commonroad_geometric.dataset.extraction.traffic.edge_drawers.implementations import VoronoiEdgeDrawer
from crgeo.commonroad_geometric.dataset.extraction.traffic.traffic_extractor import TrafficExtractorOptions
from crgeo.commonroad_geometric.dataset.extraction.traffic.traffic_extractor_factory import TrafficExtractorFactory
from crgeo.commonroad_geometric.dataset.scenario.iteration.scenario_bundle import ScenarioBundle
from crgeo.commonroad_geometric.dataset.scenario.iteration.scenario_iterator import ScenarioIterator
from crgeo.commonroad_geometric.dataset.scenario.preprocessing.preprocessors.implementations import DepopulateScenarioPreprocessor
from crgeo.commonroad_geometric.dataset.scenario.preprocessing.preprocessors.scenario_preprocessor import ScenarioPreprocessor
from crgeo.commonroad_geometric.simulation.interfaces.static.scenario_simulation import ScenarioSimulationOptions
from crgeo.commonroad_geometric.simulation.simulation_factory import SimulationFactory


def collect_data_from_scenarios_over(
    scenario_dir: Path,
    preprocessor: ScenarioPreprocessor,
    #samples_per_scenario: int,
) -> list[CommonRoadData]:
    samples_per_scenario = 1
    collector = DatasetCollector(
        extractor_factory=TrafficExtractorFactory(
            options=TrafficExtractorOptions(
                edge_drawer=VoronoiEdgeDrawer(dist_threshold=50),
            )
        ),
        simulation_factory=SimulationFactory(
            options=ScenarioSimulationOptions()  # We could specify options for the simulation here
        ),
        progress=True
    )

    scenario_iterator = ScenarioIterator(
        directory=scenario_dir,
        preprocessor=preprocessor,
        workers=1
    )

    scenario_data: list[CommonRoadData] = []
    for scenario_bundle, _ in scenario_iterator:
        scenario_bundle: ScenarioBundle
        print(f"Collecting data for {scenario_bundle.scenario_path}")
        for time_step, data in collector.collect(
            scenario=scenario_bundle.preprocessed_scenario,
            planning_problem_set=scenario_bundle.preprocessed_planning_problem_set,
            max_samples=samples_per_scenario,
        ):
            #TODO:标签怎么处理？
            #data.y=torch.tensor([1])
            scenario_data.append(data)
        break
        

    return scenario_data


if __name__ == '__main__':
    dataset_ = collect_data_from_scenarios_over(
        scenario_dir=Path('data/highd-sample'),
        #samples_per_scenario=1,
        preprocessor=DepopulateScenarioPreprocessor(depopulator=5)
    )
    print(f"Collected {len(dataset_)} samples")
