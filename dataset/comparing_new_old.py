import os
import pickle
import statistics
import sys
import yaml
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

from SMP.batch_processing.batch_processing_parallel import run_parallel_processing
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch
from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig


class StudentMotionPlanner(GreedyBestFirstSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)

    def evaluation_function(self, node_current: PriorityNode) -> float:

        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:
        raise NotImplementedError("This method should be replaced dynamically")

    def multiply(self, x, y):
        return x * y


def inject_method(cls, method_code):
    namespace = {}
    exec(method_code, globals(), namespace)
    method_name = list(namespace.keys())[0]
    setattr(cls, method_name, namespace[method_name])


with open('./structure_file.txt', 'r') as f:
    basic_structure = f.read()
CHECK_VERSION = 10042157

path_notebook = os.getcwd()
sys.path.append(os.path.join(path_notebook, "../../"))
# load scenario
path_scenario = os.path.join(path_notebook, "../../scenarios/exercise/")

with open(f'../dataset/raw/{CHECK_VERSION}.yaml', 'r') as file:
    code_states = yaml.safe_load(file)

initial_heuristic = code_states['input']['heuristic_function'].lstrip().replace("\\ \n ", "")
improved_heuristic = code_states['output']['improved_heuristic_function'].lstrip().replace("\\ \n", "")


def run_tests(initial_heuristic, improved_heuristic):
    """ Injects the respective code and executes the tests"""
    initial = initial_heuristic.split('\n')
    initial[0] = '    ' + initial[0]
    with open('../../SMP/motion_planner/search_algorithms/automatic.py', 'w') as f:
        f.write(basic_structure)
        f.write('    \n'.join(initial))
    initial_results = run_parallel_processing()

    improved = improved_heuristic.split('\n')
    improved[0] = '    ' + improved[0]
    with open('../../SMP/motion_planner/search_algorithms/automatic.py', 'w') as f:
        f.write(basic_structure)
        f.write('    \n'.join(improved))

    improved_results = run_parallel_processing()

    return initial_results, improved_results

def execute_all_files(folder_path, yaml_folder='raw'):
    """Iterates over all files, parses them and calls the run_tests function"""
    results = {}
    # Loop through each file in the directory
    yaml_folder = os.path.join(folder_path, yaml_folder)
    for idx, filename in enumerate(os.listdir(yaml_folder)):
        if filename.endswith('.yml'):
            id = Path(filename).stem
            yaml_file = os.path.join(yaml_folder, filename)

            with open(yaml_file, 'r') as file:
                code_states = yaml.safe_load(file)

            initial_heuristic = code_states['input']['heuristic_function'].lstrip().replace("\\ \n ", "")
            improved_heuristic = code_states['output']['improved_heuristic_function'].lstrip().replace("\\ \n", "")

            initial, improved = run_tests(initial_heuristic, improved_heuristic)
            results[id] = {}
            results[id]['initial'] = initial
            results[id]['improved'] = improved

            with open(f'./results/{id}.pkl', 'wb') as f:
                pickle.dump(results[id], f)

    with open(f'./results/full.pkl', 'wb') as f:
        pickle.dump(results, f)


execute_all_files(f'../dataset/', 'raw_new_samples')

