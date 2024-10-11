import numpy as np
from typing import Union, List
import itertools

from drplanner.utils.gpt import token_cost


# def load_yaml(file_name: Union[Path, str]) -> Union[Dict, None]:
#     """
#     Loads configuration setup from a yaml file
#
#     :param file_name: name of the yaml file
#     """
#     file_name = Path(file_name)
#     config = YAML().load(file_name)
#     return config


class Statistics:
    def __init__(self):
        self.total_costs: list = []

        self.duration: float = 0.0
        self.token_count: int = 0
        self.missing_parameter_count: int = 0
        self.flawed_helper_methods_count: int = 0
        self.missing_few_shot_count: int = 0
        self.added_few_shot_count: int = 0

    def update_iteration(self, total_cost):
        self.total_costs.append(total_cost)

    def get_iteration_data(self) -> list:
        row = [
            self.duration,
            self.token_count,
            self.missing_parameter_count,
            self.flawed_helper_methods_count,
            self.missing_few_shot_count,
            self.added_few_shot_count,
        ]
        row.extend(self.total_costs)
        return row

    def __str__(self):
        cost = token_cost(self.token_count, "gpt-4o-mini")
        return (
            f"* DrPlanner needed {self.duration} seconds. There were {len(self.total_costs)} iterations.\n"
            f"* DrPlanner used {self.token_count} tokens, which cost {cost}$.\n"
            f"* The LLMs forgot to provide essential parameters {self.missing_parameter_count} times.\n"
            f"* The repair LLM did not provide proper helper methods {self.flawed_helper_methods_count} times.\n"
            f"* The memory did not provide few-shots {self.missing_few_shot_count} times.\n"
            f"* The memory was updated {self.added_few_shot_count} times.\n"
            f"* Total costs: {self.total_costs}"
        )


def pass_at_k(n, c, k):
    """
    from "Evaluating Large Language Models Trained on Code"
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


if __name__ == "__main__":
    print(pass_at_k(500, 50, 100))
    print(estimate_pass_at_k([100, 100], [1, 10], 1))

    total = np.array([100, 100])

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, [1, 10], k).mean()
        for k in [1, 10, 100]
        if (total >= k).all()
    }
    print(pass_at_k)
