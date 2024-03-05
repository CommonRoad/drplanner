import numpy as np
from pathlib import Path
from typing import Dict, Union, List
import itertools
from ruamel.yaml import YAML


def load_yaml(file_name: Union[Path, str]) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    file_name = Path(file_name)
    config = YAML().load(file_name)
    return config


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
