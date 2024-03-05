from dataclasses import dataclass


@dataclass
class DiagnoserConfiguration:
    t_h = 30  # time horizon
    dt = 0.1  # time step size

    api_key = ""
    gpt_version = "gpt-4-turbo-preview"  # "gpt-4-1106-preview"

    token_limit = 8000
    cost_threshold = 100.00

    temperature = 0.0

    iteration_max = 100

    desired_cost = 0.16
