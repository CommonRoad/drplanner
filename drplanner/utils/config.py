from dataclasses import dataclass


@dataclass
class DrPlannerConfiguration:
    t_h = 30  # time horizon
    dt = 0.1  # time step size

    openai_api_key = ""  # your api key for openai
    gpt_version = "gpt-4-turbo-preview"  # "gpt-4-1106-preview"

    token_limit = 8000  # token limits
    temperature = 0.0  # temperature for the LLM

    cost_threshold = 100.00  # threshold for the cost function
    desired_cost = 0.16  # desired cost function value
    iteration_max = 100  # maximum number of iterations

    visualize = False  # flag for whether visualize the intermediate results
    save_solution = True  # flag for whether to save the solution
