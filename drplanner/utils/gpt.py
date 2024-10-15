import math

import tiktoken


def token_cost(amount_of_tokens: int, version: str) -> float:
    million = 1000000.0
    percent = float(amount_of_tokens) / million
    if version == "gpt-4o-mini":
        return 0.15 * percent
    elif version == "gpt-4o":
        return 5.0 * percent
    elif version == "gpt-4-turbo":
        return 10.0 * percent
    else:
        return math.inf


def num_tokens_from_messages(messages, model: str):
    """Return the number of tokens used by a list of messages.
    based on the OpenAI Cookbook, see https://github.com/openai/openai-cookbook."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("*\t\t Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("*\t\t Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("*\t\t Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("*\t\t Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""*\t\t num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    print(f"*\t {num_tokens} prompt tokens counted by num_tokens_from_messages().")
    return num_tokens
