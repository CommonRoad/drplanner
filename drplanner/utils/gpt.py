import math

import tiktoken


def token_cost(amount_of_tokens: int, version: str) -> float:
    million = 1000000.0
    percent = float(amount_of_tokens)/million
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
    based on the OpenAI Cookbook."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("*\t Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "*\t Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "*\t Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""*\t num_tokens_from_messages() is not implemented for model {model}. 
            See https://github.com/openai/openai-python/blob/main/chatml.md for information 
            on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    print(f"*\t {num_tokens} prompt tokens counted by num_tokens_from_messages().")
    return num_tokens
