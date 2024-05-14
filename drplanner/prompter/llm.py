# adapt from https://github.com/real-stanford/reflect
from enum import Enum
from typing import List, Dict

import os
import openai
import json
from datetime import datetime


def check_openai_api_key(api_key, mockup=False):
    if mockup:
        return True
    openai.api_key = api_key
    try:
        openai.models.list()
    except openai.AuthenticationError as e:
        return False
    else:
        return True


def write_prompt_to(filename, messages: List[Dict[str, str]]):
    with open(filename, "w") as file:
        for m in messages:
            for key, value in m.items():
                file.write(value + '\n')


def mockup_query(iteration, save_dir, scenario_id, messages, folder="sampling_mockup"):
    filenames = [
        "iter-1.json",
        "iter-0.json",
        "iter-2.json",
        "iter-3.json",
    ]
    index = iteration % len(filenames)
    filename_result = filenames[index]
    filename_prompt = os.path.join(save_dir, scenario_id, folder, f"prompt_{index}.txt")
    write_prompt_to(filename_prompt, messages)
    path = os.path.join(save_dir, scenario_id, folder, filename_result)
    with open(path) as f:
        # Load the JSON data into a Python data structure
        return json.load(f)


class LLMFunction:
    def __init__(self):
        # define summary object
        summary_object = {
            "diagnosis": LLMFunction._string_parameter("diagnosis"),
            "prescription": LLMFunction._string_parameter("prescription"),
        }

        # initialize parameters with summary
        parameters_object = {
            "summary": LLMFunction._add_array_parameter(
                LLMFunction._object_parameter(summary_object),
                "Diagnostic and prescriptive summary",
            )
        }

        self.parameters: dict = LLMFunction._object_parameter(parameters_object)

    def get_functions(self):
        return [
            {
                "name": "planner_diagnosis",
                "description": "automatic diagnosis of a motion planner",
                "parameters": self.parameters,
            }
        ]

    def add_string_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction._string_parameter(
            parameter_description
        )

    def add_code_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction._code_parameter(
            parameter_description
        )

    @staticmethod
    def _string_parameter(description: str) -> dict:
        return {"type": "string", "description": description}

    @staticmethod
    def _code_parameter(description: str) -> dict:
        return {"type": "string", "format": "python-code", "description": description}

    @staticmethod
    def _object_parameter(properties: dict) -> dict:
        return {"type": "object", "properties": properties}

    @staticmethod
    def _add_array_parameter(items: dict, description: str) -> dict:
        return {"type": "array", "items": items, "description": description}


class LLM:
    def __init__(
            self, gpt_version, api_key, llm_function: LLMFunction, temperature=0.2
    ) -> None:
        self.gpt_version = gpt_version
        if api_key is None:
            raise ValueError("*\t <LLM> OpenAI API key is not provided.")
        else:
            is_valid = check_openai_api_key(api_key, mockup=True)
            if is_valid:
                openai.api_key = api_key
            else:
                raise ValueError(
                    f"*\t <LLM> The given OpenAI API key {api_key} is not valid."
                )

        self.temperature = temperature
        self.llm_function = llm_function

        self._save = True

    def query(
            self,
            scenario_id: str,
            planner_id: str,
            messages: List[Dict[str, str]],
            nr_iter: int = 1,
            save_dir: str = "../outputs/",
            mockup=-1,
    ):
        if mockup > -1:
            return mockup_query(mockup, save_dir, scenario_id, messages)

        functions = self.llm_function.get_functions()
        response = openai.chat.completions.create(
            model=self.gpt_version,
            messages=messages,
            functions=functions,
            function_call={"name": functions[0]["name"]},
            temperature=self.temperature,
        )

        print("RESPONSE: ", response)
        if self._save and response:
            key = datetime.now().strftime("%Y%m%d-%H%M%S")
            content = response.choices[0].message.function_call.arguments
            content_json = json.loads(content)
            print(
                f"*\t <Prompt> Iteration {nr_iter} succeeds, "
                f"{response.usage.total_tokens} tokens are used"
            )
            filename_result = (
                f"result_{planner_id}_{scenario_id}_iter-{nr_iter}_{key}.json"
            )
            filename_prompt = (
                f"prompt_{planner_id}_{scenario_id}_iter-{nr_iter}_{key}.json"
            )
            # Save the content to a JSON file
            save_dir = os.path.dirname(
                os.path.join(save_dir, scenario_id, planner_id, filename_result)
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            # save the prompt
            with open(os.path.join(save_dir, filename_prompt), "w") as file:
                json.dump(messages, file)
            # save the result
            with open(os.path.join(save_dir, filename_result), "w") as file:
                json.dump(content_json, file)
            return content_json
        else:
            print(f"*\t <Prompt> Iteration {nr_iter} failed, no response is generated")
            return None
