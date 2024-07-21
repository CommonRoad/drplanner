# adapt from https://github.com/real-stanford/reflect
import base64
from typing import List, Dict, Union

import os
import openai
import json


def check_openai_api_key(api_key, mockup=False):
    if mockup:
        return True
    openai.api_key = api_key
    try:
        openai.models.list()
    except openai.AuthenticationError as _:
        return False
    else:
        return True


def mockup_query(
    iteration,
    directory="/home/sebastian/Documents/Uni/Bachelorarbeit/DrPlanner_Data/mockup/debug",
):
    filenames = []
    # finds all .jsons in the directory and assumes them to be mockup responses
    for file_name in os.listdir(directory):
        if file_name.endswith(".json") and file_name.startswith("result"):
            # Construct the full file path
            file_path = os.path.join(directory, file_name)
            filenames.append(file_path)

    filenames.sort()
    index = iteration % len(filenames)
    filename_response = filenames[index]

    with open(filename_response) as f:
        return json.load(f)


# A class representing an OpenAI api function call
class LLMFunction:
    def __init__(self, custom=False):
        if not custom:
            # define content of summary object
            summary_dict = {
                "diagnosis": LLMFunction.get_string_parameter("diagnosis"),
                "prescription": LLMFunction.get_string_parameter("prescription"),
            }

            # initialize function parameters with summary array
            function_parameters_dict = {
                "summary": LLMFunction.get_array_parameter(
                    LLMFunction.get_object_parameter(summary_dict),
                    "Diagnostic and prescriptive summary",
                )
            }
        else:
            function_parameters_dict = {}

        self.parameters: dict = LLMFunction.get_object_parameter(
            function_parameters_dict
        )

    # transforms the function into a form required by the OpenAI api
    def get_function_as_list(self):
        parameters = self.parameters["properties"]
        self.parameters["required"] = list(parameters.keys())
        return [
            {
                "name": "planner_diagnosis",
                "description": "automatic diagnosis of a motion planner",
                "parameters": self.parameters,
            }
        ]

    def add_string_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][
            parameter_name
        ] = LLMFunction.get_string_parameter(parameter_description)

    def add_code_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction.get_code_parameter(
            parameter_description
        )

    def add_number_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][
            parameter_name
        ] = LLMFunction.get_number_parameter(parameter_description)

    def add_array_parameter(
        self, parameter_name: str, parameter_description: str, items: dict
    ):
        self.parameters["properties"][parameter_name] = LLMFunction.get_array_parameter(
            items, parameter_description
        )

    def add_object_parameter(self, parameter_name: str, properties: dict):
        self.parameters["properties"][
            parameter_name
        ] = LLMFunction.get_object_parameter(properties)

    @staticmethod
    def get_number_parameter(description: str) -> dict:
        return {"type": "number", "description": description}

    @staticmethod
    def get_string_parameter(description: str) -> dict:
        return {"type": "string", "description": description}

    @staticmethod
    def get_code_parameter(description: str) -> dict:
        return {"type": "string", "format": "python-code", "description": description}

    @staticmethod
    def get_object_parameter(properties: dict) -> dict:
        return {"type": "object", "properties": properties}

    @staticmethod
    def get_array_parameter(items: dict, description: str) -> dict:
        return {"type": "array", "items": items, "description": description}


# interface class managing communication with OpenAI api through query method
class LLM:
    def __init__(
        self,
        gpt_version,
        api_key,
        llm_function: LLMFunction,
        temperature=0.2,
        mockup=False,
    ) -> None:

        self.gpt_version = gpt_version

        # make sure the api key is provided and valid
        if api_key is None:
            raise ValueError("*\t <LLM> OpenAI API key is not provided.")
        else:
            is_valid = check_openai_api_key(api_key, mockup=mockup)
            if is_valid:
                openai.api_key = api_key
            else:
                raise ValueError(
                    f"*\t <LLM> The given OpenAI API key {api_key} is not valid."
                )

        self.temperature = temperature
        self.llm_function = llm_function

        self._save = True

    # send <messages> to the OpenAI api
    def query(
        self,
        messages: List[Dict[str, str]],
        save_dir: str = "../../outputs/",
        nr_iter: int = 1,
        mockup_nr_iter: int = -1,
    ):
        # check whether this is a mockup run
        if mockup_nr_iter > -1:
            response = mockup_query(mockup_nr_iter)
        # otherwise send the query
        else:
            functions = self.llm_function.get_function_as_list()
            response = openai.chat.completions.create(
                model=self.gpt_version,
                messages=messages,
                functions=functions,
                function_call={"name": functions[0]["name"]},
                temperature=self.temperature,
            )

        print("RESPONSE: ", response)

        # save the result to save_dir with a structure specified by the input parameters:
        # planner_id/scenario_id/gpt_version/start_time
        if self._save and response:
            if mockup_nr_iter > -1:
                content_json = response
            else:
                content = response.choices[0].message.function_call.arguments
                content_json = json.loads(content)
                print(
                    f"*\t <Prompt> Iteration {nr_iter} succeeds, "
                    f"{response.usage.total_tokens} tokens are used"
                )

            self._save_iteration_as_json(messages, content_json, nr_iter, save_dir)
            self._save_iteration_as_txt(messages, content_json, nr_iter, save_dir)
            return content_json
        else:
            print(f"*\t <Prompt> Iteration {nr_iter} failed, no response is generated")
            return None

    @staticmethod
    def get_messages(
        system_prompt: str, user_prompt: str, scenario_png: Union[str, None]
    ):
        if scenario_png:
            base64_image = LLM._encode_image(scenario_png)
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        else:
            user_content = user_prompt
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def extract_text_from_messages(messages: list) -> list:
        content = messages[1]["content"]

        if isinstance(content, list):
            content = content[0]["text"]
        messages[1]["content"] = content
        return messages

    # helper function to save both prompts and responses in a human-readable form
    @staticmethod
    def _save_iteration_as_txt(messages, content_json, nr_iter, save_dir: str):
        messages = LLM.extract_text_from_messages(messages)
        text_filename_result = f"result_iter-{nr_iter}.txt"
        text_filename_prompt = f"prompt_iter-{nr_iter}.txt"
        save_dir = os.path.join(save_dir, "texts")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, text_filename_result), "w") as txt_file:
            for value in content_json.values():
                if isinstance(value, str):
                    txt_file.write(value + "\n")
                elif isinstance(value, list):
                    for item in value:
                        txt_file.write(json.dumps(item) + "\n")

        with open(os.path.join(save_dir, text_filename_prompt), "w") as txt_file:
            for d in messages:
                for value in d.values():
                    if type(value) is str:
                        txt_file.write(value + "\n")
                    elif type(value) is list:
                        for item in value:
                            txt_file.write(json.dumps(item))

    # helper function to save both prompts and responses as parsable json
    @staticmethod
    def _save_iteration_as_json(messages, content_json, nr_iter, save_dir: str):
        messages = LLM.extract_text_from_messages(messages)
        filename_result = f"result_iter-{nr_iter}.json"
        filename_prompt = f"prompt_iter-{nr_iter}.json"
        save_dir = os.path.join(save_dir, "jsons")
        # Save the content to a JSON file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # save the prompt
        with open(os.path.join(save_dir, filename_prompt), "w") as file:
            json.dump(messages, file)
        # save the result
        with open(os.path.join(save_dir, filename_result), "w") as file:
            json.dump(content_json, file)

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
