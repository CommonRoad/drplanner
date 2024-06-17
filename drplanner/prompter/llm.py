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
        if file_name.endswith(".json"):
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
    def __init__(self):
        # define content of summary object
        summary_dict = {
            "diagnosis": LLMFunction._string_parameter("diagnosis"),
            "prescription": LLMFunction._string_parameter("prescription"),
        }

        # initialize function parameters with summary array
        function_parameters_dict = {
            "summary": LLMFunction._array_parameter(
                LLMFunction._object_parameter(summary_dict),
                "Diagnostic and prescriptive summary",
            )
        }

        self.parameters: dict = LLMFunction._object_parameter(function_parameters_dict)

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
        self.parameters["properties"][parameter_name] = LLMFunction._string_parameter(
            parameter_description
        )

    def add_code_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction._code_parameter(
            parameter_description
        )

    def add_number_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction._number_parameter(
            parameter_description
        )

    @staticmethod
    def _number_parameter(description: str) -> dict:
        return {"type": "number", "description": description}

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
    def _array_parameter(items: dict, description: str) -> dict:
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
        scenario_id=None,
        planner_id=None,
        start_time=None,
        save_dir: str = "../outputs/",
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
            path_variables = [
                save_dir,
                planner_id,
                scenario_id,
                self.gpt_version,
                start_time,
            ]
            path_variables = [x for x in path_variables if x is not None]

            self._save_iteration_as_json(
                messages, content_json, nr_iter, path_variables
            )
            self._save_iteration_as_txt(messages, content_json, nr_iter, path_variables)
            return content_json
        else:
            print(f"*\t <Prompt> Iteration {nr_iter} failed, no response is generated")
            return None

    @staticmethod
    def get_messages(system_prompt: str, user_prompt: str, scenario_png: Union[str, None]):
        if scenario_png:
            base64_image = LLM._encode_image(scenario_png)
            user_content = [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        else:
            user_content = user_prompt
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
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
    def _save_iteration_as_txt(messages, content_json, nr_iter, path_variables: list):
        messages = LLM.extract_text_from_messages(messages)
        text_filename_result = f"result_iter-{nr_iter}.txt"
        text_filename_prompt = f"prompt_iter-{nr_iter}.txt"
        path_variables.append("texts")
        path_variables.append(text_filename_result)
        # Parse the saved content into a txt file
        txt_save_dir = os.path.dirname(os.path.join(*path_variables))

        if not os.path.exists(txt_save_dir):
            os.makedirs(txt_save_dir, exist_ok=True)
        with open(os.path.join(txt_save_dir, text_filename_result), "w") as txt_file:
            for value in content_json.values():
                if isinstance(value, str):
                    txt_file.write(value + "\n")
                elif isinstance(value, list):
                    for item in value:
                        txt_file.write(json.dumps(item) + "\n")

        with open(os.path.join(txt_save_dir, text_filename_prompt), "w") as txt_file:
            for d in messages:
                for value in d.values():
                    if type(value) is str:
                        txt_file.write(value + "\n")
                    elif type(value) is list:
                        for item in value:
                            txt_file.write(json.dumps(item))

    # helper function to save both prompts and responses as parsable json
    @staticmethod
    def _save_iteration_as_json(messages, content_json, nr_iter, path_variables: list):
        messages = LLM.extract_text_from_messages(messages)
        filename_result = f"result_iter-{nr_iter}.json"
        filename_prompt = f"prompt_iter-{nr_iter}.json"
        paths = path_variables.copy()
        paths.append("jsons")
        paths.append(filename_result)
        # Save the content to a JSON file
        json_save_dir = os.path.dirname(os.path.join(*paths))
        if not os.path.exists(json_save_dir):
            os.makedirs(json_save_dir, exist_ok=True)
        # save the prompt
        with open(os.path.join(json_save_dir, filename_prompt), "w") as file:
            json.dump(messages, file)
        # save the result
        with open(os.path.join(json_save_dir, filename_result), "w") as file:
            json.dump(content_json, file)

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
