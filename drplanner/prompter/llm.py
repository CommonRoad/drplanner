# adapt from https://github.com/real-stanford/reflect
from typing import List, Dict

import os
import openai
import json
from datetime import datetime


def check_openai_api_key(api_key, mockup=False):
    openai.api_key = api_key
    try:
        openai.models.list()
    except openai.AuthenticationError as e:
        return False
    else:
        return True


class LLM:
    def __init__(self, gpt_version, api_key, temperature=0.2) -> None:
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

        self.HEURISTIC_FUNCTION = "improved_heuristic_function"
        self.MOTION_PRIMITIVES = "motion_primitives"
        self.EXTRA_INFORMATION = "extra_information"
        self.functions = [
            {
                "name": "planner_diagnosis",
                "description": "automatic diagnosis of a motion planner",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "diagnosis": {
                                        "type": "string",
                                        "description": "diagnosis",
                                    },
                                    "prescription": {
                                        "type": "string",
                                        "description": "prescription",
                                    },
                                },
                            },
                            "description": "Diagnostic and prescriptive summary",
                        },
                        self.HEURISTIC_FUNCTION: {
                            "type": "string",
                            "format": "python-code",
                            "description": "updated heuristic function",
                        },
                        self.MOTION_PRIMITIVES: {
                            "type": "string",
                            "description": "name of the new motion primitives",
                        },
                        self.EXTRA_INFORMATION: {
                            "type": "string",
                            "description": "extra information",
                        },
                    },
                },
            }
        ]

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
            filenames = [
                "iter-0.json",
                "iter-1.json",
                "iter-2.json",
                "iter-3.json",
            ]
            index = mockup % len(filenames)
            filename_result = filenames[index]
            path = os.path.join(save_dir, scenario_id, "mockup", filename_result)
            with open(path) as f:
                # Load the JSON data into a Python data structure
                return json.load(f)

        # openai.api_key = "pk-RdWHZwMzoNERnWoFTcjSPONQoNOSfzFMnRfcEwliTIEXTXAU"
        # openai.base_url = "https://api.pawan.krd/gpt-3.5-unfiltered/v1/"

        response = openai.chat.completions.create(
            model=self.gpt_version,
            messages=messages,
            functions=self.functions,
            function_call={"name": self.functions[0]["name"]},
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
