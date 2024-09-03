import os
import textwrap
from typing import Tuple

from drplanner.utils.gpt import num_tokens_from_messages

from drplanner.utils.config import DrPlannerConfiguration

from drplanner.prompter.llm import LLM

from drplanner.modular_approach.module import Module
from drplanner.prompter.base import Prompt
from drplanner.prompter.llm import LLMFunction
from drplanner.planners.reactive_planner import ReactiveMotionPlanner
from drplanner.modular_approach.module import Reflection
from drplanner.utils.general import Statistics


# module that generates a prescription based on diagnosis
class PrescriptionModule(Module):
    def __init__(
        self,
        config: DrPlannerConfiguration,
        statistic: Statistics,
        save_dir: str,
        gpt_version: str,
        temperature: float,
    ):
        super().__init__(config, statistic)
        self.prompt_structure = [
            "documentation",
            "cost_function",
            "diagnoses",
            "advice",
        ]

        if self.config.memory_module:
            self.prompt_structure.insert(0, "memory")
        self.save_dir = save_dir
        self.gpt_version = gpt_version

        self.system_prompt = Prompt(["base"])
        path_to_system_prompt = os.path.join(
            self.path_to_prompts,
            "prescription_system_prompt.txt",
        )
        with open(path_to_system_prompt, "r") as file:
            self.system_prompt.set("base", file.read())

        llm_function = LLMFunction(custom=True)
        llm_function.add_array_parameter(
            "helper_methods",
            "Array to collect helper methods",
            llm_function.get_code_parameter("A custom helper method"),
        )
        llm_function.add_code_parameter(
            "cost_function", "The improved cost function of the motion planner"
        )
        self.prescription_llm = LLM(
            gpt_version,
            self.config.openai_api_key,
            llm_function,
            temperature=temperature,
            mockup=self.config.mockup_openAI,
        )

    def generate_user_prompt(
        self, cost_function: str, diagnosis: str, reflection: str, memories: list[str]
    ) -> Prompt:
        user_prompt = Prompt(self.prompt_structure)

        if reflection:
            reflection_prompt = "After analyzing what you did in the previous iteration, here is some advice you should follow:\n"
            reflection_prompt += f"{self.separator}{reflection}{self.separator}"
            user_prompt.set("advice", reflection_prompt)

        # set memory prompt
        if memories:
            memory_prompt = "For reference, here are excerpts of changes which you made in similar situations:\n"
            for m in memories:
                memory_prompt += m + "\n"
            user_prompt.set("memory", memory_prompt[:-1])

        # set documentation
        path_to_doc_prompt = os.path.join(
            self.path_to_prompts, "prescription_documentation_prompt.txt"
        )
        with open(path_to_doc_prompt, "r") as file:
            user_prompt.set("documentation", file.read())

        # set default prompt parts
        cost_function_prompt = (
            "This is the planner's cost function code which you need to improve:\n"
        )
        cost_function_prompt += f"{self.separator}{cost_function}\n{self.separator}"
        user_prompt.set("cost_function", cost_function_prompt)

        user_prompt.set("diagnoses", diagnosis)
        return user_prompt

    def run(
        self,
        diagnosis: str,
        cost_function: str,
        reflection: Reflection,
        few_shots: list[Tuple[str, str]],
        iteration_id: int,
    ) -> ReactiveMotionPlanner:
        memories = [x[1] for x in few_shots]

        if reflection.repair_reflection:
            reflection = reflection.repair_reflection
        else:
            reflection = ""

        user_prompt = self.generate_user_prompt(
            cost_function, diagnosis, reflection, memories
        )

        # query the llm
        messages = LLM.get_messages(
            self.system_prompt.__str__(), user_prompt.__str__(), None
        )
        self.statistic.token_count += num_tokens_from_messages(
            LLM.extract_text_from_messages(messages),
            self.gpt_version,
        )
        save_dir = os.path.join(self.save_dir, self.gpt_version, "prescription")
        mockup_path = ""
        if self.config.mockup_openAI:
            mockup_path = save_dir

        result = self.prescription_llm.query(
            messages, nr_iter=iteration_id, save_dir=save_dir, mockup_path=mockup_path
        )
        if "cost_function" in result.keys():
            cost_function_string = result["cost_function"]
            cost_function_string = textwrap.dedent(cost_function_string)
            if "helper_methods" in result.keys():
                helper_methods = result["helper_methods"]
                helper_methods = [textwrap.dedent(x) for x in helper_methods]
                return ReactiveMotionPlanner(cost_function_string, helper_methods, None, None)
            else:
                return ReactiveMotionPlanner(cost_function_string, None, None, None)
        else:
            self.statistic.missing_parameter_count += 1
            raise ValueError("The llm did not provide an essential parameter!")
