import os
from typing import Tuple

from drplanner.utils.gpt import num_tokens_from_messages
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.utils.general import Statistics

from drplanner.prompter.llm import LLM, LLMFunction
from drplanner.prompter.base import Prompt

from drplanner.modular_approach.module import Module, Diagnosis
from drplanner.modular_approach.module import Reflection

from drplanner.planners.reactive_planner_wrapper import ReactiveMotionPlannerWrapper


# module that generates a diagnosis based on reactive planner performance
class DiagnosisModule(Module):
    def __init__(
        self,
        config: DrPlannerConfiguration,
        statistic: Statistics,
        save_dir: str,
        gpt_version: str,
        temperature: float,
    ):
        super().__init__(config, statistic)
        self.save_dir = save_dir
        self.gpt_version = gpt_version
        self.include_plot = False

        self.prompt_structure = [
            "memory",
            "evaluation",
            "cost_function",
            "task",
            "advice",
        ]

        self.system_prompt = Prompt(["base"])
        path_to_system_prompt = os.path.join(
            self.path_to_prompts,
            "diagnosis_system_prompt.txt",
        )
        with open(path_to_system_prompt, "r") as file:
            appendix = ""
            if self.config.repair_sampling_parameters:
                appendix = "Your second task is to determine the planning configuration for the planner. For this purpose, you can modify the planning horizon length and the allowed lateral deviation from the reference trajectory."

            self.system_prompt.set("base", f"{file.read()}{appendix}")

        llm_function = LLMFunction(custom=True)

        if self.include_plot:
            diagnosis_structure = {
                "scenario": LLMFunction.get_string_parameter(
                    "Analysis of the scenario plot"
                )
            }
        else:
            diagnosis_structure = {}
            self.path_to_plot = ""

        other_parts = {
            "evaluation": LLMFunction.get_string_parameter(
                "Analysis of planner's performance"
            ),
            "cost_function": LLMFunction.get_string_parameter(
                "Analysis of planner's code"
            ),
            "problem": LLMFunction.get_string_parameter(
                "Conclusion on which problems the planner has"
            ),
            "prescriptions": LLMFunction.get_array_parameter(
                LLMFunction.get_string_parameter(
                    "Prescription of a single, precise change to the planner"
                ),
                "Recommended steps to solve the planner's problems",
            ),
        }
        diagnosis_structure.update(other_parts)

        llm_function.add_object_parameter("diagnosis", diagnosis_structure)
        if self.config.repair_sampling_parameters:
            llm_function.add_number_parameter(
                "planning_horizon", "Planning horizon between [2] and [5] seconds"
            )
            llm_function.add_number_parameter(
                "lateral_deviation", "Can be any value between 0 and 5 meters"
            )
        self.llm = LLM(
            gpt_version,
            self.config.openai_api_key,
            llm_function,
            temperature=temperature,
            mockup=self.config.mockup_openAI,
        )

    def generate_user_prompt(
        self,
        planner: ReactiveMotionPlannerWrapper,
        evaluation: str,
        reflection: str,
        memories: list[str],
    ) -> Prompt:
        user_prompt = Prompt(self.prompt_structure)
        path_to_advice_prompt = os.path.join(
            self.path_to_prompts, "diagnosis_advice_prompt.txt"
        )
        path_to_doc_prompt = os.path.join(
            self.path_to_prompts, "diagnosis_documentation_prompt.txt"
        )
        # --- advice section ---
        with open(path_to_advice_prompt, "r") as file:
            advice_prompt = (
                f"It is very important, that you remember the following:\n{file.read()}"
            )
            if self.config.repair_sampling_parameters:
                advice_prompt += f"Planning horizon is a value between [2 - 5] seconds and higher values can greatly improve performance but if it is too long the planner might fail.\n"
        if reflection:
            advice_prompt += "After analyzing what you did in the previous iteration, here is some advice you should follow:\n"
            advice_prompt += f"{self.separator}{reflection}{self.separator}"
        user_prompt.set("advice", advice_prompt)
        # --- memory section ---
        if memories:
            memory_prompt = "Here are some example diagnoses which might be helpful:\n"
            for m in memories:
                memory_prompt += f"{self.separator}{m}\n{self.separator}"
            user_prompt.set("memory", memory_prompt[:-1])
        # --- evaluation section
        if self.include_plot:
            evaluation_prompt = "First of all, process the plot of the current driving situation. The blue rectangles are other cars. The dark orange path represents the trajectory driven by the current planner. The light orange area is the goal area. What are some special features/challenges of this driving task?\n"
        else:
            evaluation_prompt = ""
        evaluation_prompt += "Next, look at the evaluation rating the trajectory generated by the current planner:\n"
        evaluation_prompt += f"{self.separator}{evaluation}\n{self.separator}"
        evaluation_prompt += "Judging from this performance rating, in which areas can the planner improve and which areas are probably not that important?"
        user_prompt.set("evaluation", evaluation_prompt)
        # --- cost function section ---
        cost_function_prompt = "Next, take a more detailed look at the planner itself. Here is the cost function it currently uses:\n"
        cost_function_prompt += (
            f"{self.separator}{planner.cost_function_string}\n{self.separator}"
        )
        if self.config.repair_sampling_parameters:
            cost_function_prompt += f"Furthermore, this is the current sampling configuration: A planning horizon of {planner.max_time_steps/10.0} seconds and an allowed deviation from the reference path of {planner.d} meters.\n"
        cost_function_prompt += "What are the influencing factors and how could they be causing the problem?"
        user_prompt.set("cost_function", cost_function_prompt)
        # --- explanation of the task ---
        task_prompt = "Finally, based on the previous analysis, identify the major problems of the current planner and recommend a strategy to solve these issues.\n"
        with open(path_to_doc_prompt, "r") as file:
            task_prompt += file.read()
        user_prompt.set("task", task_prompt)
        return user_prompt

    def run(
        self,
        evaluation: str,
        planner: ReactiveMotionPlannerWrapper,
        reflection: Reflection,
        few_shots: list[Tuple[str, str]],
        iteration_id: int,
    ) -> Tuple[Diagnosis, int, float]:
        """
        Generates a diagnosis of an input motion planner and prescribes specific changes.
        If desired, adjusts specific sampling intervals of the reactive planner.
        """
        memories = [x[0] for x in few_shots]  # map to diagnosis memories

        if reflection.summary:
            reflection = reflection.summary
        elif reflection.diagnosis_reflection:
            reflection = reflection.diagnosis_reflection
        else:
            reflection = ""

        user_prompt = self.generate_user_prompt(
            planner, evaluation, reflection, memories
        )

        messages = LLM.get_messages(
            self.system_prompt.__str__(), user_prompt.__str__(), None
        )
        self.statistic.token_count += num_tokens_from_messages(
            LLM.extract_text_from_messages(messages),
            self.gpt_version,
        )

        save_dir = os.path.join(self.save_dir, self.gpt_version, "diagnosis")
        mockup_path = ""
        if self.config.mockup_openAI:
            mockup_path = save_dir

        result = self.llm.query(
            messages,
            nr_iter=iteration_id,
            save_dir=save_dir,
            path_to_plot=self.path_to_plot,
            mockup_path=mockup_path,
        )

        if "diagnosis" in result.keys():
            diagnosis = result["diagnosis"]
            scenario_analysis = ""
            if "scenario" in diagnosis.keys():
                scenario_analysis = diagnosis["scenario"]

            evaluation_analysis = ""
            if "evaluation" in diagnosis.keys():
                evaluation_analysis = diagnosis["evaluation"]

            cost_function_analysis = ""
            if "cost_function" in diagnosis.keys():
                cost_function_analysis = diagnosis["cost_function"]

            problem = ""
            if "problem" in diagnosis.keys():
                problem = diagnosis["problem"]

            prescriptions = []
            if "prescriptions" in diagnosis.keys():
                prescriptions = diagnosis["prescriptions"]
            diagnosis = Diagnosis(
                scenario_analysis,
                evaluation_analysis,
                cost_function_analysis,
                problem,
                prescriptions,
            )
        else:
            self.statistic.missing_parameter_count += 1
            raise ValueError("LLM did not provide a diagnosis")

        if "planning_horizon" in result.keys():
            planning_horizon = result["planning_horizon"]
            planning_horizon = int(10 * planning_horizon)
        else:
            planning_horizon = 30

        if "lateral_deviation" in result.keys():
            lateral_distance = result["lateral_deviation"]
        else:
            lateral_distance = 3

        return diagnosis, planning_horizon, lateral_distance
