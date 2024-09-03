import os

from commonroad_dc.costs.evaluation import PlanningProblemCostResult
from drplanner.utils.gpt import num_tokens_from_messages

from drplanner.utils.config import DrPlannerConfiguration

from drplanner.describer.trajectory_description import TrajectoryCostDescription
from drplanner.prompter.llm import LLM

from drplanner.modular_approach.module import Module, Reflection
from drplanner.prompter.base import Prompt
from drplanner.prompter.llm import LLMFunction
from drplanner.utils.general import Statistics


# module that generates a diagnosis based on reactive planner performance
class ReflectionModule(Module):
    def __init__(
        self,
        config: DrPlannerConfiguration,
        statistic: Statistics,
        save_dir: str,
        gpt_version: str,
        temperature: float,
    ):
        super().__init__(config, statistic)
        self.prompt_structure = ["evaluation", "analysis"]
        self.save_dir = save_dir
        self.gpt_version = gpt_version

        self.system_prompt = Prompt(["base"])
        path_to_system_prompt = os.path.join(
            self.path_to_prompts,
            "reflection_system_prompt.txt",
        )
        with open(path_to_system_prompt, "r") as file:
            self.system_prompt.set("base", file.read())

        feedback_llm_function = LLMFunction(custom=True)
        analysis_structure = {
            "evaluation": feedback_llm_function.get_string_parameter(
                "Short analysis of the evaluation"
            ),
            "diagnosis-repair": feedback_llm_function.get_string_parameter(
                "4-step Analysis of the repair-system"
            ),
        }
        feedback_llm_function.add_object_parameter("analysis", analysis_structure)
        feedback_llm_function.add_string_parameter(
            "analysis_summary", "combined summary of every part of the analysis"
        )

        self.feedback_llm = LLM(
            self.gpt_version,
            self.config.openai_api_key,
            feedback_llm_function,
            temperature=temperature,
            mockup=self.config.mockup_openAI,
        )

        reflection_llm_function = LLMFunction(custom=True)
        reflection_structure = {
            "diagnosis": feedback_llm_function.get_string_parameter(
                "reflection on the diagnosis process"
            ),
            "repair": feedback_llm_function.get_string_parameter(
                "reflection on the repair process"
            ),
        }
        reflection_llm_function.add_object_parameter("reflection", reflection_structure)
        self.reflection_llm = LLM(
            self.gpt_version,
            self.config.openai_api_key,
            reflection_llm_function,
            temperature=temperature,
            mockup=self.config.mockup_openAI,
        )

    def generate_feedback_user_prompt(
        self,
        evaluation: str,
        diagnosis: str,
        repair: str,
    ) -> Prompt:
        user_prompt = Prompt(self.prompt_structure)

        evaluation_prompt = "First of all, here is some information which compares the initial planner and its repaired version (if a penalty is infinite that means that planning likely failed completely) :\n"
        evaluation_prompt += f"{self.separator}{evaluation}\n{self.separator}"
        evaluation_prompt += "Analyze how the performance changed and which factors had a big influence on it."
        user_prompt.set("evaluation", evaluation_prompt)

        analysis_prompt = "Now here are the recommendations of the diagnosis-system:\n"
        analysis_prompt += f"{self.separator}{diagnosis}\n{self.separator}"
        analysis_prompt += "And here is their implementation by the repair-system:\n"
        analysis_prompt += f"{self.separator}{repair}\n{self.separator}"
        analysis_prompt += "Now analyze the following questions in-depth: "
        analysis_prompt += (
            "1) Which parts of the diagnosis were successfully implemented (if any)? "
        )
        analysis_prompt += "2) Which parts of the diagnosis were ignored or badly implemented (if any)? "
        analysis_prompt += (
            "3) Were the recommendations helpful for improving the planner? "
        )
        analysis_prompt += "4) Which recommendations affected planner performance the most in your opinion?"
        analysis_prompt += "Finally, summarize your thoughts compactly into a reflection to help improving the next iteration."
        user_prompt.set("analysis", analysis_prompt)
        return user_prompt

    def generate_reflection_user_prompt(
        self, past_reflections: list[Reflection]
    ) -> Prompt:
        user_prompt = Prompt(self.prompt_structure)
        eval_prompt = "Here are the summaries of all previous iterations:\n"
        for i, reflection in enumerate(past_reflections):
            eval_prompt += f"Iteration nr. {i}:\n"
            eval_prompt += f"{self.separator}{reflection.summary}\n{self.separator}"

        eval_prompt += "Now for both the diagnosis and the repair process, briefly reflect on what went well and what did not!"
        user_prompt.set("evaluation", eval_prompt)
        return user_prompt

    def run(
        self,
        cr_initial: PlanningProblemCostResult,
        cr_repaired: PlanningProblemCostResult,
        diagnosis: str,
        repair: str,
        past_reflections: list[Reflection],
        iteration_id: int,
    ) -> Reflection:
        if (iteration_id + 1) % self.config.reflect_at == 0:
            user_prompt = self.generate_reflection_user_prompt(past_reflections)
            # query the llm
            messages = LLM.get_messages(
                self.system_prompt.__str__(), user_prompt.__str__(), None
            )
            self.statistic.token_count += num_tokens_from_messages(
                LLM.extract_text_from_messages(messages),
                self.gpt_version,
            )
            save_dir = os.path.join(
                self.save_dir, self.config.gpt_version, "reflection"
            )
            mockup_path = ""
            if self.config.mockup_openAI:
                mockup_path = save_dir

            result: dict = self.reflection_llm.query(
                messages,
                nr_iter=iteration_id,
                save_dir=save_dir,
                mockup_path=mockup_path,
            )
            if "reflection" in result.keys():
                reflection = result["reflection"]
                diagnosis_reflection = ""
                if "diagnosis" in reflection.keys():
                    diagnosis_reflection = reflection["diagnosis"]

                repair_reflection = ""
                if "repair" in reflection.keys():
                    repair_reflection = reflection["repair"]
            else:
                self.statistic.missing_parameter_count += 1
                raise ValueError("Reflection module did not reflect")

            return Reflection(
                "",
                diagnosis_reflection=diagnosis_reflection,
                repair_reflection=repair_reflection,
            )
        else:
            evaluation = TrajectoryCostDescription.evaluate(
                cr_initial, cr_repaired, "initial", "repaired"
            )
            user_prompt = self.generate_feedback_user_prompt(
                evaluation, diagnosis, repair
            )
            # query the llm
            messages = LLM.get_messages(
                self.system_prompt.__str__(), user_prompt.__str__(), None
            )
            self.statistic.token_count += num_tokens_from_messages(
                LLM.extract_text_from_messages(messages),
                self.gpt_version,
            )
            save_dir = os.path.join(
                self.save_dir, self.config.gpt_version, "reflection"
            )
            mockup_path = ""
            if self.config.mockup_openAI:
                mockup_path = save_dir

            result: dict = self.feedback_llm.query(
                messages,
                nr_iter=iteration_id,
                save_dir=save_dir,
                mockup_path=mockup_path,
            )
            if "analysis_summary" in result.keys():
                summary = result["analysis_summary"]
            else:
                self.statistic.missing_parameter_count += 1
                raise ValueError("Reflection module did not reflect")
            return Reflection(summary)
