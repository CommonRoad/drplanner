import os

from commonroad_dc.costs.evaluation import PlanningProblemCostResult
from drplanner.utils.config import DrPlannerConfiguration

from describer.trajectory_description import TrajectoryCostDescription
from drplanner.prompter.llm import LLM

from drplanner.modular_approach.module import Module
from drplanner.memory.memory import FewShotMemory
from drplanner.prompter.base import Prompt
from drplanner.prompter.llm import LLMFunction
from modular_approach.iteration import Reflection


# module that generates a diagnosis based on reactive planner performance
class ReflectionModule(Module):
    def __init__(
        self,
        config: DrPlannerConfiguration,
        memory: FewShotMemory,
        save_dir: str,
        gpt_version: str,
        temperature: float,
    ):
        super().__init__(config)
        self.prompt_structure = ["evaluation", "analysis"]
        self.memory = memory
        self.save_dir = save_dir
        self.gpt_version = gpt_version

        self.system_prompt = Prompt(["base"])
        path_to_system_prompt = os.path.join(
            self.path_to_prompts,
            "reflection_system_prompt.txt",
        )
        with open(path_to_system_prompt, "r") as file:
            self.system_prompt.set("base", file.read())

        llm_function = LLMFunction(custom=True)

        analysis_structure = {
            "evaluation": llm_function.get_string_parameter(
                "Short analysis of the evaluation"
            ),
            "diagnosis-repair": llm_function.get_string_parameter(
                "4-step Analysis of the repair-system"
            ),
        }
        llm_function.add_object_parameter("analysis", analysis_structure)

        reflection_structure = {
            "diagnosis_reflection": llm_function.get_string_parameter(
                "Reflection on the diagnosis process"
            ),
            #"repair_reflection": llm_function.get_string_parameter(
            #    "Reflection on the repair process"
            #),
        }
        llm_function.add_string_parameter(
            "reflection", "summary of the previous analysis"
        )

        self.reflection_llm = LLM(
            self.gpt_version,
            self.config.openai_api_key,
            llm_function,
            temperature=temperature,
            mockup=self.config.mockup_openAI,
        )

    def generate_user_prompt(
        self,
        evaluation: str,
        diagnosis: str,
        repair: str,
        past_reflections: list[Reflection],
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
        analysis_prompt += "1) Which parts of the diagnosis were successfully implemented (if any)? "
        analysis_prompt += "2) Which parts of the diagnosis were ignored or badly implemented (if any)? "
        analysis_prompt += "3) Were the recommendations helpful for improving the planner? "
        analysis_prompt += "4) Which recommendations affected planner performance the most in your opinion?"
        analysis_prompt += "Finally, summarize your thoughts compactly into a reflection to help improving the next iteration."
        user_prompt.set("analysis", analysis_prompt)
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
        evaluation = TrajectoryCostDescription.evaluate(cr_initial, cr_repaired, "initial", "repaired")
        user_prompt = self.generate_user_prompt(
            evaluation, diagnosis, repair, past_reflections
        )
        # query the llm
        messages = LLM.get_messages(
            self.system_prompt.__str__(), user_prompt.__str__(), None
        )
        save_dir = os.path.join(self.save_dir, self.config.gpt_version, "reflection")
        mockup_path = ""
        if self.config.mockup_openAI:
            mockup_path = save_dir

        result: dict = self.reflection_llm.query(
            messages,
            nr_iter=iteration_id,
            save_dir=save_dir,
            mockup_path=mockup_path
        )
        if "reflection" in result.keys():
            reflection = result["reflection"]
        else:
            raise ValueError("Reflection module did not reflect")
        return Reflection(reflection)
