import math
import os
import textwrap
from abc import abstractmethod, ABC
from typing import Tuple, Union

from drplanner.prompter.llm import LLMFunction, LLM

from commonroad.common.solution import CostFunction

from drplanner.prompter.base import PrompterBase, Prompt

from drplanner.describer.trajectory_description import TrajectoryCostDescription
from memory.memory import FewShotMemory
from planners.reactive_planner import (
    ReactiveMotionPlanner,
    get_basic_configuration_path,
)
from utils.config import DrPlannerConfiguration


class Module(ABC):
    def __init__(self):
        self.config = DrPlannerConfiguration()
        self.separator = '"""\n'
        self.path_to_prompts = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prompts"
        )

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


# module that evaluates reactive planner on a scenario
class EvaluationModule(Module):
    def __init__(self, cost_function_type: CostFunction = CostFunction.SM1):
        super().__init__()
        self.absolute_config_path = get_basic_configuration_path()
        self.cost_function_type = cost_function_type

    def run(
        self, absolute_scenario_path: str, motion_planner: ReactiveMotionPlanner
    ) -> Tuple[str, float]:
        try:
            cost_result = motion_planner.evaluate_on_scenario(
                absolute_scenario_path,
                self.absolute_config_path,
                cost_type=self.cost_function_type,
            )
            evaluation = TrajectoryCostDescription().generate(
                cost_result, self.config.desired_cost
            )
            total_cost = cost_result.total_costs
        except Exception as e:
            evaluation = PrompterBase.generate_exception_description(e)
            total_cost = math.inf
        return evaluation, total_cost


# module that generates a diagnosis based on reactive planner performance
class DiagnosisModule(Module):
    def __init__(
        self,
        memory: FewShotMemory,
        prompt_structure: list[str],
        few_shot_amount: int,
        save_dir: str,
        gpt_version: str,
        temperature: float,
    ):
        super().__init__()
        self.prompt_structure = prompt_structure
        self.memory = memory
        self.few_shot_amount = few_shot_amount
        self.save_dir = save_dir
        self.gpt_version = gpt_version

        self.system_prompt = Prompt(["base"])
        path_to_system_prompt = os.path.join(
            self.path_to_prompts,
            "diagnosis_system_prompt.txt",
        )
        with open(path_to_system_prompt, "r") as file:
            self.system_prompt.set("base", file.read())

        llm_function = LLMFunction(custom=True)
        diagnosis_structure = {
            "general": LLMFunction.get_string_parameter(
                "General: general analysis of the problem"
            ),
            "specific": LLMFunction.get_string_parameter(
                "Specific: analysis of the cost-function flaws"
            ),
            "strategy": LLMFunction.get_string_parameter(
                "Strategy: recommendation on how to tune the cost-function"
            ),
        }
        llm_function.add_object_parameter("diagnosis", diagnosis_structure)
        self.llm = LLM(
            gpt_version,
            self.config.openai_api_key,
            llm_function,
            temperature=temperature,
            mockup=self.config.mockup_openAI,
        )

    def generate_user_prompt(self, cost_function: str, evaluation: str) -> Prompt:
        user_prompt = Prompt(self.prompt_structure)

        # set reflection prompt
        feedback = ""
        # TODO init feedback
        # if self.feedback[0]:
        #     feedback += self.feedback[0] + "\n"
        # if self.feedback[1]:
        #     feedback += self.feedback[1] + "\n"
        if feedback:
            reflection_prompt = "What follows is an in-depth analysis of the previous iteration. Pay close attention to its advice and incorporate it into your diagnosis."
            reflection_prompt += f"{self.separator}{feedback}{self.separator}"
            user_prompt.set("reflection", reflection_prompt)

        if feedback:
            offset = 0
        else:
            offset = 1
        # set memory prompt
        if self.few_shot_amount > 0:
            memories = self.memory.retrieve(
                evaluation,
                collection_name="diagnosis",
                n=(self.few_shot_amount + offset),
            )
            memory_prompt = "Here are some example diagnoses which might be helpful:\n"
            for m in memories:
                memory_prompt += f"{self.separator}{m}\n{self.separator}"
            user_prompt.set("memory", memory_prompt[:-1])

        # set documentation
        path_to_doc_prompt = os.path.join(
            self.path_to_prompts, "diagnosis_documentation_prompt.txt"
        )
        with open(path_to_doc_prompt, "r") as file:
            prompt = (
                f"Here is a short overview over all available tools:\n{file.read()}"
            )
            user_prompt.set("documentation", prompt)

        # set default prompt parts
        problem_prompt = "First of all, only look at this general feedback about the current planner:\n"
        problem_prompt += f"{self.separator}{evaluation}\n{self.separator}"
        problem_prompt += "From a top-level perspective what could be the planner's problem and its underlying cause?"
        user_prompt.set("general", problem_prompt)

        reason_prompt = "Next, take a more detailed look at the cost function which the planner currently uses:\n"
        reason_prompt += f"{self.separator}{cost_function}\n{self.separator}"
        reason_prompt += "What are the influencing factors and how could they be causing the problem?"
        user_prompt.set("specific", reason_prompt)

        strategy_prompt = "Finally, based on the previously identified reason, determine a strategy on how to solve the problem:\n"
        strategy_prompt += "In case your strategy includes adjusting weights, use natural language terms to describe the change!"
        user_prompt.set("strategy", strategy_prompt)

        advice_prompt = "It is very important, that you remember the following:\n"
        advice_prompt += "If the penalty related to some partial cost factor is *high*, this is due to this cost factor having a *low* weight, not the other way round!"
        user_prompt.set("advice", advice_prompt)
        return user_prompt

    def run(
        self, evaluation: str, cost_function: str, iteration_id: int
    ) -> list[dict[str, str]]:
        # build user prompt
        user_prompt = self.generate_user_prompt(cost_function, evaluation)

        # query the llm
        messages = LLM.get_messages(
            self.system_prompt.__str__(), user_prompt.__str__(), None
        )
        save_dir = os.path.join(self.save_dir, self.gpt_version, "diagnosis")
        result = self.llm.query(messages, nr_iter=iteration_id, save_dir=save_dir)
        if "diagnosis" in result.keys():
            return [result["diagnosis"]]
        else:
            return []


# module that generates a prescription based on diagnosis
class PrescriptionModule(Module):
    def __init__(
        self,
        memory: FewShotMemory,
        prompt_structure: list[str],
        few_shot_amount: int,
        save_dir: str,
        gpt_version: str,
        temperature: float,
    ):
        super().__init__()
        self.prompt_structure = prompt_structure
        self.memory = memory
        self.few_shot_amount = few_shot_amount
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
        llm_function.add_string_parameter(
            "solution",
            "A description of the exact changes which need to be implemented",
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

    def generate_user_prompt(self, cost_function: str, diagnosis: str) -> Prompt:
        user_prompt = Prompt(self.prompt_structure)

        # set reflection prompt
        feedback = ""
        # TODO init feedback
        # if self.feedback[0]:
        #     feedback += self.feedback[0] + "\n"
        # if self.feedback[1]:
        #     feedback += self.feedback[1] + "\n"
        # if feedback:
        #     reflection_prompt = "With regard to your last changes, here is some general advice on what you can do better:\n"
        #     reflection_prompt += f"{self.separator}{feedback}{self.separator}"
        #     user_prompt.set("reflection", reflection_prompt)

        # set memory prompt
        if self.few_shot_amount > 0:
            memories = self.memory.retrieve(
                diagnosis,
                collection_name="prescription",
                n=self.few_shot_amount,
            )
            memory_prompt = "For reference, here are excerpts of changes which you made in similar situations:\n"
            for m in memories:
                memory_prompt += m + "\n"
            user_prompt.set("memory", memory_prompt[:-1])

        # set documentation
        path_to_doc_prompt = os.path.join(
            self.path_to_prompts, "prescription_documentation_prompt.txt"
        )
        with open(path_to_doc_prompt, "r") as file:
            prompt = f"Here is a documentation of all helper functions and libraries:\n{file.read()}"
            user_prompt.set("documentation", prompt)

        # set default prompt parts
        cost_function_prompt = (
            "This is the planner's cost function code which you need to improve:\n"
        )
        cost_function_prompt += f"{self.separator}{cost_function}\n{self.separator}"
        user_prompt.set("cost_function", cost_function_prompt)

        diagnoses_prompt = "And here are some reflections on what could be the problem and how it could be fixed:"
        diagnoses_prompt += f"{self.separator}{diagnosis}\n{self.separator}"
        user_prompt.set("diagnoses", diagnoses_prompt)
        return user_prompt

    def run(
        self, diagnosis: str, cost_function: str, iteration_id: int
    ) -> Tuple[ReactiveMotionPlanner, Union[None, str]]:
        user_prompt = self.generate_user_prompt(cost_function, diagnosis)

        # query the llm
        messages = LLM.get_messages(
            self.system_prompt.__str__(), user_prompt.__str__(), None
        )
        save_dir = os.path.join(self.save_dir, self.gpt_version, "prescription")
        result = self.prescription_llm.query(
            messages,
            nr_iter=iteration_id,
            save_dir=save_dir,
        )
        if "solution" in result.keys():
            solution = result["solution"]
        else:
            solution = None
        if "cost_function" in result.keys():
            cost_function_string = result["cost_function"]
            cost_function_string = textwrap.dedent(cost_function_string)
            return ReactiveMotionPlanner(cost_function_string), solution
        else:
            raise ValueError("The llm did not provide an essential parameter!")


# module that generates a diagnosis based on reactive planner performance
class ReflectionModule(Module):
    def __init__(
        self,
        memory: FewShotMemory,
        prompt_structure: list[str],
        save_dir: str,
        gpt_version: str,
        temperature: float,
    ):
        super().__init__()
        self.prompt_structure = prompt_structure
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
        llm_function.add_string_parameter(
            "analysis", "Reflection on the repair process"
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
        initial_cf: str,
        initial_e: str,
        diagnosis: str,
        final_cf: str,
        final_e: str,
    ) -> Prompt:
        user_prompt = Prompt(self.prompt_structure)
        # set default prompt parts
        initial_prompt = (
            "First of all, here is some information about the initial planner:\n"
        )
        initial_prompt += "It consisted of the following cost function:\n"
        initial_prompt += f"{self.separator}{initial_cf}\n{self.separator}"
        initial_prompt += "Its cost function lead to the following external rating:\n"
        initial_prompt += f"{self.separator}{initial_e}\n{self.separator}"
        user_prompt.set("initial", initial_prompt)

        diagnosis_prompt = "Now here is the diagnosis of the repair-system:\n"
        diagnosis_prompt += f"{self.separator}{diagnosis}\n{self.separator}"
        user_prompt.set("diagnosis", diagnosis_prompt)

        prescription_prompt = (
            "The diagnosis triggered the following adjustments of the cost function:\n"
        )
        prescription_prompt += f"{self.separator}{final_cf}\n{self.separator}"
        prescription_prompt += (
            "The repaired cost function lead to the following external rating:\n"
        )
        prescription_prompt += f"{self.separator}{final_e}\n{self.separator}"
        user_prompt.set("prescription", prescription_prompt)

        with open(
            os.path.join(self.path_to_prompts, "reflection_task_prompt.txt"), "r"
        ) as file:
            task_prompt = file.read()
        user_prompt.set("task", task_prompt)
        return user_prompt

    def run(
        self,
        initial_cf: str,
        initial_e: str,
        diagnosis: str,
        final_cf: str,
        final_e: str,
        iteration_id: int,
    ) -> Tuple[str, str, str]:
        user_prompt = self.generate_user_prompt(
            initial_cf, initial_e, diagnosis, final_cf, final_e
        )

        # query the llm
        messages = LLM.get_messages(
            self.system_prompt.__str__(), user_prompt.__str__(), None
        )
        save_dir = os.path.join(self.save_dir, self.config.gpt_version, "reflection")
        result: dict = self.reflection_llm.query(
            messages,
            nr_iter=iteration_id,
            save_dir=save_dir,
        )
        analysis = ""
        diagnosis_feedback = ""
        prescription_feedback = ""
        if "analysis" in result.keys():
            analysis = result["analysis"]
        if "diagnosis" in result.keys():
            diagnosis_feedback = result["diagnosis"]
        if "prescription" in result.keys():
            prescription_feedback = result["prescription"]
        return analysis, diagnosis_feedback, prescription_feedback
