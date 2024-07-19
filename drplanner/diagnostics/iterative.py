import math
import os
import textwrap
from typing import Tuple

from drplanner.prompter.llm import LLM, LLMFunction

from drplanner.prompter.base import PrompterBase, Prompt

from drplanner.describer.trajectory_description import TrajectoryCostDescription

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.planners.reactive_planner import ReactiveMotionPlanner
from drplanner.memory.memory import FewShotMemory

from dataclasses import dataclass
from commonroad.common.solution import CostFunction


@dataclass
class DrPlannerIterationConfiguration:
    documentation = False
    memory = True
    diagnosis_shots = 2
    prescription_shots = 3
    cost_function = CostFunction.SM1
    iterations = 4


# A single DrPlanner iteration which tries to improve a motion planner
class DrPlannerIteration:
    def __init__(
        self,
        motion_planner: ReactiveMotionPlanner,
        config: DrPlannerConfiguration,
        memory: FewShotMemory,
    ):
        self.iteration_config = DrPlannerIterationConfiguration()
        self.diagnosis_prompt_structure = ["cost_function", "evaluation", "advice"]
        self.prescription_prompt_structure = ["cost_function", "diagnoses"]

        if self.iteration_config.documentation:
            self.diagnosis_prompt_structure.insert(0, "documentation")
            self.prescription_prompt_structure.insert(0, "documentation")
        if self.iteration_config.memory:
            self.diagnosis_prompt_structure.insert(0, "memory")
            self.prescription_prompt_structure.insert(0, "memory")

        self.motion_planner = motion_planner
        self.config = config
        self.memory = memory
        self.path_to_prompts = os.path.join(
            self.config.project_path,
            "prompter",
            "reactive-planner",
            "iterative",
        )

        # initialize diagnosis llm
        llm_function = LLMFunction(custom=True)
        diagnosis_structure = {
            "problem": LLMFunction.get_string_parameter("The motion planner's problem"),
            "reason": LLMFunction.get_string_parameter("The reason for the problem"),
            "approach": LLMFunction.get_string_parameter(
                "The approach for solving the problem"
            )
        }
        items = LLMFunction.get_object_parameter(diagnosis_structure)
        llm_function.add_array_parameter(
            "diagnoses", "One or several diagnoses of a motion planner", items
        )
        self.diagnosis_llm = LLM(
            self.config.gpt_version,
            self.config.openai_api_key,
            llm_function,
            temperature=self.config.temperature,
            mockup=self.config.mockup_openAI,
        )

        # initialize prescription llm
        llm_function = LLMFunction(custom=True)
        llm_function.add_string_parameter("solution", "A description of the exact changes which need to be implemented")
        llm_function.add_code_parameter(
            "cost_function", "The improved cost function of the motion planner"
        )
        self.prescription_llm = LLM(
            self.config.gpt_version,
            self.config.openai_api_key,
            llm_function,
            temperature=self.config.temperature,
            mockup=self.config.mockup_openAI,
        )

    def update_motion_planner(self, motion_planner: ReactiveMotionPlanner):
        self.motion_planner = motion_planner

    def evaluate(self, absolute_scenario_path: str) -> Tuple[str, float]:
        try:
            absolute_config_path = os.path.join(
                self.config.project_path, "planners", "standard-config.yaml"
            )
            cost_result = self.motion_planner.evaluate_on_scenario(
                absolute_scenario_path, absolute_config_path, cost_type=self.iteration_config.cost_function
            )
            evaluation = TrajectoryCostDescription().generate(
                cost_result, self.config.desired_cost
            )
            total_cost = cost_result.total_costs
        except Exception as e:
            evaluation = PrompterBase.generate_exception_description(e)
            total_cost = math.inf
        return evaluation, total_cost

    def diagnose_user_prompt(self, cost_function: str, evaluation: str) -> Prompt:
        user_prompt = Prompt(self.diagnosis_prompt_structure)

        # set memory prompt
        memories = self.memory.retrieve(evaluation, collection_name="diagnosis", n=self.iteration_config.diagnosis_shots)
        memory_prompt = (
            "Here are some old diagnoses which you made in similar situations:\n"
        )
        for m in memories:
            memory_prompt += m + "\n"
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
        user_prompt.set(
            "cost_function", f"This is the planner's cost function code:\n{cost_function}"
        )
        user_prompt.set(
            "evaluation",
            f"And this is what happens if you run the planner:\n{evaluation}",
        )
        advice_prompt = "It is very important, that you remember the following:\n"
        advice_prompt += "If the penalty related to some partial cost factor is *high*, this is due to this cost factor having a *low* weight, not the other way round!"
        user_prompt.set(
            "advice",
            advice_prompt
        )
        return user_prompt

    def diagnose(self, evaluation: str, iteration_id: int) -> list[dict[str, str]]:
        cost_function = self.motion_planner.cost_function_string

        # build system prompt
        system_prompt = Prompt(["base"])
        path_to_system_prompt = os.path.join(
            self.path_to_prompts,
            "diagnosis_system_prompt.txt",
        )
        with open(path_to_system_prompt, "r") as file:
            system_prompt.set("base", file.read())

        # build user prompt
        user_prompt = self.diagnose_user_prompt(cost_function, evaluation)

        # query the llm
        messages = LLM.get_messages(
            system_prompt.__str__(), user_prompt.__str__(), None
        )
        result = self.diagnosis_llm.query(
            messages, planner_id="diagnose", nr_iter=iteration_id
        )
        if "diagnoses" in result.keys():
            return result["diagnoses"]
        else:
            return []

    def prescribe_user_prompt(
        self, cost_function: str, diagnoses: list[dict]
    ) -> Prompt:
        user_prompt = Prompt(self.prescription_prompt_structure)

        # set memory prompt
        memories = self.memory.retrieve(diagnoses, collection_name="prescription", n=self.iteration_config.prescription_shots)
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
        user_prompt.set(
            "cost_function", f"This is the planner's cost function code which you need to improve/rewrite:\n{cost_function}"
        )
        diagnoses_string = ""
        for diagnosis in diagnoses:
            for key, value in diagnosis.items():
                if value.lower().startswith(key):
                    diagnoses_string += value + "\n"
                else:
                    diagnoses_string += f"{key}: {value}\n"
        user_prompt.set(
            "diagnoses",
            f"And here are some reflections on what could be the problem and how it could be fixed:\n{diagnoses_string[:-2]}",
        )
        return user_prompt

    def prescribe(
        self, diagnoses: list[dict], iteration_id: int
    ) -> ReactiveMotionPlanner:
        cost_function = self.motion_planner.cost_function_string

        # build system prompt
        system_prompt = Prompt(["base"])
        path_to_system_prompt = os.path.join(
            self.path_to_prompts,
            "prescription_system_prompt.txt",
        )
        with open(path_to_system_prompt, "r") as file:
            system_prompt.set("base", file.read())

        # build user prompt
        user_prompt = self.prescribe_user_prompt(cost_function, diagnoses)

        # query the llm
        messages = LLM.get_messages(
            system_prompt.__str__(), user_prompt.__str__(), None
        )
        result = self.prescription_llm.query(
            messages, planner_id="prescribe", nr_iter=iteration_id
        )
        if "cost_function" in result.keys():
            cost_function_string = result["cost_function"]
            cost_function_string = textwrap.dedent(cost_function_string)
            return ReactiveMotionPlanner(cost_function_string)
        else:
            raise ValueError("The llm did not provide an essential parameter!")

    def run(
        self, absolute_scenario_path: str, start_evaluation: str, iteration_id: int = 0
    ) -> Tuple[str, str, float, list[dict[str, str]]]:
        print(start_evaluation)
        # second let a llm analyze the reasons for this performance
        diagnosis = self.diagnose(start_evaluation, iteration_id)
        # third let a llm repair the planner given the diagnosis
        self.motion_planner = self.prescribe(diagnosis, iteration_id)
        # last run the repaired version and evaluate it
        end_cf = self.motion_planner.cost_function_string
        end_evaluation, end_total_cost = self.evaluate(absolute_scenario_path)
        return end_cf, end_evaluation, end_total_cost, diagnosis


def run_iteration(
        scenario_path: str = "USA_Lanker-2_19_T-1", config: DrPlannerConfiguration = None
) -> Tuple[list[float], DrPlannerConfiguration]:
    if not config:
        config = DrPlannerConfiguration()

    if not scenario_path.endswith(".xml"):
        scenario_path = os.path.join(
            os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
        )
    # noinspection PyTypeChecker
    motion_planner = ReactiveMotionPlanner(None)
    memory = FewShotMemory()
    dr_planner = DrPlannerIteration(
        motion_planner, config, memory
    )
    # first run the initial planner once to obtain the initial values for the loop:
    cf0 = motion_planner.cost_function_string
    e0, tc0 = dr_planner.evaluate(scenario_path)
    cost_functions = [cf0]
    evaluations = [e0]
    cost_results = [tc0]
    diagnoses = []
    iteration_id = 0

    while iteration_id < dr_planner.iteration_config.iterations:
        dr_planner.update_motion_planner(ReactiveMotionPlanner(cost_functions.pop()))
        cf, e, tc, d = dr_planner.run(
            scenario_path, evaluations.pop(), iteration_id=iteration_id
        )
        cost_functions.append(cf)
        evaluations.append(e)
        cost_results.append(tc)
        diagnoses.append(d)
        iteration_id += 1
    return cost_results, config


def get_explorer_llm(config: DrPlannerConfiguration) -> Tuple[LLM, Prompt, Prompt]:
    llm_function = LLMFunction(custom=True)
    llm_function.add_code_parameter("variant_1", "First cost function variant")
    llm_function.add_code_parameter("variant_2", "Second cost function variant")
    llm_function.add_code_parameter("variant_3", "Third cost function variant")
    llm_function.add_code_parameter("variant_4", "Forth cost function variant")
    llm = LLM(
        config.gpt_version,
        config.openai_api_key,
        llm_function,
        temperature=config.temperature,
        mockup=config.mockup_openAI,
    )
    path_to_prompts = os.path.join(
        config.project_path,
        "prompter",
        "reactive-planner",
        "iterative",
    )
    with open(os.path.join(path_to_prompts, "explorer_system_prompt.txt"), "r") as file:
        system_prompt = Prompt(["base"])
        system_prompt.set("base", file.read())
    with open(os.path.join(path_to_prompts, "explorer_user_prompt.txt"), "r") as file:
        user_prompt = Prompt(["base"])
        user_prompt.set("base", file.read())
    return llm, system_prompt, user_prompt


def run(scenario_path: str = "USA_Lanker-2_19_T-1"):
    config = DrPlannerConfiguration()
    if not scenario_path.endswith(".xml"):
        scenario_path = os.path.join(
            os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
        )

    explorer_llm, system_prompt, user_prompt = get_explorer_llm(config)
    cost_function_variants: list[str] = []
    messages = LLM.get_messages(system_prompt.__str__(), user_prompt.__str__(), None)
    query = explorer_llm.query(messages, planner_id="explorer")
    counter = 1
    while counter > 0:
        key = f"variant_{counter}"
        if key in query.keys():
            cost_function_variants.append(query[key])
            counter += 1
        else:
            counter = 0

    memory = FewShotMemory()
    results: list[Tuple[str, float]] = []
    signature = "def evaluate(self, trajectory: TrajectorySample) -> float:"
    for i, variant in enumerate(cost_function_variants):
        if not variant.startswith(signature):
            lines = variant.split("\n")
            lines[0] = signature
            variant = "\n".join(lines)
        planner = ReactiveMotionPlanner(variant)
        dr_planner = DrPlannerIteration(planner, config, memory)
        evaluation_string, start_total_cost = dr_planner.evaluate(scenario_path)
        print(f"Total initial cost of variant nr.{i}: {start_total_cost}")
        end_cf, _, end_total_cost, _ = dr_planner.run(scenario_path, evaluation_string, iteration_id=i)
        print(f"Total final cost of variant nr.{i}: {end_total_cost}")
        results.append((end_cf, end_total_cost))

    top_result = min(results, key=lambda x: x[1])
    print(f"Top result with total cost of {top_result[1]}:\n{top_result[0]}")
