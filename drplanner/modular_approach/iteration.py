import itertools
import math
import os
import textwrap
from typing import Tuple, Union

from drplanner.prompter.llm import LLM, LLMFunction

from drplanner.prompter.base import PrompterBase, Prompt

from drplanner.describer.trajectory_description import TrajectoryCostDescription

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.planners.reactive_planner import ReactiveMotionPlanner
from drplanner.memory.memory import FewShotMemory

from dataclasses import dataclass
from commonroad.common.solution import CostFunction

from modular_approach.module import (
    EvaluationModule,
    DiagnosisModule,
    PrescriptionModule,
    ReflectionModule,
)


@dataclass
class DrPlannerIterationConfiguration:
    documentation = False
    memory = True
    reflect = True
    update_memory = False
    diagnosis_shots = 2
    prescription_shots = 0
    cost_function = CostFunction.SM1
    iterations = 4


class Reflection:
    def __init__(
        self,
        general: Union[str, None],
        diagnosis: Union[str, None],
        prescription: Union[str, None],
    ):
        self.general = general
        self.diagnosis = diagnosis
        self.prescription = prescription


class Iteration:
    def __init__(self, memory: FewShotMemory, save_dir: str):
        self.memory = memory
        self.save_dir = save_dir
        self.general_config = DrPlannerConfiguration()
        self.iteration_config = DrPlannerIterationConfiguration()
        diagnosis_prompt_structure = ["general", "specific", "strategy", "advice"]
        prescription_prompt_structure = ["cost_function", "diagnoses"]
        reflection_prompt_structure = ["initial", "diagnosis", "prescription", "task"]

        if self.iteration_config.reflect:
            diagnosis_prompt_structure.insert(0, "reflection")
            prescription_prompt_structure.insert(0, "reflection")
        if self.iteration_config.documentation:
            diagnosis_prompt_structure.insert(0, "documentation")
            prescription_prompt_structure.insert(0, "documentation")
        if self.iteration_config.memory:
            diagnosis_prompt_structure.insert(0, "memory")
            prescription_prompt_structure.insert(0, "memory")

        self.evaluation_module = EvaluationModule()
        self.diagnosis_module = DiagnosisModule(
            memory,
            diagnosis_prompt_structure,
            self.iteration_config.diagnosis_shots,
            self.save_dir,
            self.general_config.gpt_version,
            self.general_config.temperature,
        )
        self.prescription_module = PrescriptionModule(
            memory,
            prescription_prompt_structure,
            self.iteration_config.prescription_shots,
            self.save_dir,
            self.general_config.gpt_version,
            self.general_config.temperature,
        )
        self.reflection_module = ReflectionModule(
            memory,
            reflection_prompt_structure,
            self.save_dir,
            self.general_config.gpt_version,
            self.general_config.temperature,
        )

    @staticmethod
    def diagnoses_array_to_str(diagnoses_array: list[dict[str, str]]):
        # turn diagnosis into string
        diagnosis = diagnoses_array[0]
        diagnosis_string = ""
        for key, value in diagnosis.items():
            if value.lower().startswith(key):
                diagnosis_string += value + "\n"
            else:
                diagnosis_string += f"{key}: {value}\n"
        return diagnosis_string

    @staticmethod
    def improved(
        initial_cost: float, final_cost: float, min_improvement: float = 0.05
    ) -> bool:
        if not final_cost < math.inf:
            improvement = -1.0
        elif not initial_cost < math.inf:
            improvement = 1.0
        else:
            improvement = 1.0 - (final_cost / initial_cost)

        return improvement > min_improvement

    def run(
        self,
        absolute_scenario_path: str,
        motion_planner: ReactiveMotionPlanner,
        initial_evaluation: str,
        initial_total_cost: float,
        previous_reflection: Reflection,
        iteration_id: int,
    ):
        initial_cost_function = motion_planner.cost_function_string
        # generate diagnosis
        diagnoses_array = self.diagnosis_module.run(
            initial_evaluation, initial_cost_function, iteration_id
        )
        # format diagnosis
        diagnosis = self.diagnoses_array_to_str(diagnoses_array)
        # repair the planner
        repaired_motion_planner, description = self.prescription_module.run(
            diagnosis, initial_cost_function, iteration_id
        )
        repaired_cost_function = repaired_motion_planner.cost_function_string
        # evaluate the repair
        repair_evaluation, repair_total_cost = self.evaluation_module.run(
            absolute_scenario_path, repaired_motion_planner
        )
        # reflect on the repair
        (
            analysis,
            diagnosis_reflection,
            prescription_reflection,
        ) = self.reflection_module.run(
            initial_cost_function,
            initial_evaluation,
            diagnosis,
            repaired_cost_function,
            repair_evaluation,
            iteration_id,
        )
        reflection = Reflection(analysis, diagnosis_reflection, prescription_reflection)
        # if the iteration was successful, add its insights to memory
        if self.iteration_config.update_memory and self.improved(
            initial_total_cost, repair_total_cost
        ):
            self.memory.insert(
                initial_evaluation, diagnosis, collection_name="diagnosis"
            )
            if description:
                self.memory.insert(
                    diagnosis, description, collection_name="prescription"
                )

        return repair_evaluation, repair_total_cost, repaired_motion_planner, reflection


def run_iterative_repair(
    scenario_path: str = "USA_Lanker-2_19_T-1",
    config: DrPlannerConfiguration = None,
    templates: list[str] = None,
    motion_planner: ReactiveMotionPlanner = None,
) -> list[float]:
    if not config:
        config = DrPlannerConfiguration()

    if not templates:
        templates = load_templates()

    if not scenario_path.endswith(".xml"):
        scenario_id = scenario_path
        scenario_path = os.path.join(
            os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
        )
    else:
        scenario_id = os.path.basename(scenario_path)[:-4]

    memory = FewShotMemory()
    save_dir = os.path.join(config.save_dir, scenario_id)
    iteration = Iteration(memory, save_dir)

    if not motion_planner:
        motion_planner = determine_template_cost_function(
            scenario_path, templates, iteration
        )

    reflection = Reflection("", "", "")
    # first run the initial planner once to obtain the initial values for the loop:
    cf0 = motion_planner.cost_function_string
    e0, tc0 = iteration.evaluation_module.run(scenario_path, motion_planner)
    # collect all interesting data produced throughout the loop
    cost_functions = [cf0]
    evaluations = [e0]
    cost_results = [tc0]
    iteration_id = 0
    print(e0)

    while iteration_id < iteration.iteration_config.iterations:
        e, tc, motion_planner, reflection = iteration.run(
            scenario_path,
            motion_planner,
            evaluations.pop(),
            cost_results[-1],
            reflection,
            iteration_id=iteration_id,
        )
        print(e)
        cost_functions.append(motion_planner.cost_function_string)
        evaluations.append(e)
        cost_results.append(tc)
        iteration_id += 1
    return cost_results


def load_templates() -> list[str]:
    filenames = [
        "DEU_Frankfurt-191_12_I-1.cr.txt",
        "DEU_Frankfurt-11_8_I-1.cr.txt",
        "DEU_Lohmar-34_1_I-1-1.cr.txt",
        "DEU_Muc-19_1_I-1-1.cr.txt",
        "DEU_Frankfurt-95_9_I-1.cr.txt",
        "ESP_Mad-1_8_I-1-1.cr.txt",
    ]
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    templates = []
    for name in filenames:
        with open(os.path.join(folder_path, name), "r") as file:
            templates.append(file.read())
    return templates


def determine_template_cost_function(
    absolute_scenario_path: str, templates: list[str], iteration: Iteration
) -> ReactiveMotionPlanner:
    if len(templates) <= 0:
        raise ValueError("no templates provided")

    best_score = math.inf
    best_planner = None
    for template in templates:
        planner = ReactiveMotionPlanner(template)
        _, score = iteration.evaluation_module.run(absolute_scenario_path, planner)
        if score < best_score:
            best_score = score
            best_planner = planner

    return best_planner


# def run_iteration(
#         scenario_path: str = "USA_Lanker-2_19_T-1", config: DrPlannerConfiguration = None
# ) -> list[float]:
#     if not config:
#         config = DrPlannerConfiguration()
#
#     if not scenario_path.endswith(".xml"):
#         scenario_id = scenario_path
#         scenario_path = os.path.join(
#             os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
#         )
#     else:
#         scenario_id = os.path.basename(scenario_path)[:-4]
#
#     # noinspection PyTypeChecker
#     motion_planner = ReactiveMotionPlanner(None)
#     memory = FewShotMemory()
#     save_dir = os.path.join(config.save_dir, scenario_id)
#     dr_planner = DrPlannerIteration(motion_planner, config, memory, save_dir)
#     # first run the initial planner once to obtain the initial values for the loop:
#     cf0 = motion_planner.cost_function_string
#     e0, tc0 = dr_planner.evaluate(scenario_path)
#     cost_functions = [cf0]
#     evaluations = [e0]
#     cost_results = [tc0]
#     diagnoses = []
#     iteration_id = 0
#     print(e0)
#
#     while iteration_id < dr_planner.iteration_config.iterations:
#         dr_planner.update_motion_planner(ReactiveMotionPlanner(cost_functions.pop()))
#         cf, e, tc, d = dr_planner.run(
#             scenario_path,
#             evaluations.pop(),
#             cost_results[-1],
#             iteration_id=iteration_id,
#         )
#         print(e)
#         cost_functions.append(cf)
#         evaluations.append(e)
#         cost_results.append(tc)
#         diagnoses.append(d)
#         iteration_id += 1
#     return cost_results
#
#
# def get_explorer_llm(config: DrPlannerConfiguration) -> Tuple[LLM, Prompt, Prompt]:
#     llm_function = LLMFunction(custom=True)
#     llm_function.add_code_parameter("variant_1", "First cost function variant")
#     llm_function.add_code_parameter("variant_2", "Second cost function variant")
#     llm_function.add_code_parameter("variant_3", "Third cost function variant")
#     llm_function.add_code_parameter("variant_4", "Forth cost function variant")
#     llm = LLM(
#         config.gpt_version,
#         config.openai_api_key,
#         llm_function,
#         temperature=config.temperature,
#         mockup=config.mockup_openAI,
#     )
#     path_to_prompts = os.path.join(
#         config.project_path,
#         "prompter",
#         "reactive-planner",
#         "prompts",
#     )
#     with open(os.path.join(path_to_prompts, "explorer_system_prompt.txt"), "r") as file:
#         system_prompt = Prompt(["base"])
#         system_prompt.set("base", file.read())
#     with open(os.path.join(path_to_prompts, "explorer_user_prompt.txt"), "r") as file:
#         user_prompt = Prompt(["base"])
#         user_prompt.set("base", file.read())
#     return llm, system_prompt, user_prompt
#
#
# def run(
#         scenario_path: str = "USA_Lanker-2_19_T-1", config: DrPlannerConfiguration = None
# ):
#     if not config:
#         config = DrPlannerConfiguration()
#
#     if not scenario_path.endswith(".xml"):
#         scenario_path = os.path.join(
#             os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
#         )
#
#     explorer_llm, system_prompt, user_prompt = get_explorer_llm(config)
#     cost_function_variants: list[str] = []
#     messages = LLM.get_messages(system_prompt.__str__(), user_prompt.__str__(), None)
#     temp = os.path.join(config.save_dir, config.gpt_version, "explorer")
#     query = explorer_llm.query(messages, save_dir=str(temp))
#     counter = 1
#     while counter > 0:
#         key = f"variant_{counter}"
#         if key in query.keys():
#             cost_function_variants.append(query[key])
#             counter += 1
#         else:
#             counter = 0
#
#     memory = FewShotMemory()
#     results: list[Tuple[str, float]] = []
#     signature = "def evaluate(self, trajectory: TrajectorySample) -> float:"
#     for i, variant in enumerate(cost_function_variants):
#         if not variant.startswith(signature):
#             lines = variant.split("\n")
#             lines[0] = signature
#             variant = "\n".join(lines)
#         planner = ReactiveMotionPlanner(variant)
#         dr_planner = DrPlannerIteration(planner, config, memory, config.save_dir)
#         evaluation_string, start_total_cost = dr_planner.evaluate(scenario_path)
#         print(f"Total initial cost of variant nr.{i}: {start_total_cost}")
#         end_cf, _, end_total_cost, _ = dr_planner.run(
#             scenario_path, evaluation_string, start_total_cost, iteration_id=i
#         )
#         print(f"Total final cost of variant nr.{i}: {end_total_cost}")
#         results.append((end_cf, end_total_cost))
#
#     top_result = min(results, key=lambda x: x[1])
#     print(f"Top result with total cost of {top_result[1]}:\n{top_result[0]}")
