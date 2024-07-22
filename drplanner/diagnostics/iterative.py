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


@dataclass
class DrPlannerIterationConfiguration:
    documentation = False
    memory = True
    reflect = True
    update_memory = False
    diagnosis_shots = 3
    prescription_shots = 0
    cost_function = CostFunction.SM1
    iterations = 4


# A single DrPlanner iteration which tries to improve a motion planner
class DrPlannerIteration:
    def __init__(
        self,
        motion_planner: ReactiveMotionPlanner,
        config: DrPlannerConfiguration,
        memory: FewShotMemory,
        save_dir: str,
    ):
        self.save_dir = save_dir
        self.iteration_config = DrPlannerIterationConfiguration()
        self.diagnosis_prompt_structure = ["general", "specific", "strategy", "advice"]
        self.prescription_prompt_structure = ["cost_function", "diagnoses"]
        self.reflection_prompt_structure = ["initial", "diagnosis", "prescription", "task"]

        if self.iteration_config.reflect:
            self.diagnosis_prompt_structure.insert(0, "reflection")
            self.prescription_prompt_structure.insert(0, "reflection")
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
        self.diagnosis_llm = LLM(
            self.config.gpt_version,
            self.config.openai_api_key,
            llm_function,
            temperature=self.config.temperature,
            mockup=self.config.mockup_openAI,
        )

        # initialize prescription llm
        llm_function = LLMFunction(custom=True)
        llm_function.add_string_parameter(
            "solution",
            "A description of the exact changes which need to be implemented",
        )
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

        llm_function = LLMFunction(custom=True)
        llm_function.add_string_parameter(
            "analysis", "Reflection on the repair process"
        )
        self.reflection_llm = LLM(
            self.config.gpt_version,
            self.config.openai_api_key,
            llm_function,
            temperature=self.config.temperature,
            mockup=self.config.mockup_openAI,
        )

        self.separator = '"""\n'
        self.feedback: list[str] = ["", "", ""]

    def update_motion_planner(self, motion_planner: ReactiveMotionPlanner):
        self.motion_planner = motion_planner

    def evaluate(self, absolute_scenario_path: str) -> Tuple[str, float]:
        try:
            absolute_config_path = os.path.join(
                self.config.project_path, "planners", "standard-config.yaml"
            )
            cost_result = self.motion_planner.evaluate_on_scenario(
                absolute_scenario_path,
                absolute_config_path,
                cost_type=self.iteration_config.cost_function,
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

        # set reflection prompt
        feedback = ""
        if self.feedback[0]:
            feedback += self.feedback[0] + "\n"
        if self.feedback[1]:
            feedback += self.feedback[1] + "\n"
        if feedback:
            reflection_prompt = "With regard to your last changes, here is some general advice on what you can do better:\n"
            reflection_prompt += f"{self.separator}{feedback}{self.separator}"
            user_prompt.set("reflection", reflection_prompt)

        # set memory prompt
        if self.iteration_config.diagnosis_shots > 0:
            memories = self.memory.retrieve(
                evaluation,
                collection_name="diagnosis",
                n=self.iteration_config.diagnosis_shots,
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
        save_dir = os.path.join(self.save_dir, self.config.gpt_version, "diagnosis")
        result = self.diagnosis_llm.query(
            messages, nr_iter=iteration_id, save_dir=save_dir
        )
        if "diagnosis" in result.keys():
            return [result["diagnosis"]]
        else:
            return []

    def prescribe_user_prompt(self, cost_function: str, diagnoses: str) -> Prompt:
        user_prompt = Prompt(self.prescription_prompt_structure)

        # set reflection prompt
        feedback = ""
        if self.feedback[0]:
            feedback += self.feedback[0] + "\n"
        if self.feedback[2]:
            feedback += self.feedback[2] + "\n"
        if feedback:
            reflection_prompt = "With regard to your last changes, here is some general advice on what you can do better:\n"
            reflection_prompt += f"{self.separator}{feedback}{self.separator}"
            user_prompt.set("reflection", reflection_prompt)

        # set memory prompt
        if self.iteration_config.prescription_shots > 0:
            memories = self.memory.retrieve(
                diagnoses,
                collection_name="prescription",
                n=self.iteration_config.prescription_shots,
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
        diagnoses_prompt += f"{self.separator}{diagnoses}\n{self.separator}"
        user_prompt.set("diagnoses", diagnoses_prompt)
        return user_prompt

    def prescribe(
        self, diagnoses: str, iteration_id: int
    ) -> Tuple[ReactiveMotionPlanner, Union[None, str]]:
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
        save_dir = os.path.join(self.save_dir, self.config.gpt_version, "prescription")
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

    def reflect_user_prompt(
        self,
        initial_cf: str,
        initial_e: str,
        diagnosis: str,
        final_cf: str,
        final_e: str,
    ) -> Prompt:
        user_prompt = Prompt(self.reflection_prompt_structure)
        # set default prompt parts
        initial_prompt = "First of all, here is some information about the initial planner:\n"
        initial_prompt += "It consisted of the following cost function:\n"
        initial_prompt += f"{self.separator}{initial_cf}\n{self.separator}"
        initial_prompt += "Its cost function lead to the following external rating:\n"
        initial_prompt += f"{self.separator}{initial_e}\n{self.separator}"
        user_prompt.set("initial", initial_prompt)

        diagnosis_prompt = "Now here is the diagnosis of the repair-system:\n"
        diagnosis_prompt += f"{self.separator}{diagnosis}\n{self.separator}"
        user_prompt.set("diagnosis", diagnosis_prompt)

        prescription_prompt = "The diagnosis triggered the following adjustments of the cost function:\n"
        prescription_prompt += f"{self.separator}{final_cf}\n{self.separator}"
        prescription_prompt += "The repaired cost function lead to the following external rating:\n"
        prescription_prompt += f"{self.separator}{final_e}\n{self.separator}"
        user_prompt.set("prescription", prescription_prompt)

        with open(os.path.join(self.path_to_prompts, "reflection_task_prompt.txt"), "r") as file:
            task_prompt = file.read()
        user_prompt.set("task", task_prompt)
        return user_prompt

    def reflect(
        self,
        initial_cf: str,
        initial_e: str,
        diagnosis: str,
        final_cf: str,
        final_e: str,
        iteration_id: int,
    ):
        # build system prompt
        system_prompt = Prompt(["base"])
        path_to_system_prompt = os.path.join(
            self.path_to_prompts,
            "reflection_system_prompt.txt",
        )
        with open(path_to_system_prompt, "r") as file:
            system_prompt.set("base", file.read())

        # build user prompt
        user_prompt = self.reflect_user_prompt(initial_cf, initial_e, diagnosis, final_cf, final_e)

        # query the llm
        messages = LLM.get_messages(
            system_prompt.__str__(), user_prompt.__str__(), None
        )
        save_dir = os.path.join(self.save_dir, self.config.gpt_version, "reflection")
        result: dict = self.reflection_llm.query(
            messages,
            nr_iter=iteration_id,
            save_dir=save_dir,
        )
        if "analysis" in result.keys():
            self.feedback[0] = result["analysis"]
        if "diagnosis" in result.keys():
            self.feedback[1] = result["diagnosis"]
        if "prescription" in result.keys():
            self.feedback[2] = result["prescription"]

    @staticmethod
    def improved(
        start_cost: float, end_cost: float, min_improvement: float = 0.05
    ) -> bool:
        if not end_cost < math.inf:
            improvement = -1.0
        elif not start_cost < math.inf:
            improvement = 1.0
        else:
            improvement = 1.0 - (end_cost / start_cost)

        return improvement > min_improvement

    def run(
        self,
        absolute_scenario_path: str,
        start_evaluation: str,
        start_total_cost: float,
        iteration_id: int = 0,
    ) -> Tuple[str, str, float, list[dict[str, str]]]:
        start_cost_function = self.motion_planner.cost_function_string
        # let a llm analyze the reasons for this performance
        diagnosis = self.diagnose(start_evaluation, iteration_id)[0]
        # turn diagnosis into string
        diagnosis_string = ""
        for key, value in diagnosis.items():
            if value.lower().startswith(key):
                diagnosis_string += value + "\n"
            else:
                diagnosis_string += f"{key}: {value}\n"

        # let a llm repair the planner given the diagnosis
        self.motion_planner, solution = self.prescribe(diagnosis_string, iteration_id)
        # run the repaired version and evaluate it
        end_cf = self.motion_planner.cost_function_string
        end_evaluation, end_total_cost = self.evaluate(absolute_scenario_path)
        # reflect on the changes
        if self.iteration_config.reflect:
            self.reflect(start_cost_function, start_evaluation, diagnosis_string, end_cf, end_evaluation, iteration_id)
        # if the iteration improved the planner, add its insight to memory
        if self.iteration_config.update_memory and self.improved(
            start_total_cost, end_total_cost
        ):
            self.memory.insert(
                start_evaluation, diagnosis_string, collection_name="diagnosis"
            )
            if solution:
                self.memory.insert(
                    diagnosis_string, solution, collection_name="prescription"
                )

        return end_cf, end_evaluation, end_total_cost, [diagnosis]


def run_iteration(
    scenario_path: str = "USA_Lanker-2_19_T-1", config: DrPlannerConfiguration = None
) -> list[float]:
    if not config:
        config = DrPlannerConfiguration()

    if not scenario_path.endswith(".xml"):
        scenario_id = scenario_path
        scenario_path = os.path.join(
            os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
        )
    else:
        scenario_id = os.path.basename(scenario_path)[:-4]

    # noinspection PyTypeChecker
    motion_planner = ReactiveMotionPlanner(None)
    memory = FewShotMemory()
    save_dir = os.path.join(config.save_dir, scenario_id)
    dr_planner = DrPlannerIteration(motion_planner, config, memory, save_dir)
    # first run the initial planner once to obtain the initial values for the loop:
    cf0 = motion_planner.cost_function_string
    e0, tc0 = dr_planner.evaluate(scenario_path)
    cost_functions = [cf0]
    evaluations = [e0]
    cost_results = [tc0]
    diagnoses = []
    iteration_id = 0
    print(e0)

    while iteration_id < dr_planner.iteration_config.iterations:
        dr_planner.update_motion_planner(ReactiveMotionPlanner(cost_functions.pop()))
        cf, e, tc, d = dr_planner.run(
            scenario_path,
            evaluations.pop(),
            cost_results[-1],
            iteration_id=iteration_id,
        )
        print(e)
        cost_functions.append(cf)
        evaluations.append(e)
        cost_results.append(tc)
        diagnoses.append(d)
        iteration_id += 1
    return cost_results


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


def run(
    scenario_path: str = "USA_Lanker-2_19_T-1", config: DrPlannerConfiguration = None
):
    if not config:
        config = DrPlannerConfiguration()

    if not scenario_path.endswith(".xml"):
        scenario_path = os.path.join(
            os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
        )

    explorer_llm, system_prompt, user_prompt = get_explorer_llm(config)
    cost_function_variants: list[str] = []
    messages = LLM.get_messages(system_prompt.__str__(), user_prompt.__str__(), None)
    temp = os.path.join(config.save_dir, config.gpt_version, "explorer")
    query = explorer_llm.query(messages, save_dir=str(temp))
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
        dr_planner = DrPlannerIteration(planner, config, memory, config.save_dir)
        evaluation_string, start_total_cost = dr_planner.evaluate(scenario_path)
        print(f"Total initial cost of variant nr.{i}: {start_total_cost}")
        end_cf, _, end_total_cost, _ = dr_planner.run(
            scenario_path, evaluation_string, start_total_cost, iteration_id=i
        )
        print(f"Total final cost of variant nr.{i}: {end_total_cost}")
        results.append((end_cf, end_total_cost))

    top_result = min(results, key=lambda x: x[1])
    print(f"Top result with total cost of {top_result[1]}:\n{top_result[0]}")
