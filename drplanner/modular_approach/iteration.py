import math
import os

from commonroad_dc.costs.evaluation import PlanningProblemCostResult

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.planners.reactive_planner import ReactiveMotionPlanner
from drplanner.memory.memory import FewShotMemory

from drplanner.modular_approach.diagnosis import DiagnosisModule
from drplanner.modular_approach.module import EvaluationModule, Reflection
from drplanner.modular_approach.prescription import PrescriptionModule
from drplanner.modular_approach.reflection import ReflectionModule


class Iteration:
    def __init__(self, config: DrPlannerConfiguration, memory: FewShotMemory, save_dir: str):
        self.memory = memory
        self.save_dir = save_dir
        self.config = config

        self.evaluation_module = EvaluationModule(config)
        self.diagnosis_module = DiagnosisModule(
            config,
            self.save_dir,
            self.config.gpt_version,
            self.config.temperature,
        )
        self.prescription_module = PrescriptionModule(
            config,
            self.save_dir,
            self.config.gpt_version,
            0.2,
        )
        self.reflection_module = ReflectionModule(
            config,
            self.save_dir,
            self.config.gpt_version,
            self.config.temperature,
        )

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
        initial_cost_result: PlanningProblemCostResult,
        previous_reflections: list[Reflection],
        iteration_id: int,
    ):
        last_reflection = previous_reflections[-1]
        initial_cost_function = motion_planner.cost_function_string
        # retrieve few_shots
        if self.config.memory_module:
            few_shots = self.memory.get_few_shots(initial_evaluation, self.evaluation_module.path_to_plot, self.config.n_shots)
        else:
            few_shots = []
        # generate diagnosis
        try:
            diagnosis, max_time_steps = self.diagnosis_module.run(
                initial_evaluation, initial_cost_function, last_reflection, few_shots, iteration_id
            )
        except ValueError as _:
            print("No diagnosis provided")
            diagnosis = None
            max_time_steps = motion_planner.max_time_steps

        # repair the planner
        try:
            repaired_motion_planner = self.prescription_module.run(
                diagnosis.__str__(), initial_cost_function, last_reflection, few_shots, iteration_id
            )
        except ValueError as _:
            print("No repair provided")
            repaired_motion_planner = motion_planner

        repaired_motion_planner.max_time_steps = max_time_steps
        # evaluate the repair
        repair_evaluation, repair_cost_result = self.evaluation_module.run(
            absolute_scenario_path, repaired_motion_planner
        )

        # reflect on the repair
        try:
            reflection = self.reflection_module.run(
                initial_cost_result,
                repair_cost_result,
                diagnosis.__str__(),
                repaired_motion_planner.__str__(),
                previous_reflections,
                iteration_id,
            )
        except ValueError as _:
            print("No reflection provided")
            reflection = Reflection("")

        if self.config.update_memory_module and self.improved(
            initial_cost_result.total_costs, repair_cost_result.total_costs
        ):
            self.memory.insert(
                diagnosis.to_few_shot(),
                repaired_motion_planner.cost_function_string,
                repair_cost_result.total_costs,
                initial_evaluation,
                self.evaluation_module.path_to_plot
            )

        return repair_evaluation, repair_cost_result, repaired_motion_planner, reflection


def run_iterative_repair(
    scenario_path: str = "USA_Lanker-2_19_T-1",
    config: DrPlannerConfiguration = None,
    motion_planner: ReactiveMotionPlanner = None,
) -> list[PlanningProblemCostResult]:
    if not config:
        config = DrPlannerConfiguration()

    if not scenario_path.endswith(".xml"):
        scenario_id = scenario_path
        scenario_path = os.path.join(
            os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
        )
    else:
        scenario_id = os.path.basename(scenario_path)[:-4]

    memory = FewShotMemory()
    save_dir = os.path.join(config.save_dir, scenario_id)
    iteration = Iteration(config, memory, save_dir)

    if not motion_planner:
        motion_planner = ReactiveMotionPlanner(None, None, None)

    reflection = Reflection("")
    # first run the initial planner once to obtain the initial values for the loop:
    cf0 = motion_planner.cost_function_string
    e0, cr0 = iteration.evaluation_module.run(scenario_path, motion_planner)
    # collect all interesting data produced throughout the loop
    cost_functions = [cf0]
    evaluations = [e0]
    cost_results = [cr0]
    reflections = [reflection]
    iteration_id = 0
    print(e0)

    while iteration_id < config.iteration_max:
        e, tc, motion_planner, new_reflection = iteration.run(
            scenario_path,
            motion_planner,
            evaluations.pop(),
            cost_results[-1],
            reflections,
            iteration_id=iteration_id,
        )
        print(e)
        cost_functions.append(motion_planner.cost_function_string)
        evaluations.append(e)
        cost_results.append(tc)
        reflections.append(new_reflection)
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
        planner = ReactiveMotionPlanner(template, [], None)
        _, score = iteration.evaluation_module.run(absolute_scenario_path, planner, plot=False)
        if score.total_costs < best_score:
            best_score = score.total_costs
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
