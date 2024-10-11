import math
import os
import time
from typing import Tuple

from commonroad_dc.costs.evaluation import PlanningProblemCostResult

from drplanner.utils.config import DrPlannerConfiguration
from drplanner.utils.general import Statistics

from drplanner.planners.reactive_planner_wrapper import ReactiveMotionPlannerWrapper
from drplanner.memory.memory import FewShotMemory

from drplanner.modular_approach.diagnosis import DiagnosisModule
from drplanner.modular_approach.module import EvaluationModule, Reflection, Diagnosis
from drplanner.modular_approach.prescription import PrescriptionModule
from drplanner.modular_approach.reflection import ReflectionModule


class Iteration:
    def __init__(
        self,
        config: DrPlannerConfiguration,
        statistic: Statistics,
        memory: FewShotMemory,
        save_dir: str,
    ):
        self.memory = memory
        self.last_memory_few_shots = []
        self.statistic = statistic
        self.save_dir = save_dir
        self.config = config

        self.evaluation_module = EvaluationModule(config, statistic)
        self.diagnosis_module = DiagnosisModule(
            config,
            statistic,
            self.save_dir,
            self.config.gpt_version,
            self.config.temperature,
        )
        self.prescription_module = PrescriptionModule(
            config,
            statistic,
            self.save_dir,
            self.config.gpt_version,
            0.6,  # hardcoded optimal temperature
        )
        if self.config.reflection_module:
            self.reflection_module = ReflectionModule(
                config,
                statistic,
                self.save_dir,
                self.config.gpt_version,
                self.config.temperature,
            )

    @staticmethod
    def get_relative_improvement(initial_cost: float, final_cost: float) -> float:
        if not final_cost < math.inf:
            return -1.0
        elif not initial_cost < math.inf:
            return 0.1  # keep low to avoid irreplaceable memories
        else:
            return 1.0 - (final_cost / initial_cost)

    def run(
        self,
        absolute_scenario_path: str,
        motion_planner: ReactiveMotionPlannerWrapper,
        initial_evaluation: str,
        initial_cost_result: PlanningProblemCostResult,
        previous_reflections: list[Reflection],
        iteration_id: int,
        start_total_cost: float,
    ) -> Tuple[
        str, PlanningProblemCostResult, ReactiveMotionPlannerWrapper, Reflection
    ]:
        """
        A single modular DrPlanner iteration:
        Diagnose, repair, evaluate, reflect.
        """
        last_reflection = previous_reflections[-1]
        initial_cost_function = motion_planner.cost_function_string
        # retrieve related few_shot examples from memory
        if self.config.memory_module and iteration_id <= 0:
            few_shots = self.memory.get_few_shots(
                self.config.path_to_plot,
                self.config.n_shots,
                threshold=self.config.memory_threshold,
            )
            if not few_shots:
                self.statistic.missing_few_shot_count += 1
        else:
            few_shots = []

        # generate diagnosis
        try:
            diagnosis, max_time_steps, sampling_d = self.diagnosis_module.run(
                initial_evaluation,
                motion_planner,
                last_reflection,
                few_shots,
                iteration_id,
            )
        except ValueError as _:
            print("No diagnosis provided")
            diagnosis = Diagnosis("", "", "", "", [])
            max_time_steps = motion_planner.max_time_steps
            sampling_d = motion_planner.d

        # repair the planner
        try:
            repaired_motion_planner = self.prescription_module.run(
                diagnosis.__str__(),
                initial_cost_function,
                last_reflection,
                few_shots,
                iteration_id,
            )
        except ValueError as _:
            print("No repair provided")
            repaired_motion_planner = motion_planner

        repaired_motion_planner.max_time_steps = max_time_steps
        repaired_motion_planner.d = sampling_d

        # evaluate the repair
        repair_evaluation, repair_cost_result, exception = self.evaluation_module.run(
            absolute_scenario_path, repaired_motion_planner
        )

        # reflect on the repair
        try:
            if not self.config.reflection_module:
                raise ValueError
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

        improvement = self.get_relative_improvement(
            start_total_cost, repair_cost_result.total_costs
        )
        # if improvement is noticeable
        if self.config.update_memory_module and improvement > 0.01:
            inserted = self.memory.insert(
                diagnosis.to_few_shot(max_time_steps, sampling_d),
                repaired_motion_planner.cost_function_string,
                improvement,
                self.config.path_to_plot,
            )
            if inserted:
                self.statistic.added_few_shot_count += 1

        data = repair_cost_result.total_costs
        if not data < math.inf:
            data = exception
        self.statistic.update_iteration(data)

        return (
            repair_evaluation,
            repair_cost_result,
            repaired_motion_planner,
            reflection,
        )


def run_iterative_repair(
    statistic: Statistics = None,
    scenario_path: str = "USA_Lanker-2_19_T-1",
    config: DrPlannerConfiguration = None,
    motion_planner: ReactiveMotionPlannerWrapper = None,
) -> Tuple[list[PlanningProblemCostResult], Statistics]:
    """
    Interface function to start the modular DrPlanner
    """
    if not config:
        config = DrPlannerConfiguration()

    if not statistic:
        statistic = Statistics()

    if not scenario_path.endswith(".xml"):
        scenario_id = scenario_path
        scenario_path = os.path.join(
            os.path.dirname(config.project_path), "scenarios", f"{scenario_path}.xml"
        )
    else:
        scenario_id = os.path.basename(scenario_path)[:-4]

    memory = FewShotMemory()
    save_dir = os.path.join(config.save_dir, scenario_id)
    iteration = Iteration(config, statistic, memory, save_dir)

    if not motion_planner:
        motion_planner = ReactiveMotionPlannerWrapper()

    reflection = Reflection("")
    # first run the initial planner once to obtain the initial values for the loop:
    cf0 = motion_planner.cost_function_string
    e0, cr0, _ = iteration.evaluation_module.run(scenario_path, motion_planner)
    start_total_cost = cr0.total_costs
    # collect all interesting data produced throughout the loop
    cost_functions = [cf0]
    evaluations = [e0]
    cost_results = [cr0]
    reflections = [reflection]
    iteration_id = 0
    print(e0)

    start_time = time.time()

    while iteration_id < config.iteration_max:
        e, tc, motion_planner, new_reflection = iteration.run(
            scenario_path,
            motion_planner,
            evaluations.pop(),
            cost_results[-1],
            reflections,
            iteration_id,
            start_total_cost,
        )
        print(e)
        if motion_planner:
            cost_functions.append(motion_planner.cost_function_string)
        evaluations.append(e)
        cost_results.append(tc)
        reflections.append(new_reflection)
        iteration_id += 1

    end_time = time.time()
    statistic.duration = end_time - start_time
    return cost_results, statistic
