import ast
import re
from typing import Callable

from drplanner.describer.base import DescriptionBase, FunctionDescriptionBase


class MethodCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.method_calls = []

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            self.method_calls.append(node.attr)
        self.generic_visit(node)


class HeuristicDescription(FunctionDescriptionBase):
    domain = "heuristic"

    def __init__(self, heuristic_function: Callable):
        super().__init__(heuristic_function)

    def generate(self, class_or_instance):
        function_description = (
            "the current heuristic function of the A* search algorithm"
        )
        return self.generate_function_description(
            class_or_instance, function_description
        )


class CostFunctionDescription(FunctionDescriptionBase):
    domain = "cost"

    def __init__(self, cost_function: Callable):
        super().__init__(cost_function)

    def generate(self, class_or_instance):
        function_description = "the current cost function of the reactive planner"
        return self.generate_function_description(
            class_or_instance, function_description
        )


class MotionPrimitiveDescription(DescriptionBase):
    domain = "motion_primitives"

    def __init__(self):
        super().__init__()
        self.base_statement = (
            "\n Motion primitives are short trajectories that are drivable by a given vehicle model. "
            "By concatenating the primitives, a drivable trajectory can be constructed that leads"
            " the vehicle from the initial state to the goal state. "
            "Generating too sparse primitives (low branching factor) may restrict the search space "
            "such that no feasible solution can be found. On the other hand, generating too dense "
            "primitives (high branching factor) may dramatically increase the time of search."
        )

    def generate(self, mp_name):
        # Regular expression pattern to capture the values
        pattern = r"V_(?P<v_min>\d+\.?\d*)_+(?P<v_max>\d+\.?\d*)_Vstep_(?P<v_step>\d+\.?\d*)_SA_(?P<sa_min>-?\d+\.?\d*)_+(?P<sa_max>-?\d+\.?\d*)_SAstep_(?P<sa_step>\d+\.?\d*)_T_(?P<duration>\d+\.?\d*)_Model_.*\.xml"
        match = re.match(pattern, mp_name)
        selection_statement = (
            f'The currently used motion primitives are named as "{mp_name}",'
        )
        selection_statement += "adhering to the format V_{velocity_min}_{velocity_max}_Vstep_{velocity_step_size}_SA_{steering_angle_min}_{steering_angle_max}_SAstep_{steering_angle_step_size}_T_{time_durantion}_Model_{vehicle_model}."

        if match:
            # Extracting the values from the matched groups
            v_min = match.group("v_min")
            v_max = match.group("v_max")
            v_step = match.group("v_step")
            sa_min = match.group("sa_min")
            sa_max = match.group("sa_max")
            sa_step = match.group("sa_step")
            duration = match.group("duration")

            # Form the statement
            velocity_statement = (
                f"This name implies the setup of the motion primitives: "
                f"The velocity range is set from {v_min}m/s to {v_max}m/s "
                f"with incremental steps of {v_step}m/s."
            )
            steering_statement = (
                f"The steering angle varies from {sa_min} rad to {sa_max} rad "
                f"with a step size of {sa_step}rad."
            )
            steps = float(duration) / 0.1
            duration_statement = (
                f"The motion primitives have a fixed duration of {steps} time steps."
            )

            # Combine the statements
            self.description = f"{self.base_statement} {selection_statement} {velocity_statement} {steering_statement} {duration_statement}"

        else:
            assert "The provided filename does not match the expected pattern."
            self.description = ""
        return self.description
