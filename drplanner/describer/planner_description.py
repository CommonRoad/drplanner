import ast
import re
import textwrap
import inspect
import warnings
from typing import Callable

from drplanner.describer.base import DescriptionBase


class MethodCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.method_calls = []

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            self.method_calls.append(node.attr)
        self.generic_visit(node)


def extract_self_method_calls_from_func(func):
    code = inspect.getsource(func)
    return extract_self_method_calls(code)


def extract_self_method_calls(code: str) -> list[str]:
    code = textwrap.dedent(code)  # dedent the code before parsing
    tree = ast.parse(code)

    method_calls = []

    for node in ast.walk(tree):
        # Check if it's an attribute accessed through 'self'
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            # Check if this attribute is part of a method call
            for parent in ast.walk(tree):
                if (
                    isinstance(parent, ast.Call) and parent.func == node
                ):  # Ensure that the call's function is the attribute we found
                    method_calls.append(node.attr)
                    break  # If we found a match, break out of the inner loop

    return method_calls


def extract_called_functions(func: Callable):
    """Extract the names of the functions called within the given function."""
    # Extract the source code of the function
    source = inspect.getsource(func)

    # Dedent the source code
    dedented_source = textwrap.dedent(source)

    # Parse the adjusted source code
    tree = ast.parse(dedented_source)

    # Get all function calls
    function_calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    return function_calls


def extract_function_info(func: Callable):
    """Extract the function name and its docstring."""
    function_name = func.__name__
    docstring = func.__doc__
    return function_name, docstring


def clean_docstring(docstring):
    # Remove lines starting with ":param" and the subsequent line (if it doesn't start with ":")
    cleaned = re.sub(r"\s*:param.*?(?=\n:|$)", "", docstring, flags=re.DOTALL)

    # Remove any leading and trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


class HeuristicDescription(DescriptionBase):
    domain = "heuristic"

    def __init__(self, heuristic_function: Callable):
        super().__init__()
        self.called_functions = list(
            set(extract_self_method_calls_from_func(heuristic_function))
        )

    def generate(self, class_or_instance):
        description = (
            f"In the current heuristic function of the A* search algorithm, the following functions are "
            f"called: "
        )
        for func_name in self.called_functions:
            func = getattr(
                class_or_instance, func_name, None
            )  # get the method from the class or instance
            if func:
                name, doc = extract_function_info(func)
                if doc:
                    # only adding the explanation when it exists, avoiding error
                    description += f'"{name}" {clean_docstring(doc)} '
                else:
                    warnings.warn(f'the docstring of function "{name}" is missing')
            else:
                description += f"{func_name} not found; "
        self.description = description.replace("\n", "").replace("\t", "")
        return self.description


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
