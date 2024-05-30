import ast
import inspect
import re
import textwrap
import traceback
import warnings
from abc import ABC, abstractmethod
import os
from typing import Callable


class DescriptionBase(ABC):
    domain = "base"

    def __init__(self):
        self.description = None

    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Abstract method to generate the description.
        Subclasses must implement this method.
        """
        pass

    def output_description(self, save_path="prompt/"):
        """
        Method to output the generated description to a text file.
        The file name can be specified; defaults to 'description.txt'.
        """
        filename = f"{self.domain}.txt"
        if self.description is not None:
            full_path = os.path.join(save_path, filename)
            with open(full_path, "w") as file:
                file.write(self.description)
            print(f"<output> Description saved to {full_path}.")
        else:
            print("<output> No description generated yet.")


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
    function_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                function_calls.append(node.func.id)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                function_calls.append(node.func.attr)

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


class FunctionDescriptionBase(DescriptionBase, ABC):
    def __init__(self, function: Callable):
        super().__init__()
        self.called_functions = list(set(extract_self_method_calls_from_func(function)))

    def generate_function_description(
        self, class_or_instance, function_description: str
    ):
        description = (
            "In " + function_description + ", the following functions are called:"
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


class ExceptionDescription(DescriptionBase):
    def __init__(self, exception: Exception):
        super().__init__()
        self.exception = exception

    def generate(self):
        tb = traceback.extract_tb(self.exception.__traceback__)  # Extract the traceback
        frame = None
        for frame_summary in tb:
            if "drplanner" in frame_summary.filename:
                frame = frame_summary

        if not frame:
            frame = tb[-1]

        summary = f"TYPE: {type(self.exception)} METHOD: {frame.name} LINE: {frame.line}"
        return summary


class PlanningException(Exception):
    def __init__(self, cause: str):
        super().__init__()
        self.description = f"The planner failed: {cause}"


class CompilerException(Exception):
    def __init__(self, cause: Exception):
        super().__init__("The python code provided by the LLM could not be compiled")
        self.cause = cause


class MissingParameterException(Exception):
    def __init__(self, parameter: str):
        super().__init__()
        self.description = f"The LLM did not provide the essential parameter <{parameter}>"
