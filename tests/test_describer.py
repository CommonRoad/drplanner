import unittest
import ast
import inspect
import textwrap

from drplanner.describer.planner_description import (
    MethodCallVisitor,
    HeuristicDescription,
    MotionPrimitiveDescription,
)

from drplanner.describer.base import (
    extract_self_method_calls_from_func,
    extract_called_functions,
    extract_function_info,
    clean_docstring,
)


class ExampleClass:
    def __init__(self):
        pass

    def example_method(self):
        self.do_something()
        self.compute_value(42)

    def another_method(self):
        """Another method for testing."""
        print("This method does not use self explicitly.")

    def do_something(self):
        """Do something"""
        pass  # Stub method for testing purposes

    def compute_value(self, value):
        """Compute a value based on the input value."""
        pass  # Stub method for testing purposes


class SelfMethodCallsTests(unittest.TestCase):
    def test_method_call_visitor(self):
        visitor = MethodCallVisitor()
        source = inspect.getsource(ExampleClass.example_method)
        dedented_source = textwrap.dedent(source)
        visitor.visit(ast.parse(dedented_source))
        self.assertEqual(
            sorted(visitor.method_calls), sorted(["do_something", "compute_value"])
        )

    def test_extract_self_method_calls_from_func(self):
        method_calls = extract_self_method_calls_from_func(ExampleClass.example_method)
        self.assertListEqual(
            sorted(method_calls), sorted(["do_something", "compute_value"])
        )

    def test_extract_self_method_calls_no_self(self):
        method_calls = extract_self_method_calls_from_func(ExampleClass.another_method)
        self.assertListEqual(method_calls, [])  # no self-calls in another_method

    def test_extract_called_functions(self):
        called_functions = extract_called_functions(ExampleClass.example_method)
        self.assertListEqual(
            sorted(called_functions), sorted(["do_something", "compute_value"])
        )

    def test_extract_function_info(self):
        func_name, docstring = extract_function_info(ExampleClass.example_method)
        self.assertEqual(func_name, "example_method")
        self.assertIsNone(docstring)  # Assuming no docstring in example_method

    def test_clean_docstring(self):
        docstring = """
        This function does something.
        :param x: the x value
        :return: None
        """
        clean = clean_docstring(docstring)
        self.assertEqual(clean, "This function does something.")

    def test_heuristic_description(self):
        heuristic_function = ExampleClass.example_method
        class_instance = ExampleClass()
        heuristic_description = HeuristicDescription(heuristic_function)
        desc = heuristic_description.generate(class_instance)
        self.assertIn("do_something", desc)

    def test_motion_primitive_description(self):
        mp_name = (
            "V_0.0_20.0_Vstep_4.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml"
        )
        motion_description = MotionPrimitiveDescription()
        desc = motion_description.generate(mp_name)
        self.assertIn("velocity range is set from 0.0m/s to 20.0m/s", desc)
