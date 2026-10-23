"""CW-5 B-2c item 3 -- unit tests for the functional-equivalence harness's ported
Java-code-construction functions (pure string/regex logic, no JDK needed for these) plus
one JDK-gated end-to-end compile+run test skipped automatically if the local JDK
(`scripts/cw_5_functional_equivalence.py::JAVAC/JAVA`) is not present, mirroring this
repo's other environment-conditional tests (e.g. GPU-only tests skip on CPU-only Macs)."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "cw_5_functional_equivalence", REPO / "scripts" / "cw_5_functional_equivalence.py")
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)

JDK_PRESENT = pathlib.Path(fe.JAVAC).exists() and pathlib.Path(fe.JAVA).exists()


def test_extract_function_features_simple_method():
    code = "public int sortaSum(int a, int b)\n{\n    return a + b;\n}\n"
    name, output_type = fe.extract_function_features(code)
    assert name == "sortaSum"
    assert output_type == "int"


def test_extract_function_features_no_method_returns_none():
    name, output_type = fe.extract_function_features("int x = 5;")
    assert name is None
    assert output_type is None


def test_parse_java_code_splits_signature_and_body():
    code = "public int f(int a)\n{\n    return a;\n}"
    signature, body = fe.parse_java_code(code)
    assert signature == "public int f(int a)"
    assert body.strip() == "return a;"


def test_construct_java_code_produces_compilable_shape():
    student_code = "public int sortaSum(int a, int b)\n{\n    int sum = a + b;\n    return sum;\n}\n"
    java_code = fe.construct_java_code(["3, 4", "10, 9"], "sortaSum", "Sol", student_code, "int")
    assert "public class Sol" in java_code
    assert "sortaSum_test" in java_code
    assert "System.out.println(sortaSum(3, 4));" in java_code
    assert "System.out.println(sortaSum(10, 9));" in java_code
    assert "public static void main" in java_code
    # exception-wrapped: the student's own return sits inside a try block
    assert "try {" in java_code
    assert "catch (Exception e)" in java_code


def test_construct_java_code_handles_a_helper_method():
    """A student submission with a helper method (public keyword appearing a second
    time in the body) must not be mangled -- `handle_exception`'s own helper-detection
    branch, exercised directly rather than assumed correct."""
    student_code = (
        "public int f(int a)\n{\n    return helper(a);\n}\n\n"
        "public int helper(int a)\n{\n    return a * 2;\n}\n"
    )
    java_code = fe.construct_java_code(["5"], "f", "Sol", student_code, "int")
    assert "helper" in java_code
    assert java_code.count("public int helper") == 1


@pytest.mark.skipif(not JDK_PRESENT, reason="local JDK not found at scripts/cw_5_functional_equivalence.py's JAVAC/JAVA paths")
def test_score_submission_end_to_end_correct_and_wrong_code():
    """Real compile+run against a tiny local JDK: a correct submission must score 2/2,
    a submission with a real logic bug must score < 2/2 -- both directions exercised, not
    just the happy path (own positive control first, this codebase's standing rule)."""
    test_cases = [{"input": "3, 4", "expected_output": "7"}, {"input": "10, 9", "expected_output": "20"}]
    correct_code = ("public int sortaSum(int a, int b)\n{\n    int sum = a + b;\n"
                    "    if (sum >= 10 && sum <= 19) { return 20; }\n    return sum;\n}\n")
    wrong_code = "public int sortaSum(int a, int b)\n{\n    return a + b;\n}\n"  # misses the 10-19 rule

    correct_result = fe.score_submission(correct_code, test_cases)
    assert correct_result["compiled"] is True
    assert correct_result["n_correct"] == 2

    wrong_result = fe.score_submission(wrong_code, test_cases)
    assert wrong_result["compiled"] is True
    assert wrong_result["n_correct"] == 1  # 3+4=7 passes, 10+9=19 should be 20 and fails
    assert wrong_result["test_vector"] == [True, False]


@pytest.mark.skipif(not JDK_PRESENT, reason="local JDK not found")
def test_score_submission_reports_compile_error_not_a_crash():
    test_cases = [{"input": "3, 4", "expected_output": "7"}]
    broken_code = "public int f(int a, int b {\n    return a + b\n}\n"  # syntax errors
    result = fe.score_submission(broken_code, test_cases)
    assert result["compiled"] is False
    assert result["n_correct"] == 0
