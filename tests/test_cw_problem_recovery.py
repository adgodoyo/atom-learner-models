import pandas as pd

from atom.data.cw_problem_recovery import (
    extract_method_name,
    identify_problem_method_names,
    normalize_statement,
)


def test_extract_method_name_simple_signature():
    code = "public int sortaSum(int a, int b)\n{\n    return a + b;\n}"
    assert extract_method_name(code) == "sortaSum"


def test_extract_method_name_static_and_array_return_type():
    code = "public static int[] fix45(int[] nums)\n{\n    return nums;\n}"
    assert extract_method_name(code) == "fix45"


def test_extract_method_name_returns_none_on_malformed_code():
    assert extract_method_name("this is not java at all") is None


def test_extract_method_name_takes_first_match_even_with_a_helper_method():
    code = (
        "public boolean sameEnds(int[] nums, int n)\n{\n    return helper(nums, n);\n}\n"
        "private boolean helper(int[] nums, int n) { return true; }"
    )
    assert extract_method_name(code) == "sameEnds"


def test_identify_problem_method_names_modal_and_agreement():
    attempts = pd.DataFrame({
        "problem_id": [1, 1, 1, 2, 2],
        "code": [
            "public int sortaSum(int a, int b) { return a; }",
            "public int sortaSum(int a, int b) { return b; }",
            "public int sortaSum(int a, int b) { }",  # same name, different body
            "public boolean atFirst(String str) { return true; }",
            "garbage, not java",  # no extractable name
        ],
    })
    out = identify_problem_method_names(attempts)
    assert len(out) == 2
    row1 = out[out["problem_id"] == 1].iloc[0]
    assert row1["method_name"] == "sortaSum"
    assert row1["agreement"] == 1.0
    assert row1["n_submissions"] == 3
    row2 = out[out["problem_id"] == 2].iloc[0]
    assert row2["method_name"] == "atFirst"
    assert row2["agreement"] == 1.0  # 1/1 named (the garbage row contributes no name)
    assert row2["n_named"] == 1


def test_identify_problem_method_names_disagreement_lowers_agreement():
    attempts = pd.DataFrame({
        "problem_id": [1, 1, 1, 1],
        "code": [
            "public int sortaSum(int a, int b) { return a; }",
            "public int sortaSum(int a, int b) { return a; }",
            "public int sortaSum(int a, int b) { return a; }",
            "public int wrongName(int a, int b) { return a; }",
        ],
    })
    out = identify_problem_method_names(attempts)
    row = out.iloc[0]
    assert row["method_name"] == "sortaSum"
    assert row["agreement"] == 0.75


def test_identify_problem_method_names_all_unnamed_reports_zero_agreement():
    attempts = pd.DataFrame({"problem_id": [1, 1], "code": ["garbage", "also garbage"]})
    out = identify_problem_method_names(attempts)
    row = out.iloc[0]
    assert row["method_name"] is None
    assert row["agreement"] == 0.0
    assert row["n_submissions"] == 2


def test_identify_problem_method_names_ignores_missing_code_rows():
    attempts = pd.DataFrame({
        "problem_id": [1, 1],
        "code": ["public int sortaSum(int a, int b) { return a; }", None],
    })
    out = identify_problem_method_names(attempts)
    assert out.iloc[0]["n_submissions"] == 1  # the NaN-code row is excluded entirely


def test_normalize_statement_strips_wrapper_and_collapses_whitespace():
    a = "Write a function in Java that implements the following logic:  Given a number n,   return true."
    b = "given a number n, return true."
    assert normalize_statement(a) == b


def test_normalize_statement_case_insensitive_and_whitespace_only_differences_match():
    a = "Write a function in Java that implements the following logic: Foo\nBar."
    b = "Write a function in java that implements the following logic: foo bar."
    assert normalize_statement(a) == normalize_statement(b)


def test_normalize_statement_different_content_stays_different():
    a = "Write a function in Java that implements the following logic: return the sum."
    b = "Write a function in Java that implements the following logic: return the max."
    assert normalize_statement(a) != normalize_statement(b)
