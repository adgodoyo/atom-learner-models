#!/usr/bin/env python
"""Functional-equivalence harness: compile and run a Java submission against a problem's TIKTOC test cases.

`score_submission` turns one attempt's code into a test vector (one bool per test case, in
the order of `load_test_cases`), which callers compare with the learner's own test vector.
Coverage is the 13 problems in `HEADLINE_PROBLEM_IDS`: those whose `coding_prompt_id` ->
`ProblemID` join is verified, whose TIKTOC test-case status is "All Good", and which are not
in the item holdout.
The Java-construction functions are copied unmodified from TIKTOC's
`gen_code_compiler.py` (only the `javac`/`java` paths are parametrised). That module cannot
be imported directly because it pulls in TIKTOC's full training stack at import time.
`javac`/`java` are resolved in order from the `CW5_JAVAC`/`CW5_JAVA` environment variables,
`PATH`, then a Homebrew keg-only fallback; each candidate must actually run `-version`,
because macOS ships a non-functional `/usr/bin/java` stub that `shutil.which` would find.
Run: `python scripts/cw_5_functional_equivalence.py` runs a positive/negative control on
real S19 attempts (3 problems) and writes `reports/cw_5_functional_equivalence_smoke.json`.
Needs a JDK. Serves Table 11 (generation).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
DATASETS = REPO.parent / "Datasets"
TIKTOC = DATASETS / "tiktoc-main"

_MAC_JAVAC_FALLBACK = "/opt/homebrew/opt/openjdk/bin/javac"
_MAC_JAVA_FALLBACK = "/opt/homebrew/opt/openjdk/bin/java"


def _first_working_java_tool(env_var: str, candidates: list[str]) -> str:
    """First of `candidates` that exists AND actually runs `<tool> -version` cleanly.
    Returns the last candidate unverified if none work (so JDK_PRESENT-style checks
    downstream still get a path to report as missing/broken, rather than None)."""
    override = os.environ.get(env_var)
    if override:
        candidates = [override] + candidates
    last = candidates[-1]
    for cand in candidates:
        if not cand or not pathlib.Path(cand).exists():
            continue
        try:
            r = subprocess.run([cand, "-version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return cand
        except Exception:
            continue
        last = cand
    return last


JAVAC = _first_working_java_tool("CW5_JAVAC", [shutil.which("javac") or "", _MAC_JAVAC_FALLBACK])
JAVA = _first_working_java_tool("CW5_JAVA", [shutil.which("java") or "", _MAC_JAVA_FALLBACK])

# From reports/cw_5_functional_equivalence_join_check.md: the join-verified,
# Test-Case-Status=="All Good", retained (not held-out) problem IDs. ProblemID 108
# is a disclosed near-miss (real cases, but TIKTOC's own audit flags "Mismatch") --
# deliberately excluded from the headline set, matching gen_code_compiler.py's own
# `test_case_check` convention of filtering to All Good only.
HEADLINE_PROBLEM_IDS = [1, 3, 5, 12, 13, 17, 20, 21, 22, 24, 25, 46, 71]


def load_test_cases(problem_id: int) -> list[dict]:
    """Real TIKTOC test cases for one problem, `input`/`expected_output` pairs, in the
    same raw Java-argument-list string form `construct_java_code` splices directly into
    a method call (e.g. `"3, 4"` -> `sortaSum(3, 4)`) -- no reformatting."""
    df = pd.read_csv(TIKTOC / "test-case-query-results" / "test_cases-1-26-24.csv",
                     engine="python", on_bad_lines="skip")
    rows = df[df["coding_prompt_id"] == problem_id]
    return [{"input": r.input, "expected_output": str(r.expected_output).strip()}
           for r in rows.itertuples(index=False)]


# -- ported verbatim from Datasets/tiktoc-main/gen_code_compiler.py --------------------

def extract_function_features(code: str):
    code = code.split("\n")
    for line in code:
        if "public" in line or "private" in line:
            line = line.split(" ")
            for i in range(len(line)):
                word = line[i]
                if "(" in word:
                    method_name = word.split("(")[0]
                    output_type = line[i - 1]
                    return method_name, output_type
    return None, None


def create_unit_test_case_function(function_name: str, test_cases_list: list[str]) -> str:
    unit_test_case_function_code = "public void {:s}_test(String []s)".format(function_name)
    unit_test_case_function_code += "{\n"
    test_cases_prints = ""
    for test_case in test_cases_list:
        test_cases_prints += "System.out.println({:s}({:s}));\n".format(function_name, test_case)
    unit_test_case_function_code += test_cases_prints
    unit_test_case_function_code += "}\n"
    return unit_test_case_function_code


def get_main_function_code(class_name: str, function_name: str) -> str:
    main_function_code = "public static void main(String []args)"
    main_function_code += "{\n"
    main_function_code += "{:s} obj = new {:s}();\n".format(class_name, class_name)
    main_function_code += "obj.{:s}_test(args);\n".format(function_name)
    main_function_code += "}\n"
    return main_function_code


def get_complete_java_code(class_name: str, code: str, unit_test_case_function_code: str,
                           main_function_code: str) -> str:
    complete_java_code = "public class {:s}".format(class_name)
    complete_java_code += "{\n"
    complete_java_code += code
    complete_java_code += "\n"
    complete_java_code += unit_test_case_function_code
    complete_java_code += "\n"
    complete_java_code += main_function_code
    complete_java_code += "\n"
    complete_java_code += "}\n"
    return complete_java_code


def parse_java_code(java_code: str):
    java_code = java_code.strip()
    signature = re.search(r"(.*?){", java_code, re.DOTALL).group(1).strip()
    body_start_index = java_code.index("{") + 1
    body_end_index = java_code.rindex("}")
    body = java_code[body_start_index:body_end_index].strip()
    return signature, body


def handle_exception(student_code: str, output_type: str | None = None) -> str:
    function_signature, main_body = parse_java_code(student_code)
    helper_keyword_pattern = re.compile(r"\bpublic\b")
    check_helper = helper_keyword_pattern.search(main_body)
    if bool(check_helper):
        start_pos = check_helper.start()
        end_bracket = main_body.rfind("}", 0, start_pos)
        main_part = main_body[:end_bracket]
        helper = main_body[start_pos:] + "\n" + "} \n"
    else:
        main_part = main_body
        helper = ""

    except_code = "try {\n"
    except_code += main_part
    except_code += "\n"
    except_code += "} catch (Exception e) {\n"
    except_code += 'System.out.print("Error ");\n'
    if output_type == "int":
        except_code += "return -999;\n"
    elif output_type == "boolean":
        except_code += "return false;\n"
    elif output_type == "String":
        except_code += 'return "ERROR";\n'
    elif output_type == "int[]":
        except_code += "return new int[] {-999};\n"
    elif output_type == "String[]":
        except_code += 'return new String[] {"ERROR"};\n'
    except_code += "}\n"

    new_code = function_signature + "{\n" + except_code + "}\n"
    new_code += helper
    return new_code


def construct_java_code(test_cases: list[str], function_name: str, class_name: str,
                        student_code: str, output_type: str | None) -> str:
    unit_test_case_function_code = create_unit_test_case_function(function_name, test_cases)
    main_function_code = get_main_function_code(class_name, function_name)
    exception_handeled_code = handle_exception(student_code, output_type)
    return get_complete_java_code(class_name, exception_handeled_code,
                                  unit_test_case_function_code, main_function_code)


# -- end verbatim port -------------------------------------------------------------

def compile_and_run(java_code: str, class_name: str, timeout: float = 15.0) -> dict:
    """Writes `java_code` to `<tmp>/<class_name>.java`, compiles+runs it with the local
    JDK (`run_command_subprocess`'s compile-then-run shape, ported, with the resolved
    `JAVAC`/`JAVA` binaries substituted for the bare `javac`/`java` TIKTOC's script
    assumes are on `PATH`). Returns `{stage: 'compile'|'run'|'ok', stdout, stderr,
    returncode}` -- `stdout` (one println per test case, in order) is what the caller
    scores against `expected_output`."""
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / f"{class_name}.java"
        src.write_text(java_code)
        compiled = subprocess.run([JAVAC, str(src)], capture_output=True, text=True, timeout=timeout)
        if compiled.returncode != 0:
            return {"stage": "compile", "stdout": "", "stderr": compiled.stderr, "returncode": compiled.returncode}
        try:
            ran = subprocess.run([JAVA, "-cp", d, class_name], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"stage": "run", "stdout": "", "stderr": "TIMEOUT", "returncode": -1}
        stage = "ok" if ran.returncode == 0 else "run"
        return {"stage": stage, "stdout": ran.stdout, "stderr": ran.stderr, "returncode": ran.returncode}


def score_submission(student_code: str, test_cases: list[dict], timeout: float = 15.0) -> dict:
    """One attempt's code against one problem's real test cases -> the test vector
    (`[bool, ...]`, one per case, in the SAME order as `test_cases` -- caller compares to
    the learner's OWN test vector for functional equivalence) plus the compile/run outcome. `class_name` is fixed per call (a temp dir
    isolates every compilation, so no collision risk across parallel/sequential calls).
    `timeout` threads through to `compile_and_run`; callers scoring the per-attempt
    test-vector labels pass `timeout=30.0` explicitly, other callers keep the 15.0s default."""
    function_name, output_type = extract_function_features(student_code)
    if function_name is None:
        return {"compiled": False, "stage": "parse", "test_vector": [False] * len(test_cases),
               "n_correct": 0, "n_cases": len(test_cases), "error": "no public/private method found"}
    class_name = "Sol"
    inputs = [tc["input"] for tc in test_cases]
    java_code = construct_java_code(inputs, function_name, class_name, student_code, output_type)
    result = compile_and_run(java_code, class_name, timeout=timeout)
    if result["stage"] != "ok":
        return {"compiled": result["stage"] != "compile", "stage": result["stage"],
               "test_vector": [False] * len(test_cases), "n_correct": 0, "n_cases": len(test_cases),
               "error": (result["stderr"] or "")[:500]}
    actual_lines = [ln.strip() for ln in result["stdout"].splitlines()]
    test_vector = []
    for i, tc in enumerate(test_cases):
        actual = actual_lines[i] if i < len(actual_lines) else None
        test_vector.append(actual is not None and actual == tc["expected_output"])
    return {"compiled": True, "stage": "ok", "test_vector": test_vector,
           "n_correct": sum(test_vector), "n_cases": len(test_cases), "error": None}


def main() -> int:
    """Positive and negative control on real S19 attempts (positive control first): a
    full-credit submission should pass every case, a partial-credit one should fail at
    least one. Not the full corpus sweep."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from atom.data.cw_events import build_attempts, load_raw_tables

    main_tbl, code_states = load_raw_tables("S19", include_code=True)
    attempts = build_attempts(main_tbl, code_states)

    results = []
    for pid in HEADLINE_PROBLEM_IDS[:3]:  # 3 problems, a smoke run, not the full 13
        test_cases = load_test_cases(pid)
        sub = attempts[(attempts["problem_id"].astype(int) == pid)
                      & (attempts["compile_result"] == "Success")]
        if len(sub) == 0:
            continue
        # POSITIVE control: a real full-credit (score_frac==1.0) submission should pass
        # every case its own grader passed it on.
        full_credit = sub[sub["score_frac"] == 1.0]
        # NEGATIVE control: a real partial-credit submission should fail at least one.
        partial = sub[(sub["score_frac"] > 0) & (sub["score_frac"] < 1.0)]
        for label, pool in [("full_credit", full_credit), ("partial_credit", partial)]:
            if len(pool) == 0:
                continue
            row = pool.iloc[0]
            r = score_submission(row["code"], test_cases)
            r.update(problem_id=pid, label=label, cw_score_frac=float(row["score_frac"]))
            results.append(r)
            print(f"[func-eq] problem={pid} label={label} cw_score_frac={row['score_frac']:.3f} "
                 f"compiled={r['compiled']} n_correct={r['n_correct']}/{r['n_cases']} "
                 f"error={r['error']}")

    out = REPO / "reports" / "cw_5_functional_equivalence_smoke.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[func-eq] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
