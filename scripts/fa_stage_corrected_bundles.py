#!/usr/bin/env python
"""Rebuilds the FA prompt bundles with corrected question text: problem-type tag, cleaned body, MC options.

`build_prompt_completion` (`atom/simulator/fa_alt2.py`) inserts `question_text` verbatim into
its prompt, and `fa_04b_stage_bundle.py` fills that field with the raw HTML `problem_body`.
This script recomputes `question_text` once per problem_id as `[problem type] cleaned body`,
plus lettered options for multiple-choice items, and re-emits exactly the same rows
(learner, id, problem_id, t, support_code, score, score_missing and answer_text unchanged),
so the prompt text is the only thing that differs from the input bundles.
Inputs: the train and validation bundles (`--train-bundle-in`, `--val-bundle-in`) and
Problems.csv. Outputs: the corrected bundles (`--train-bundle-out`, `--val-bundle-out`)
under staging/bundles/. The bundles contain licensed text: built locally, never published.
Run:  python scripts/fa_stage_corrected_bundles.py
Serves: staging for FA-SIM3 (Tables 10 and 12): `fa_sim3_stage_bundles.py` applies
`build_corrected_question_text` to the bundles that `infra/fa_sim3_full_run.py` loads.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tarfile

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from atom.data.fa import load_problems  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
MC_OPTION_DELIM = " || "  # atom.data.fa_mc_options.MC_OPTION_DELIM, matched here to avoid
                          # importing a module scoped to interaction-level MC matching for
                          # what is here a pure problem-level rendering step.


def build_corrected_question_text(problem_row: pd.Series) -> str:
    """One corrected prompt body per problem_id: `[problem type] cleaned body` (+ rendered
    options for MC items). Fill-in items and Order/Sort items get the tag + cleaned body
    only: `mc_options` is populated only for MC rows in `Problems.csv`, so there is
    nothing to render for them."""
    body = problem_row["problem_body_clean"]
    ptype = problem_row["problem_type"]
    text = f"[{ptype}] {body}" if isinstance(ptype, str) and ptype else body

    mc_options = problem_row.get("mc_options")
    if isinstance(ptype, str) and "Multiple Choice" in ptype and isinstance(mc_options, str) and mc_options:
        options = [o.strip() for o in mc_options.split(MC_OPTION_DELIM) if o.strip()]
        if options:
            rendered = "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options))
            text = f"{text}\nOptions:\n{rendered}"
    return text


def rebuild_bundle(old_bundle_path: pathlib.Path, out_path: pathlib.Path,
                   corrected_text_by_problem: dict[int, str]) -> dict:
    extract_dir = pathlib.Path(f"/tmp/fa_sim1_stage_{old_bundle_path.stem}")
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(old_bundle_path, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")
    bundle = pd.read_json(extract_dir / "events.jsonl", lines=True)

    n_before = len(bundle)
    missing = bundle.loc[~bundle["problem_id"].isin(corrected_text_by_problem), "problem_id"].unique()
    assert len(missing) == 0, (
        f"{len(missing)} problem_id(s) in the bundle have no entry in Problems.csv "
        f"(after clean_body/type join) -- cannot silently drop rows here: {missing[:10]}")

    bundle["question_text"] = bundle["problem_id"].map(corrected_text_by_problem)
    assert len(bundle) == n_before  # the join must be 1:1 on problem_id, never fan out

    out_dir = pathlib.Path(f"/tmp/fa_sim1_out_{out_path.stem}")
    out_dir.mkdir(exist_ok=True)
    events_path = out_dir / "events.jsonl"
    bundle.to_json(events_path, orient="records", lines=True)
    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(events_path, arcname="events.jsonl")

    return {"n_rows": len(bundle), "n_unique_problems": int(bundle["problem_id"].nunique())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bundle-in", default="staging/bundles/fa_06_stage2_05pct.tar.gz")
    ap.add_argument("--val-bundle-in", default="staging/bundles/fa_06_val_eval.tar.gz")
    ap.add_argument("--train-bundle-out", default="staging/bundles/fa_sim1_train_corrected.tar.gz")
    ap.add_argument("--val-bundle-out", default="staging/bundles/fa_sim1_val_corrected.tar.gz")
    args = ap.parse_args()

    problems = load_problems(clean_body=True)
    problems = problems.drop_duplicates("problem_id").set_index("problem_id")
    corrected_text_by_problem = {
        pid: build_corrected_question_text(row) for pid, row in problems.iterrows()
    }
    print(f"[fa-stage-corrected] built corrected text for {len(corrected_text_by_problem)} problems "
         f"(type breakdown: {problems['problem_type'].value_counts().to_dict()})", flush=True)

    for tag, bundle_in, bundle_out in (
        ("train", args.train_bundle_in, args.train_bundle_out),
        ("val", args.val_bundle_in, args.val_bundle_out),
    ):
        result = rebuild_bundle(REPO / bundle_in, REPO / bundle_out, corrected_text_by_problem)
        print(f"[fa-stage-corrected] {tag}: {result}", flush=True)

    # Print one before/after example so the correction can be checked by eye; an MC
    # problem is chosen because option rendering is the branch most likely to go wrong.
    mc_pid = problems[problems["problem_type"].astype(str).str.contains("Multiple Choice", na=False)].index[0]
    print(f"[fa-stage-corrected] EXAMPLE (problem_id={mc_pid}):", flush=True)
    print(f"  BEFORE (raw): {problems.loc[mc_pid, 'problem_body'][:200]!r}", flush=True)
    print(f"  AFTER (corrected): {corrected_text_by_problem[mc_pid][:400]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
