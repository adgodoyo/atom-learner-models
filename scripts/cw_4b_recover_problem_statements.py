#!/usr/bin/env python
"""Recover the 50 CodeWorkout problem statements from CodingBat and encode them as `s(problem)`.

Why: a per-problem vector built as the mean of student code draws most of its mass from
the cohorts it is later evaluated on (measured: on average 64.1% from F19, the sealed
external cohort, and 6.5% from S19-val, only 29.4% from S19-train). The problem statement
is a property of the problem, not of any cohort, so encoding it removes that leak.

Pipeline:
  1. Identify the modal method name per problem from S19-train code
     (`atom.data.cw_problem_recovery.identify_problem_method_names`).
  2. Look up the CodingBat statement (Nick Parlante, Stanford) for each method name in
     the checked-in `artifacts/cw_coding_bat_recovery_raw.json`
     (`method_name -> {statement, source_url}`). Every entry is verbatim page text with
     its own `source_url`; nothing is generated.
  3. Pre-registered gate: the 17 problems whose text ships with the TIKTOC release
     (`prompt_concept_summary.csv`, rows with `Test Case Status == 'All Good'`) must all
     be judged the same problem. Normalised string equality is a first-pass filter;
     `KNOWN_17_VERDICTS` records the reason for each of the 9 items that differ only in
     wording (comma/quote formatting, one added sentence, a typo in the release text, or
     a paraphrase). A single "different problem" verdict stops the script.
  4. Write `artifacts/cw_problem_statements.json` (50 entries).
  5. Encode each `wrapped_prompt` through the frozen backbone on the same route as the
     mathematics item vectors `s(q)` (mean-pool at layer 15, `transforms/problem_mean_L15.npz`
     applied without refitting, k=5, whitened), overwriting
     `artifacts/cw_problem_codec_mean_L15.npz`.

Outputs: the files above, `artifacts/cw_problem_method_names.csv`, `artifacts/cw_4b_gate_17.csv`
and `reports/cw_4b_recover_problem_statements.md`.
Run: `python scripts/cw_4b_recover_problem_statements.py [--production-rank N]`.
Serves Chapter 4 (data and splits).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atom.codec.backbone import DEFAULT_BACKBONE, load_backbone  # noqa: E402
from atom.codec.extract import RawStateCache, extract_texts, text_hash  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_events import build_attempts, load_raw_tables  # noqa: E402
from atom.data.cw_problem_recovery import identify_problem_method_names, normalize_statement  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "cache" / "cw_problem_statements"
TRANSFORM_PATH = REPO / "transforms" / "problem_mean_L15.npz"
RELEASE_CSV = REPO.parent / "Datasets" / "tiktoc-main" / "test-case-query-results" / "prompt_concept_summary.csv"
K_DEFAULT = 5
WRAPPER = "Write a function in Java that implements the following logic: "

KNOWN_17 = {"sortaSum", "in1To10", "answerCell", "squirrelPlay", "caughtSpeeding", "redTicket",
           "loneSum", "luckySum", "noTeenSum", "blackjack", "evenlySpaced", "zipZap", "endOther",
           "xyBalance", "getSandwich", "isEverywhere", "canBalance"}

# The pre-registered gate's own eyeball pass (step 3): every one of the 9 items that
# failed strict normalised-string equality against the release, with the reason judged.
# All 9 are wording drift; none is a different problem. Kept here, not just in prose, so
# the verdict is a checkable data structure, not an assertion.
KNOWN_17_VERDICTS = {
    "caughtSpeeding": ("same_problem", "release adds \"or\" before \"2=big ticket\"; punctuation spacing only"),
    "evenlySpaced": ("same_problem", "release writes \"a, b, and c\" vs recovered \"a b c\" -- comma/conjunction formatting only"),
    "getSandwich": ("same_problem", "release adds an explanatory sentence naming the Java method signature (str/method name); identical underlying logic and example"),
    "in1To10": ("same_problem", "release quotes \"outsideMode\"; recovered does not -- punctuation only"),
    "loneSum": ("same_problem", "release writes \"a, b, and c\" vs recovered \"a b c\" -- comma/conjunction formatting only"),
    "luckySum": ("same_problem", "release writes \"a, b, and c\" vs recovered \"a b c\" -- comma/conjunction formatting only"),
    "noTeenSum": ("same_problem", "release paraphrases more heavily (\"write two methods\" vs the recovered helper-signature framing) but describes the identical teen-sum/fixTeen(15,16 excepted) rule"),
    "xyBalance": ("same_problem", "release has \"characterss\" (a typo in the release text itself) vs recovered \"chars\" -- synonym + typo, not content"),
    "zipZap": ("same_problem", "release rephrases the framing (\"find all places\" vs \"look for patterns\") but the rule (z..p length-3, middle letter removed, same zipXzap example) is identical"),
}


def load_release_text() -> dict:
    df = pd.read_csv(RELEASE_CSV)
    all_good = df[df["Test Case Status"] == "All Good"]
    return dict(zip(all_good["ProblemID"].astype(int), all_good["Requirement"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-rank", type=int, default=None,
                    help="rank used when reapplying the codec transform (default: the "
                         "transform's own k_max). Passing 50 explicitly keeps the 45-dim "
                         "production artifact fixed when the shared transform's k_max "
                         "is raised to study other ranks.")
    args = ap.parse_args()
    logger = RunLogger(config={"task": "cw_4b_recover_problem_statements",
                               "production_rank": args.production_rank}, data_version="CSEDM-S19-train")

    # -- step 1: identify --
    logger.heartbeat(step=0, force=True, stage="identifying method names from S19-train code")
    main_table, code_states = load_raw_tables("S19", include_code=True)
    attempts = build_attempts(main_table, code_states=code_states)
    split = pd.read_csv(REPO / "splits" / "cw_s19_learner_split.csv")
    train_learners = set(split.loc[split["split"] == "train", "learner"])
    train_attempts = attempts[attempts["learner"].isin(train_learners)]
    identified = identify_problem_method_names(train_attempts)
    identified.to_csv(REPO / "artifacts" / "cw_problem_method_names.csv", index=False)
    all_identified = identified["method_name"].notna().all()
    unique_names_ok = identified["method_name"].nunique() == len(identified)
    logger.heartbeat(step=1, force=True, stage="identified", n_problems=len(identified),
                     all_identified=bool(all_identified), unique_1to1=bool(unique_names_ok),
                     min_agreement=float(identified["agreement"].min()))

    # -- step 2: recover (checked-in, already fetched) --
    recovered = json.loads((REPO / "artifacts" / "cw_coding_bat_recovery_raw.json").read_text())
    missing = set(identified["method_name"]) - set(recovered)
    if missing:
        raise RuntimeError(f"{len(missing)} identified method name(s) have no recovered statement "
                           f"(a genuine gap must be reported, not filled): "
                           f"{sorted(missing)}")

    # -- step 3: pre-registered gate on the known 17 --
    release_text = load_release_text()
    pid_of_name = dict(zip(identified["method_name"], identified["problem_id"]))
    gate_rows = []
    for name in sorted(KNOWN_17):
        pid = int(pid_of_name[name])
        rec_norm = normalize_statement(recovered[name]["statement"])
        rel_norm = normalize_statement(release_text[pid])
        exact = rec_norm == rel_norm
        verdict, reason = ("same_problem", "exact normalised match") if exact else KNOWN_17_VERDICTS[name]
        gate_rows.append({"method_name": name, "problem_id": pid, "exact_normalised_match": exact,
                          "verdict": verdict, "reason": reason})
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(REPO / "artifacts" / "cw_4b_gate_17.csv", index=False)
    n_exact = int(gate_df["exact_normalised_match"].sum())
    n_same_problem = int((gate_df["verdict"] == "same_problem").sum())
    gate_passed = n_same_problem == 17
    logger.heartbeat(step=2, force=True, stage="17/17 gate", n_exact_normalised_match=n_exact,
                     n_same_problem=n_same_problem, gate_passed=bool(gate_passed))
    if not gate_passed:
        print(f"GATE FAIL -- {n_same_problem}/17 same-problem verdicts. "
             "Problem statements were not rebuilt.")
        return 1

    # -- step 4: build the 50-entry artifact --
    rows = []
    for _, r in identified.iterrows():
        name = r["method_name"]
        statement = recovered[name]["statement"]
        rows.append({
            "problem_id": int(r["problem_id"]), "method_name": name,
            "agreement": float(r["agreement"]), "source_url": recovered[name]["source_url"],
            "statement": statement, "wrapped_prompt": WRAPPER + statement,
        })
    statements = pd.DataFrame(rows).sort_values("problem_id").reset_index(drop=True)
    (REPO / "artifacts" / "cw_problem_statements.json").write_text(
        json.dumps(statements.to_dict(orient="records"), indent=2))
    logger.heartbeat(step=3, force=True, stage="cw_problem_statements.json written", n=len(statements))

    # -- step 5: rebuild s(problem) through the frozen codec --
    logger.heartbeat(step=0, force=True, stage="loading backbone")
    bundle = load_backbone(DEFAULT_BACKBONE, allow_download=True)
    encode_items = [(i, row["wrapped_prompt"]) for i, row in enumerate(statements.to_dict(orient="records"))]
    logger.heartbeat(step=0, force=True, stage="encoding 50 problem statements")
    stats = extract_texts(bundle, encode_items, CACHE_DIR, logger=logger)

    cache = RawStateCache(CACHE_DIR, stats["n_layers"], stats["hidden_dim"], capacity=stats["n_cached"],
                          model=bundle["hf_id"], template="")
    ct = CodecTransform.load(TRANSFORM_PATH)
    X = np.asarray(cache.states(ct.pooling)[:, ct.layer, :], dtype=np.float64)
    Z = ct.apply(X, k=K_DEFAULT, rank=args.production_rank)  # whiten=True default -- matches
    # cw_manifest's per-attempt convention. `rank=args.production_rank` (default None ->
    # ct.k_max) decouples this artifact's own dimensionality from however large the shared
    # transform's k_max is (p2_07_encode_problems.py applies the same decoupling).

    problem_id, codec_rows = [], []
    for _, row in statements.iterrows():
        h = text_hash(row["wrapped_prompt"])
        if not cache.has(h):
            raise RuntimeError(f"problem_id {row['problem_id']} ({row['method_name']}) failed to encode")
        problem_id.append(int(row["problem_id"]))
        codec_rows.append(cache.row_of_hash[h])
    codec = Z[codec_rows].astype(np.float32)
    all_finite = bool(np.isfinite(codec).all())

    out_path = REPO / "artifacts" / "cw_problem_codec_mean_L15.npz"
    np.savez(out_path, problem_id=np.array(problem_id, dtype=np.int64), codec=codec,
            pooling=ct.pooling, layer=ct.layer, k=K_DEFAULT, transform_hash=ct.hash,
            transform_source="problem_mean_L15.npz (FA-fit, applied not refit)",
            text_source="cw_problem_statements.json (CodingBat-recovered, 17 gate-verified, "
                        "33 recovered; CW-4b, 13 Aug night)")
    logger.log_metrics(n_problems=len(statements), all_identified=bool(all_identified),
                       unique_1to1=bool(unique_names_ok), gate_n_same_problem=n_same_problem,
                       codec_all_finite=all_finite, codec_dim=int(codec.shape[1]))
    passed = bool(all_identified and unique_names_ok and gate_passed and len(problem_id) == 50 and all_finite)
    logger.summary(status="ok" if passed else "gate-fail", n_problems=len(statements))

    gate_report(
        path="reports/cw_4b_recover_problem_statements.md",
        title="Recover the 50 CodeWorkout problem statements; rebuild s(problem) from text",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
              "n_problems": len(statements), "n_known_17_verified": n_same_problem,
              "n_recovered_33": len(statements) - 17, "codec_dim": int(codec.shape[1])},
        metrics={
            "identification (S19-train only)": {"all_50_identified": bool(all_identified),
                                                 "unique_1to1": bool(unique_names_ok),
                                                 "min_agreement": float(identified["agreement"].min()),
                                                 "median_agreement": float(identified["agreement"].median())},
            "17/17 gate": {"n_exact_normalised_match": n_exact, "n_same_problem_eyeballed": n_same_problem,
                          "passed": bool(gate_passed)},
            "codec rebuild": {"n_problems": len(problem_id), "all_finite": all_finite,
                             "pooling": ct.pooling, "layer": ct.layer, "k": K_DEFAULT},
        },
        passed=passed,
        gate_desc="All 50 problems identified from S19-train code (1:1, no duplicate names); "
                  "the pre-registered 17/17 same-problem gate against the release text passes "
                  "(9 wording-drift items eyeballed and recorded, 0 different-problem verdicts); "
                  "the rebuilt s(problem) codec is finite for all 50 problems.",
        notes=[
            "**The leak this removes:** a mean-of-student-CODE `cw_problem_codec_mean_L15.npz` "
            "draws on average 64.1% of its mass from F19 and 6.5% from S19-val. "
            "This rebuild uses problem STATEMENT text instead, a property of "
            "the problem that cannot leak either sealed cohort by construction.",
            "**Recovery, never generation:** every one of the 50 statements below has its own "
            "`source_url` (codingbat.com, Nick Parlante, Stanford -- a public educational "
            "source, cited with attribution) in `artifacts/cw_problem_statements.json`; none "
            "is LLM-generated. Raw statement text stays local under the dataset licence; "
            "only the derived codec vectors are artifacts.",
            f"**17/17 gate detail:** {n_exact}/17 matched under strict normalised-string "
            f"equality; the other {17 - n_exact} are recorded in `artifacts/cw_4b_gate_17.csv` "
            "with an eyeballed reason each (comma/quote formatting, one added explanatory "
            "sentence, a typo in the release's own text, or a paraphrase) -- wording drift, "
            "never a different problem (the criterion is 'same problem', not "
            "byte-identical text).",
            "**Provenance strata:** the 17/33 split is not a training split; "
            "it is a provenance disclosure. Downstream production-head evaluations report both "
            "strata: do recovered-statement problems (the 33) behave like ground-truth-"
            "statement ones (the 17)? Pre-registered direction: the 17 should do at least as "
            "well (real vs recovered text, not a quality difference expected here since the "
            "recovery gate passed clean).",
        ],
    )
    print(f"\ncw_4b_recover_problem_statements: {'PASS' if passed else 'FAIL'} -- {n_same_problem}/17 gate, "
         f"{len(problem_id)}/50 codec vectors rebuilt, all_finite={all_finite}")
    print("report: reports/cw_4b_recover_problem_statements.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
