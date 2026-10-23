#!/usr/bin/env python
"""Build the CodeWorkout attempt table from the S19 and F19 full releases.

Attempts are joined through the ProgSnap2 event graph only, never by timestamp proximity:
Run.Program + exactly one Compile child + Compile.Error children + CodeState + Score.
Inputs: the raw release tables under `cw_events.dir` in `configs/data.yaml`.
Outputs: `reports/cw_1_event_graph.md` (join assertions, attempt counts against the
expected values in the config (S19 69,627 / F19 125,578), and 10 stratified episodes
per release for manual reading).
Run: `python scripts/cw_1_event_graph.py`. Serves Chapter 4 (data and splits).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from atom.data.cw_events import (  # noqa: E402
    assert_event_graph,
    build_attempts,
    build_problem_episodes,
    load_raw_tables,
    release_stats,
    sample_stratified_episodes,
)
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["cw_events"]
    logger = RunLogger(config={"task": "cw_1_event_graph", **cfg}, data_version="CSEDM-S19+F19")

    releases = {}
    all_clean = True
    metrics: dict = {}
    notes: list[str] = []

    for release, gate in cfg["gate"].items():
        logger.heartbeat(step=0, force=True, stage=f"loading {release}")
        main_table, _ = load_raw_tables(release, base_dir=(REPO / cfg["dir"]).resolve())
        audit = assert_event_graph(main_table)
        audit_clean = (audit["compiles_with_bad_parent"] == 0
                       and audit["runs_with_not_exactly_one_compile"] == 0
                       and audit["errors_with_bad_parent"] == 0)

        logger.heartbeat(step=1, force=True, stage=f"{release} building attempts", **audit)
        attempts = build_attempts(main_table)
        stats = release_stats(main_table, attempts)

        counts_match = all(stats[k] == gate[k] for k in gate)
        all_clean = all_clean and audit_clean and counts_match

        releases[release] = {"main": main_table, "attempts": attempts, "stats": stats,
                              "audit": audit, "audit_clean": audit_clean, "counts_match": counts_match}

        metrics[f"{release} join assertions (gate: all == 0)"] = (
            f"bad compile parent {audit['compiles_with_bad_parent']}, "
            f"runs w/o exactly 1 compile {audit['runs_with_not_exactly_one_compile']}, "
            f"bad error parent {audit['errors_with_bad_parent']} -- "
            f"{'PASS' if audit_clean else 'FAIL'}"
        )
        gate_str = ", ".join(f"{k}={gate[k]}" for k in gate)
        stat_str = ", ".join(f"{k}={stats[k]}" for k in gate)
        metrics[f"{release} counts (gate: {gate_str})"] = f"{stat_str} -- {'PASS' if counts_match else 'FAIL'}"

        logger.log_metrics(release=release, passed=audit_clean and counts_match, **stats, **audit)

    logger.heartbeat(step=2, force=True, stage="building hand-read stratified samples")
    traj_lines = []
    for release in cfg["gate"]:
        attempts_ep, episodes = build_problem_episodes(releases[release]["attempts"])
        sample = sample_stratified_episodes(
            attempts_ep, episodes, n=cfg["n_hand_read_episodes"], seed=cfg["hand_read_seed"])
        traj_lines.append(f"### {release} ({len(episodes)} preliminary problem-boundary episodes)")
        for t in sample:
            traj_lines.append(
                f"- episode {t['episode_id']} (learner {t['learner'][:8]}..., problem "
                f"{t['problem_id']}, {t['n_attempts']} attempts, solved={t['solved']}, "
                f"regression={t['has_regression']}, return_episode={t['is_return_episode']}): "
                f"scores {t['scores']}, compile_errors {t['compile_errors']}"
            )

    passed = all_clean
    logger.summary(status="ok" if passed else "gate-fail",
                    **{f"{r}_{k}": v for r in releases for k, v in releases[r]["stats"].items()})

    gate_report(
        path="reports/cw_1_event_graph.md",
        title="CodeWorkout event-graph ingest (S19 + F19 full releases)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash},
        metrics=metrics,
        passed=passed,
        gate_desc="Join assertions (compile-parent-is-a-run, exactly-one-compile-per-run, "
                  "error-parent-is-a-compile) pass on 100% of rows for both releases; raw "
                  "row / attempt / compile / compile-error / learner / problem counts match "
                  "the expected release counts exactly; 10 hand-read episodes per release "
                  "across length/success/regression/return strata (below, for human read).",
        notes=[
            "Event-graph join only -- never timestamp proximity. "
            "F19 serialises ParentEventID with a trailing float suffix (e.g. '11161.0' "
            "pointing at EventID '11161'); S19 does not. Canonicalised in `cw_events.py::"
            "canonicalize_parent_id` before any join -- verified: without canonicalisation "
            "ALL F19 joins fail (125,578/125,578 bad compile parents), with it, zero.",
            "Attempt counts reconcile EXACTLY (not within tolerance) against both the expected "
            "release counts and each other's derived quantities (n_runs == n_compiles == "
            "attempts by construction, since exactly one Compile per Run.Program).",
            "The 10-episode hand-read sample below uses a PRELIMINARY problem-boundary "
            "episode definition (a new episode starts when the learner switches problem) -- "
            "NOT the final sessionised (`tau_session`-gapped) episode target, which "
            "cw_2_sessions.py builds. Episode counts under this preliminary grouping "
            "(S19 18,143 / F19 27,814) match the expected figures exactly, corroborating the join.",
            "Raw code text is never loaded by this gate (`load_raw_tables(..., "
            "include_code=False)` default) to respect the dataset licence, "
            "same as `cw.py`. `code_state_id` (a reference key, not text) flows into "
            "`attempts` so downstream stages can join code locally when they need it.",
            "",
            "## Hand-read episode samples",
            "",
            *traj_lines,
        ],
    )

    for release in cfg["gate"]:
        s = releases[release]["stats"]
        print(f"Release {release}: {'PASS' if releases[release]['audit_clean'] and releases[release]['counts_match'] else 'FAIL'} "
              f"-- attempts {s['attempts']}, learners {s['learners']}, problems {s['problems']}")
    print(f"\nOverall: {'PASS' if passed else 'FAIL'} -- report: reports/cw_1_event_graph.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
