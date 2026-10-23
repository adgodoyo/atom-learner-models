"""Auto-generated gate reports written as markdown.

A gate report is the human-readable verdict for one evaluation: what ran, under which
config hash / git SHA, the metrics, and PASS/FAIL against the numeric gate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def gate_report(
    path: str | Path,
    title: str,
    meta: dict[str, Any],
    metrics: dict[str, Any],
    passed: bool,
    gate_desc: str,
    notes: list[str] | None = None,
) -> str:
    """Write a markdown gate report and return its text."""
    verdict = "PASS ✅" if passed else "FAIL ❌"
    lines = [
        f"# {title}",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        f"**Gate:** {gate_desc}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Provenance",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for k, v in meta.items():
        lines.append(f"| {k} | {_fmt(v)} |")
    lines += ["", "## Metrics", "", "| metric | value |", "| --- | --- |"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {_fmt(v)} |")
    if notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {n}" for n in notes]
    lines.append("")
    text = "\n".join(lines)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return text


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
