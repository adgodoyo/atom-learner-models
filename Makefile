# Entry points. Everything runs through uv, so a fresh clone uses the pinned environment.
# Targets marked [data] need the corpora (see DATA.md); [gpu] need one 48 GB GPU and the backbone.
.PHONY: help setup test smoke weights figures data-fa data-cw eval-cw-classifier \
        codebleu-all50 lint clean

help:             ## List the targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ :/'

setup:            ## Create the pinned environment (Python 3.11) from pyproject.toml + uv.lock
	uv sync --extra dev --extra baselines

test:             ## Unit tests; tests needing the corpora or the backbone skip cleanly
	uv run pytest

smoke:            ## End-to-end pipeline on synthetic data (no corpora, no GPU)
	uv run python scripts/p0_03_smoke.py

weights:          ## Rebuild every trained checkpoint from checkpoints/ (downloads the two large ones)
	uv run python scripts/materialize_checkpoints.py

figures:          ## Redraw the thesis's data figures from published files
	uv run python paper/figures/fa_figures.py
	uv run python paper/figures/cw_figures.py
	uv run python paper/figures/planner_figures.py
	uv run python paper/figures/generative_figure.py

data-fa:          ## [data] FoundationalASSIST manifest, splits, lockbox hash check
	uv run python scripts/fa_00_manifest_rebuild.py

data-cw:          ## [data] CodeWorkout event graph, sessions, splits
	uv run python scripts/cw_1_event_graph.py
	uv run python scripts/cw_2_sessions.py
	uv run python scripts/cw_4c_step0_1_splits.py

eval-cw-classifier: ## [data] Table 9: CodeWorkout first-submission correctness and its baselines
	uv run python scripts/cw_first_submission_baselines.py

codebleu-all50:   ## Table 11's all-50-problem CodeBLEU from the published strata
	uv run python scripts/cw_codebleu_all50.py

# Training, generation and planner runs need run-specific arguments: see REPRODUCE.md.

lint:             ## Static checks
	uv run ruff check .

clean:            ## Remove run outputs and caches
	rm -rf runs/* .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
