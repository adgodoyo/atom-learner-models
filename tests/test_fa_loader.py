"""Loader tests against the real FA files (skipped if the dataset is absent).

The heavy full-interactions count (1.75M rows) is the gate script's job
(scripts/p1_01_fa_load.py); here we check the small files and the schema so `make test`
stays fast.
"""

import pytest

from atom.data.fa import default_fa_dir, load_problems, load_skills

pytestmark = pytest.mark.skipif(
    not (default_fa_dir() / "Data" / "Problems.csv").exists(),
    reason="FoundationalASSIST dataset not present locally",
)


def test_problems_row_count_and_schema():
    prob = load_problems()
    assert len(prob) == 3395  # ROW count is the gate definition
    for col in ["problem_id", "problem_body", "answer_types", "mc_options", "fillin_answers"]:
        assert col in prob.columns


def test_skills_unique_count():
    assert load_skills()["skill_id"].nunique() == 224


def test_clean_body_on_real_sample():
    prob = load_problems()
    from atom.data.fa_clean import clean_problem_body

    cleaned = prob["problem_body"].astype(str).head(25).map(clean_problem_body)
    # Cleaning must not raise and must strip all HTML tags from the sample.
    assert cleaned.notna().all()
    assert not cleaned.str.contains("<math|<img|<table|<p>", regex=True).any()
