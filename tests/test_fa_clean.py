"""Unit tests for the vendored FA HTML/MathML cleaner (data-free, fast)."""

import pandas as pd

from atom.data.fa_clean import clean_problem_body


def test_empty_and_nan_return_empty():
    assert clean_problem_body("") == ""
    assert clean_problem_body(None) == ""
    assert clean_problem_body(pd.NA) == ""


def test_plain_text_stripped():
    assert clean_problem_body("<p>What is 2+2?</p>") == "What is 2+2?"


def test_mathml_fraction():
    html = "<p>Compute <math><mfrac><mn>1</mn><mn>2</mn></mfrac></math> now</p>"
    out = clean_problem_body(html)
    assert "(1/2)" in out
    assert "Compute" in out and "now" in out


def test_mathml_superscript():
    assert "x^2" in clean_problem_body("<math><msup><mi>x</mi><mn>2</mn></msup></math>")


def test_image_alt_becomes_placeholder():
    out = clean_problem_body('<p>See <img alt="triangle ABC"> here</p>')
    assert "[Image: triangle ABC]" in out


def test_image_without_alt():
    assert "[image]" in clean_problem_body('<p>Look <img src="foo.png"></p>')


def test_table_rendered_with_separators():
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    out = clean_problem_body(html)
    assert "[Table:" in out and "a | b" in out and "1 | 2" in out


def test_entities_and_whitespace():
    assert "&" in clean_problem_body("<p>Tom &amp; Jerry</p>")
    assert clean_problem_body("<p>a    b\n\nc</p>") == "a b c"
