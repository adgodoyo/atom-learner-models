"""Unit tests for extraction logic + the raw-state cache (no model)."""

from types import SimpleNamespace

import numpy as np
import torch

from atom.codec.extract import (
    DEFAULT_TEMPLATE,
    RawStateCache,
    answer_token_span,
    encode_one,
    qa_prefix_full,
    text_hash,
)

_VOCAB: dict[str, int] = {}


def _wid(w: str) -> int:
    return _VOCAB.setdefault(w, len(_VOCAB) + 2)  # 0/1 reserved, 1 = BOS


class FakeTok:
    """Whitespace tokenizer with a stable vocab and a BOS token, for span tests."""

    def __call__(self, s, add_special_tokens=True):
        return SimpleNamespace(input_ids=[1] + [_wid(w) for w in s.split()])


class GraphAttachedModel:
    """Returns pre-existing graph-attached states even when called under no_grad."""

    def __init__(self):
        self.hidden_states = (
            torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).requires_grad_(),
            torch.arange(12, 24, dtype=torch.float32).reshape(1, 3, 4).requires_grad_(),
        )

    def __call__(self, **_kwargs):
        return SimpleNamespace(hidden_states=self.hidden_states)


def test_qa_prefix_full_default_template():
    prefix, full = qa_prefix_full("What is 1/2 + 1/4?", "3/4")
    assert prefix == "What is 1/2 + 1/4?\nAnswer: "
    assert full == "What is 1/2 + 1/4?\nAnswer: 3/4"
    assert full.startswith(prefix)


def test_answer_token_span_locates_answer():
    tok = FakeTok()
    prefix, full = qa_prefix_full("What is 1/2 + 1/4?", "3/4")
    start, end, full_ids = answer_token_span(tok, prefix, full)
    # answer tokens = full_ids[start:end]; here the single answer word "3/4"
    assert full_ids[start:end] == [_wid("3/4")]
    assert end == len(full_ids)


def test_answer_token_span_empty_answer_falls_back_to_last():
    tok = FakeTok()
    prefix, full = qa_prefix_full("Q?", "")
    start, end, full_ids = answer_token_span(tok, prefix, full)
    assert start == end - 1  # never an empty span


def test_text_hash_stable():
    assert text_hash("abc") == text_hash("abc")
    assert text_hash("abc") != text_hash("abd")


def test_encode_one_detaches_graph_attached_hidden_states():
    bundle = {"model": GraphAttachedModel(), "tokenizer": FakeTok(), "device": "cpu"}
    last, mean = encode_one(bundle, "question", "question answer")
    assert last.shape == mean.shape == (2, 4)
    assert last.dtype == mean.dtype == np.float32
    assert np.isfinite(last).all() and np.isfinite(mean).all()


def test_raw_state_cache_add_resume_roundtrip(tmp_path):
    L, D = 4, 8
    c = RawStateCache(tmp_path, n_layers=L, hidden_dim=D, capacity=3, model="m", template="t")
    c.add("h1", np.ones((L, D)), np.full((L, D), 2.0), ids=[10, 11])
    c.add("h2", np.zeros((L, D)), np.full((L, D), 3.0), ids=[12])
    c.flush()
    assert c.n_cached == 2
    assert c.states("last").shape == (2, L, D)

    # Reopen (resume): existing hashes are present, nothing re-encoded.
    c2 = RawStateCache(tmp_path, n_layers=L, hidden_dim=D, capacity=3, model="m", template="t")
    assert c2.has("h1") and c2.has("h2") and c2.n_cached == 2
    assert np.allclose(c2.states("mean")[c2.row_of_hash["h1"]], 2.0)
    assert c2.ids_of_hash["h1"] == [10, 11]


def test_raw_state_cache_grows_past_capacity(tmp_path):
    L, D = 3, 4
    c = RawStateCache(tmp_path, n_layers=L, hidden_dim=D, capacity=2, model="m", template="t")
    for i in range(5):  # exceeds the initial capacity of 2 -> must grow, not raise
        c.add(f"h{i}", np.full((L, D), float(i)), np.full((L, D), float(i) + 10), ids=[i])
    c.flush()
    assert c.n_cached == 5 and c.capacity >= 5
    for i in range(5):  # data preserved across growth
        assert np.allclose(c.states("last")[c.row_of_hash[f"h{i}"]], float(i))
    c2 = RawStateCache(tmp_path, L, D, c.capacity, "m", "t")  # reopen
    assert c2.n_cached == 5
    assert np.allclose(c2.states("mean")[c2.row_of_hash["h4"]], 14.0)


def test_cache_grows_on_reopen_when_full(tmp_path):
    # Regression: reopening a full cache with a larger capacity arg (as extract_texts does)
    # must use the actual on-disk size, then grow on the next add.
    L, D = 2, 3
    c = RawStateCache(tmp_path, L, D, capacity=2, model="m", template="t")
    c.add("a", np.ones((L, D)), np.ones((L, D)))
    c.add("b", np.full((L, D), 2.0), np.full((L, D), 2.0))
    c.flush()
    c2 = RawStateCache(tmp_path, L, D, capacity=999, model="m", template="t")
    assert c2.capacity == 2  # actual on-disk, not the passed 999
    c2.add("c", np.full((L, D), 3.0), np.full((L, D), 3.0))  # exceeds 2 -> must grow
    c2.flush()
    assert c2.n_cached == 3
    assert np.allclose(c2.states("last")[c2.row_of_hash["c"]], 3.0)
    assert np.allclose(c2.states("last")[c2.row_of_hash["a"]], 1.0)  # preserved


def test_default_template_shape():
    assert "{question}" in DEFAULT_TEMPLATE and "{answer}" in DEFAULT_TEMPLATE
