"""Tests for src/atom/codec/cw_extract.py (P2-5b step 1: unified per-scope extraction)."""

import numpy as np
import torch

from atom.codec.cw_extract import BoundaryCache, extract_scope

_VOCAB: dict[str, int] = {}


def _wid(w: str) -> int:
    return _VOCAB.setdefault(w, len(_VOCAB) + 2)


class _Enc(dict):
    def __getattr__(self, k):
        return self[k]


class FakeTok:
    """Whitespace tokenizer supporting both call shapes cw_extract.extract_scope needs:
    plain `.input_ids` (answer_token_span's contract) and `return_offsets_mapping=True`
    (boundary_token_positions's contract)."""

    def __call__(self, s, add_special_tokens=True, return_offsets_mapping=False):
        words = s.split()
        ids = ([1] if add_special_tokens else []) + [_wid(w) for w in words]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            spans, pos = [], 0
            for w in words:
                start = s.index(w, pos)
                end = start + len(w)
                spans.append((start, end))
                pos = end
            out["offset_mapping"] = spans
        return _Enc(out)


class FakeModel:
    """Returns hidden states = token position index broadcast over `dim`, 2 layers,
    so last/mean/boundary_mean are exactly checkable arithmetic on position indices."""

    def __init__(self, dim=4):
        self.dim = dim

    def __call__(self, input_ids, output_hidden_states=True):
        T = input_ids.shape[1]
        layer0 = torch.arange(T, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(1, T, self.dim)
        layer1 = layer0 * 10
        return type("Out", (), {"hidden_states": (layer0, layer1)})()


def _bundle():
    return {"model": FakeModel(), "tokenizer": FakeTok(), "device": "cpu"}


def test_extract_scope_code_alone_pools_whole_sequence():
    b = _bundle()
    code = "int a=1;\nreturn a;\n"
    out = extract_scope(b, code, prefix="", boundary_cap=32)
    n_words = len(code.split())
    full_ids_len = 1 + n_words  # BOS + words
    # last-token = position (full_ids_len - 1) on layer0
    assert out["last"][0, 0] == full_ids_len - 1
    # mean over the WHOLE sequence (start=0, end=full_ids_len) on layer0
    assert np.isclose(out["mean"][0, 0], np.mean(np.arange(full_ids_len)))
    assert out["last"].shape == out["mean"].shape == out["boundary_mean"].shape


def test_extract_scope_prompt_code_pools_only_the_code_span():
    b = _bundle()
    prefix = "solve this ‖ "
    code = "return 0;"
    out = extract_scope(b, code, prefix=prefix, boundary_cap=32)
    # code span starts right after the prefix's own tokens (BOS + prefix words)
    n_prefix_words = len(prefix.split())
    start = 1 + n_prefix_words
    n_code_words = len(code.split())
    end = start + n_code_words
    assert np.isclose(out["mean"][0, 0], np.mean(np.arange(start, end)))
    assert out["last"][0, 0] == end - 1


def test_extract_scope_code_alone_boundary_mean_lands_on_content_not_bos():
    """Regression test for the 11 Aug off-by-one: FakeTok's `add_special_tokens=True`
    prepends exactly one BOS id (mirroring google/gemma-4-E4B), so with one boundary at
    the ';' and one content token after BOS, the correct read is full_ids[1] (the
    content token, hidden-state position value 1 on FakeModel's layer0), never
    full_ids[0] (BOS, value 0) -- the bug the old `tok_base=0` produced."""
    b = _bundle()
    code = "x;\n"
    out = extract_scope(b, code, prefix="", boundary_cap=32)
    assert out["boundary_mean"][0, 0] == 1.0


def test_extract_scope_boundary_mean_within_code_span_range():
    b = _bundle()
    prefix = "q ‖ "
    code = "int a=1;\nint b=2;\nreturn a+b;\n"
    out = extract_scope(b, code, prefix=prefix, boundary_cap=32)
    start = 1 + len(prefix.split())
    end = start + len(code.split())
    # boundary_mean is an average of positions in [start, end) on layer0 -> must lie inside
    assert start <= out["boundary_mean"][0, 0] < end


def test_boundary_cache_roundtrip_and_grow(tmp_path):
    L, D = 2, 4
    c = BoundaryCache(tmp_path, n_layers=L, hidden_dim=D, capacity=1, model="m", scope="s")
    c.add("h1", {"last": np.ones((L, D)), "mean": np.full((L, D), 2.0),
                "boundary_mean": np.full((L, D), 3.0)})
    c.add("h2", {"last": np.zeros((L, D)), "mean": np.full((L, D), 5.0),
                "boundary_mean": np.full((L, D), 6.0)})  # exceeds capacity=1 -> must grow
    c.flush()
    assert c.n_cached == 2 and c.capacity >= 2

    c2 = BoundaryCache(tmp_path, L, D, capacity=999, model="m", scope="s")
    assert c2.has("h1") and c2.has("h2")
    assert np.allclose(c2.states("boundary_mean")[c2.row_of_hash["h1"]], 3.0)
    assert np.allclose(c2.states("mean")[c2.row_of_hash["h2"]], 5.0)
