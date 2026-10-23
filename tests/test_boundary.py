"""Tests for src/atom/codec/boundary.py (P2-5b step 1: boundary-state extraction)."""

import torch

from atom.codec.boundary import boundary_char_offsets, boundary_token_positions, extract_boundary_states


def test_boundary_char_offsets_finds_semicolons_braces_and_lines():
    code = "int a=1;\nif (a>0) {\n  return a;\n}\n"
    offsets = boundary_char_offsets(code, cap=32)
    # ';' after "int a=1" (offset 8), end of non-blank line 1 (offset 8, same char -> dedup),
    # '{' line doesn't end in ; or } but is a non-blank line -> its end offset included,
    # ';' after "return a" and both '}'/line-end for the closing brace line.
    assert 8 in offsets  # end of "int a=1;"
    assert offsets == sorted(set(offsets))  # sorted, deduped


def test_boundary_char_offsets_respects_cap_and_keeps_span():
    code = "".join(f"x{i};\n" for i in range(100))  # 100 ';'-terminated lines
    offsets = boundary_char_offsets(code, cap=32)
    assert len(offsets) <= 32
    assert offsets[0] < offsets[-1]
    assert offsets[-1] <= len(code)


def test_boundary_char_offsets_empty_code():
    assert boundary_char_offsets("", cap=32) == []


class _FakeFastTokenizer:
    """Whitespace-token stub mimicking a fast tokenizer's offset_mapping contract."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        spans, i = [], 0
        for tok in text.split(" "):
            start = text.index(tok, i)
            end = start + len(tok)
            spans.append((start, end))
            i = end
        return {"offset_mapping": spans, "input_ids": list(range(len(spans)))}


def test_boundary_token_positions_maps_char_offsets_to_tokens():
    tok = _FakeFastTokenizer()
    text = "aa bb cc dd"  # tokens at char spans (0,2) (3,5) (6,8) (9,11)
    positions = boundary_token_positions(tok, text, char_offsets=[2, 8, 11], tok_base=5)
    # offset 2 -> token 0 (span end 2 >= 2) -> position 5+0
    # offset 8 -> token 2 (span end 8 >= 8) -> position 5+2
    # offset 11 -> token 3 (span end 11 >= 11) -> position 5+3
    assert positions == [5, 7, 8]


def test_boundary_token_positions_offsets_never_exceed_last_token():
    tok = _FakeFastTokenizer()
    text = "aa bb"
    positions = boundary_token_positions(tok, text, char_offsets=[100], tok_base=0)
    assert positions == [1]  # clamps to the last token, doesn't index out of range


_VOCAB: dict[str, int] = {}


def _wid(w: str) -> int:
    return _VOCAB.setdefault(w, len(_VOCAB) + 2)


class _Enc(dict):
    def __getattr__(self, k):
        return self[k]


class _FakeTok:
    """Whitespace tokenizer prepending one BOS id under `add_special_tokens=True`
    (mirrors google/gemma-4-E4B's real behaviour) -- both call shapes
    `extract_boundary_states` needs."""

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


class _FakeModel:
    """hidden_states[layer][0, pos, :] == pos (broadcast), one layer -- boundary
    positions are exactly checkable against the token-index value itself."""

    def __call__(self, input_ids, output_hidden_states=True):
        T = input_ids.shape[1]
        layer0 = torch.arange(T, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(1, T, 4)
        return type("Out", (), {"hidden_states": (layer0,)})()


def test_extract_boundary_states_code_alone_lands_on_content_not_bos():
    """Regression test for the 11 Aug off-by-one (same defect, same fix, as
    cw_extract.py's sibling test): one boundary at ';', one content token after BOS ->
    must read full_ids[1] (value 1), never full_ids[0] (BOS, value 0)."""
    bundle = {"model": _FakeModel(), "tokenizer": _FakeTok(), "device": "cpu"}
    states, n_found = extract_boundary_states(bundle, code="x;\n", prefix="", cap=32)
    assert n_found == 1
    assert states[0, 0, 0] == 1.0
