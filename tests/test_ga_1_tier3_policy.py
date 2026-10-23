"""Lane T0 -- GA-1 tier 3: the shuffle must relabel without changing what is asked."""

from __future__ import annotations

import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ga_1_tier3_policy as P  # noqa: E402
import ga_1_tier3_states as S  # noqa: E402


class FakeTok:
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, text, **kw):
        if isinstance(text, str):
            return {"input_ids": [0] * max(1, len(text.split()))}
        n = len(text)
        enc = dict(input_ids=torch.zeros(n, 3, dtype=torch.long),
                   attention_mask=torch.ones(n, 3, dtype=torch.long))
        return type("E", (dict,), {"to": lambda self, d: self})(enc)


def _statements(n):
    return {p: {"method_name": f"m{p}", "statement": "s " * 6} for p in range(n)}


def _cands(n):
    return [{"problem_id": p, "episode_id": f"e{p}", "order": p} for p in range(n)]


def test_render_state_places_the_permuted_candidate_under_identifier_A():
    tok = FakeTok()
    cands = _cands(10)
    perm = S.crn_permutation(10, "arm", "L1", 3, 8, 0)
    prompt, shuffled = P.render_state([], cands, 8, 0, {}, _statements(10), tok,
                                      10**6, False, perm)
    assert shuffled[0] is cands[perm[0]]
    first_line = [ln for ln in prompt.splitlines() if ln.startswith("A. ")][0]
    assert f"m{cands[perm[0]]['problem_id']}" in first_line


def test_the_shuffle_changes_the_menu_but_not_the_candidate_set():
    tok = FakeTok()
    cands = _cands(12)
    p1 = S.crn_permutation(12, "armX", "L1", 3, 8, 0)
    p2 = S.crn_permutation(12, "armX", "L1", 3, 8, 1)
    a, sa = P.render_state([], cands, 8, 0, {}, _statements(12), tok, 10**6, False, p1)
    b, sb = P.render_state([], cands, 8, 1, {}, _statements(12), tok, 10**6, False, p2)
    assert a != b, "different states must render different menus under the shuffle"
    assert {c["episode_id"] for c in sa} == {c["episode_id"] for c in sb} == {
        c["episode_id"] for c in cands}


def test_slot_to_candidate_inversion_is_exact():
    """The read-out returns a SHUFFLED slot; picking the wrong candidate is silent."""
    cands = _cands(20)
    for step in range(6):
        perm = S.crn_permutation(20, "arm", "L9", 8, 16, step)
        shuffled = S.apply_permutation(cands, perm)
        for slot in range(20):
            assert shuffled[slot] is cands[perm[slot]]


def test_identifier_A_is_not_a_constant_candidate_across_states():
    """Tier 0's pathology, made impossible: `A` must not denote one item everywhere."""
    labelled = set()
    for step in range(20):
        perm = S.crn_permutation(30, "arm", "L1", 3, 16, step)
        labelled.add(perm[0])
    assert len(labelled) > 10, "identifier A is still effectively pinned to one candidate"
