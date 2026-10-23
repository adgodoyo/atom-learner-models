from __future__ import annotations

from scripts.sim2real_w import candidate_pool, draw_episode, rl2_fit_ids


def _events(n: int) -> list[dict]:
    return [{"episode_id": f"e{i}", "problem_id": i, "order": i} for i in range(n)]


def test_candidate_partition_is_disjoint_and_excludes_targets():
    prefix, candidates, targets = candidate_pool(_events(30), 3)
    assert {x["episode_id"] for x in prefix}.isdisjoint(x["episode_id"] for x in candidates)
    assert {x["episode_id"] for x in candidates}.isdisjoint(x["episode_id"] for x in targets)
    assert [x["episode_id"] for x in prefix] == ["e0", "e1", "e2"]
    assert [x["episode_id"] for x in targets] == [f"e{i}" for i in range(25, 30)]


def test_short_own_pool_uses_distinct_up_to_cap_without_replacement():
    episode = draw_episode("L", {"L": _events(12)}, seed=20260904)
    assert 1 <= len(episode["candidates"]) <= 4  # feasible start, then five targets
    assert len({x["episode_id"] for x in episode["candidates"]}) == len(episode["candidates"])


def test_rl2_fit_partition_is_sorted_and_value_joined():
    events = {f"L{i:03d}": _events(3) for i in range(5)}
    split = {key: "train" for key in events}
    assert rl2_fit_ids(events, split, expected=3) == ["L000", "L001", "L002"]
