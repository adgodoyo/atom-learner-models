"""Lane T4 rung-A unit gate.

Every test here exists because of a defect class this project has already paid for:
a batching shortcut that permuted results (tier 0), a guard that tested the mechanism
instead of the meaning (RL-2's seed-invariance premise), and an item identity leaking into
a representation-only model.
"""
from __future__ import annotations
import importlib
import json
import pathlib
import sys
import numpy as np
import pytest
import torch
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
from ga_1_tier4_common import CODEC_DIM, FEAT_DIM, STATE_DIM, draw_digest, draw_states, item_split, utility_learner_split, zscore_advantages
from ga_1_tier4_utility import _pick_values, _random_scores, _spearman_rows, batched_scores, pack, ranking_accuracy, ranking_loss
from rl_2 import PointerActorV2

def _episodes(n_learners: int=4, menu: int=6) -> list[dict]:
    return [{'learner': f'L{i}', 'candidates': [{'episode_id': f'E{i}_{j}', 'problem_id': 1 + j % 5} for j in range(menu)]} for i in range(n_learners)]

def _table(n_states: int=5, menu: int=7, seed: int=0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = rng.integers(3, menu + 1, size=n_states)
    total = int(counts.sum())
    return {'state_offset': np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64), 'state_count': counts.astype(np.int64), 'state_features': rng.normal(size=(n_states, STATE_DIM)).astype(np.float32), 'state_learner': np.asarray([f'L{i % 3}' for i in range(n_states)]), 'state_budget': np.full(n_states, 8, dtype=np.int64), 'state_step': np.arange(n_states, dtype=np.int64), 'cand_r': rng.normal(size=total), 'cand_advantage': rng.normal(size=total).astype(np.float32), 'cand_features': rng.normal(size=(total, FEAT_DIM)).astype(np.float32), 'cand_epig': rng.normal(size=total).astype(np.float32), 'cand_problem_id': rng.integers(1, 11, size=total).astype(np.int64)}

def test_item_split_is_a_seed_fixed_partition_of_the_frozen_universe():
    ids = list(range(1, 51))
    train, held = item_split(ids)
    assert len(held) == 10 and len(train) == 40
    assert sorted(train + held) == ids
    assert item_split(list(reversed(ids))) == (train, held)

def test_item_split_refuses_a_universe_that_is_not_the_frozen_fifty():
    with pytest.raises(ValueError):
        item_split(list(range(1, 40)))

def test_utility_learner_split_is_disjoint_and_deterministic():
    fit, internal = utility_learner_split([f'L{i}' for i in range(543)])
    assert len(fit) == 434 and len(internal) == 109
    assert not set(fit) & set(internal)
    assert utility_learner_split([f'L{i}' for i in reversed(range(543))]) == (fit, internal)

def test_draw_is_deterministic_content_addressed_and_order_independent():
    eps = _episodes()
    a = draw_states(eps, (3,))
    b = draw_states(list(reversed(eps)), (3,))
    by_learner = {(s['learner'], s['step']): s['picks'] for s in a}
    for s in b:
        assert by_learner[s['learner'], s['step']] == s['picks']
    assert draw_digest(a, eps) == draw_digest(draw_states(eps, (3,)), eps)

def test_draw_records_the_state_before_each_pick_and_never_repeats_a_candidate():
    eps = _episodes(n_learners=1, menu=5)
    states = [s for s in draw_states(eps, (4,)) if s['budget'] == 4]
    assert [s['step'] for s in states] == [0, 1, 2, 3]
    assert states[0]['picks'] == []
    for s in states:
        assert len(s['picks']) == s['step']
        assert not set(s['picks']) & set(s['remaining'])
        assert len(set(s['picks']) | set(s['remaining'])) == 5

def test_draw_is_bounded_by_the_menu_when_the_budget_exceeds_it():
    eps = _episodes(n_learners=1, menu=2)
    assert len([s for s in draw_states(eps, (16,))]) == 2

def test_zscore_advantages_floors_a_degenerate_state():
    z = zscore_advantages(np.full(5, 0.4))
    assert np.all(np.abs(z) < 1e-06)
    z = zscore_advantages(np.asarray([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0.0, abs=1e-06)

def test_pack_masks_padding_and_respects_the_keep_view():
    table = _table()
    codec = np.arange(11 * CODEC_DIM, dtype=np.float32).reshape(11, CODEC_DIM)
    pid_to_idx = {p: p for p in range(11)}
    keep = table['cand_problem_id'] <= 5
    view = pack(table, codec, pid_to_idx, keep=keep)
    for i, (o, c) in enumerate(zip(table['state_offset'], table['state_count'])):
        n = int(keep[o:o + c].sum())
        assert int(view['mask'][i].sum()) == n
        assert int(view['sizes'][i]) == n
        assert torch.all(view['q'][i, n:] == 0)
        assert torch.all(torch.isneginf(view['epig'][i, n:]))

def test_pack_carries_the_frozen_codec_by_value_not_an_index():
    table = _table()
    codec = np.random.default_rng(1).normal(size=(11, CODEC_DIM)).astype(np.float32)
    view = pack(table, codec, {p: p for p in range(11)})
    o, c = (int(table['state_offset'][0]), int(table['state_count'][0]))
    for j in range(c):
        pid = int(table['cand_problem_id'][o + j])
        assert np.allclose(view['q'][0, j, :CODEC_DIM].numpy(), codec[pid])

def test_utility_input_carries_no_item_identity():
    """Relabelling problem ids while holding codec vectors fixed must not move Û."""
    table = _table()
    codec = np.random.default_rng(2).normal(size=(11, CODEC_DIM)).astype(np.float32)
    base = pack(table, codec, {p: p for p in range(11)})
    relabel = {p: (p + 3) % 11 for p in range(11)}
    shifted = dict(table)
    shifted['cand_problem_id'] = np.asarray([relabel[int(p)] for p in table['cand_problem_id']])
    inverse = {relabel[p]: p for p in range(11)}
    moved = pack(shifted, codec, {q: inverse[q] for q in range(11)})
    assert torch.equal(base['q'], moved['q'])
    actor = PointerActorV2(CODEC_DIM)
    with torch.no_grad():
        assert torch.equal(batched_scores(actor, base['state'], base['q']), batched_scores(actor, moved['state'], moved['q']))

def test_batched_scores_equal_the_imported_actors_own_per_state_forward():
    table = _table(n_states=6)
    codec = np.random.default_rng(3).normal(size=(11, CODEC_DIM)).astype(np.float32)
    view = pack(table, codec, {p: p for p in range(11)})
    actor = PointerActorV2(CODEC_DIM)
    with torch.no_grad():
        batched = batched_scores(actor, view['state'], view['q'])
        for i in range(len(view['state'])):
            n = int(view['sizes'][i])
            q = view['q'][i, :n]
            one = actor(view['state'][i], q[:, :CODEC_DIM], q[:, CODEC_DIM:])
            assert torch.allclose(batched[i, :n], one, atol=1e-06)

def test_batching_never_permutes_rows():
    table = _table(n_states=8)
    codec = np.random.default_rng(4).normal(size=(11, CODEC_DIM)).astype(np.float32)
    view = pack(table, codec, {p: p for p in range(11)})
    actor = PointerActorV2(CODEC_DIM)
    with torch.no_grad():
        whole = batched_scores(actor, view['state'], view['q'])
        halves = torch.cat([batched_scores(actor, view['state'][:3], view['q'][:3]), batched_scores(actor, view['state'][3:], view['q'][3:])])
    assert torch.allclose(whole, halves, atol=1e-06)

def test_ranking_loss_and_accuracy_agree_with_a_hand_computed_case():
    scores = torch.tensor([[2.0, 1.0, 0.0]])
    r = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True]])
    assert ranking_accuracy(scores, r, mask) == pytest.approx(1.0)
    assert ranking_accuracy(-scores, r, mask) == pytest.approx(0.0)
    expected = np.mean([np.log1p(np.exp(-1.0)), np.log1p(np.exp(-2.0)), np.log1p(np.exp(-1.0))])
    assert float(ranking_loss(scores, r, mask)) == pytest.approx(expected, abs=1e-06)

def test_losses_ignore_masked_padding():
    scores = torch.tensor([[2.0, 1.0, 99.0]])
    r = torch.tensor([[3.0, 2.0, 99.0]], dtype=torch.float64)
    full = torch.tensor([[True, True, False]])
    assert ranking_accuracy(scores, r, full) == pytest.approx(1.0)
    assert float(ranking_loss(scores, r, full)) == pytest.approx(float(np.log1p(np.exp(-1.0))), abs=1e-06)

def test_ranking_loss_is_finite_when_a_state_has_no_ordered_pair():
    scores = torch.tensor([[1.0, 2.0]], requires_grad=True)
    r = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    loss = ranking_loss(scores, r, torch.tensor([[True, True]]))
    assert float(loss.detach()) == 0.0
    loss.backward()

def test_spearman_and_pick_skip_states_below_the_registered_minimum():
    scores = np.asarray([[3.0, 2.0, 1.0], [1.0, 0.0, 0.0]])
    r = np.asarray([[3.0, 2.0, 1.0], [5.0, 0.0, 0.0]])
    mask = np.asarray([[True, True, True], [True, True, False]])
    assert _spearman_rows(scores, r, mask)[0] == pytest.approx(1.0)
    assert np.isnan(_spearman_rows(scores, r, mask)[1])
    assert _pick_values(scores, r, mask)[0] == pytest.approx(3.0)
    assert np.isnan(_pick_values(scores, r, mask)[1])

def test_pick_values_read_the_true_realised_advantage_of_the_argmax():
    scores = np.asarray([[0.1, 0.9, 0.5]])
    r = np.asarray([[10.0, -1.0, 3.0]])
    assert _pick_values(scores, r, np.ones((1, 3), bool))[0] == pytest.approx(-1.0)

def test_random_comparator_is_common_random_numbers():
    learners = np.asarray(['L0', 'L1'])
    budgets, steps = (np.asarray([8, 8]), np.asarray([0, 1]))
    a = _random_scores(learners, budgets, steps, (2, 4), 'tag')
    b = _random_scores(learners, budgets, steps, (2, 4), 'tag')
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _random_scores(learners, budgets, steps, (2, 4), 'other'))
    assert not np.array_equal(a[0], a[1])
from ga_1_common import load_problem_statements
from ga_1_tier4_generation_common import collate, encode_row, sample_target, token_budget_batches, windowed_nll

class WordTok:
    """Whitespace tokenizer standing in for gemma: the gate never loads 15 GB of weights."""
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for w in str(text).split():
            ids.append(self.vocab.setdefault(w, len(self.vocab) + 1))
        return {'input_ids': ids}

def test_target_sampling_depends_on_the_seed_and_the_state():
    """Power note: at the registered tau=0.3 this advantage vector puts ~0.79 on index 0,
    so four draws land identically ~39% of the time. The assertion is about DEPENDENCE, so
    it takes enough draws to have power at the sharper tau rather than being relaxed."""
    a = np.asarray([1.0, 0.5, 0.2, -1.0])
    keys = [(s, f'L{i}') for s in (20260902, 20260903) for i in range(40)]
    assert len({sample_target(a, ln, 8, 0, s) for s, ln in keys}) > 1
    per_seed = {s: tuple((sample_target(a, f'L{i}', 8, 0, s) for i in range(40))) for s in (20260902, 20260903)}
    assert per_seed[20260902] != per_seed[20260903]

def test_eos_is_appended_only_when_the_target_fits():
    tok = WordTok()
    short = encode_row('a b c', 'd e', tok, max_len=32)
    assert short['labels'][-1] == tok.eos_token_id
    assert not short['target_censored']
    long_target = ' '.join((str(i) for i in range(60)))
    censored = encode_row('a b c', long_target, tok, max_len=20)
    assert censored['target_censored']
    assert tok.eos_token_id not in censored['labels'], 'EOS at a truncation cutoff teaches early stop'

def test_collate_pads_LEFT_so_every_supervised_span_is_flush_at_the_right_edge():
    tok = WordTok()
    rows = [encode_row('a b', 'c', tok, 32), encode_row('a b c d e', 'f g', tok, 32)]
    b = collate(rows, tok.pad_token_id, 'cpu')
    assert b['input_ids'].shape == b['labels'].shape == b['attention_mask'].shape
    width = b['input_ids'].shape[1]
    for i, r in enumerate(rows):
        n = len(r['input_ids'])
        assert int(b['attention_mask'][i].sum()) == n
        assert torch.all(b['labels'][i, :width - n] == -100)
        assert b['input_ids'][i, width - n:].tolist() == r['input_ids']
        assert b['labels'][i, -1].item() != -100
    assert int(b['window']) == max((r['n_target'] for r in rows)) + 1

def test_collate_position_ids_ignore_left_padding():
    tok = WordTok()
    rows = [encode_row('a b', 'c', tok, 32), encode_row('a b c d e', 'f g', tok, 32)]
    b = collate(rows, tok.pad_token_id, 'cpu')
    for i, r in enumerate(rows):
        n = len(r['input_ids'])
        assert b['position_ids'][i, -n:].tolist() == list(range(n))
        assert torch.all(b['position_ids'][i, :-n] == 0)

def test_batches_are_bounded_by_padded_token_cost_not_row_count():
    rows = [{'input_ids': [0] * n} for n in (10, 10, 10, 400)]
    batches = token_budget_batches(rows, token_budget=64, max_rows=8)
    assert all((len(b) * max((len(rows[i]['input_ids']) for i in b)) <= 64 or len(b) == 1 for b in batches))
    assert sorted((i for b in batches for i in b)) == [0, 1, 2, 3]

def test_batching_never_drops_or_duplicates_a_row():
    rows = [{'input_ids': [0] * (5 + i * 37 % 300)} for i in range(97)]
    batches = token_budget_batches(rows, token_budget=2048, max_rows=4)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(97))
    assert all((len(b) <= 4 for b in batches))

def test_instantaneous_overrun_does_not_fire_during_warmup():
    """The defect this class exists for: the shared trigger arms 60 s into a multi-hour
    stage and projects from the CUMULATIVE rate, so slow warmup batches kill a healthy run.
    Here the first 200 batches are 10x slow and the rest are fast; nothing may fire."""
    import time as _time
    from ga_1_tier4_generation_common import OVERRUN_BURN_IN_BATCHES, InstantaneousOverrun
    o = InstantaneousOverrun('stage', total=10000, threshold_minutes=60)
    now = [0.0]
    real = _time.time
    _time.time = lambda: now[0]
    try:
        for i in range(1, OVERRUN_BURN_IN_BATCHES + 1):
            now[0] += 2.0
            assert o.observe(i) is None
        for i in range(OVERRUN_BURN_IN_BATCHES + 1, OVERRUN_BURN_IN_BATCHES + 400):
            now[0] += 0.2
            o.observe(i)
    finally:
        _time.time = real

def test_instantaneous_overrun_still_fires_on_a_real_slowdown():
    import time as _time
    from ga_1_tier4_generation_common import OVERRUN_BURN_IN_BATCHES, InstantaneousOverrun
    o = InstantaneousOverrun('stage', total=10000, threshold_minutes=60)
    now = [0.0]
    real = _time.time
    _time.time = lambda: now[0]
    try:
        with pytest.raises(RuntimeError, match='instantaneous rate'):
            for i in range(1, OVERRUN_BURN_IN_BATCHES + 200):
                now[0] += 5.0
                o.observe(i)
    finally:
        _time.time = real

def test_instantaneous_overrun_is_disabled_at_threshold_zero():
    from ga_1_tier4_generation_common import OVERRUN_BURN_IN_BATCHES, InstantaneousOverrun
    o = InstantaneousOverrun('stage', total=1000, threshold_minutes=0)
    for i in range(1, OVERRUN_BURN_IN_BATCHES + 50):
        assert o.observe(i) is None

def test_state_stride_keeps_every_learner_and_every_budget():
    """The subsample clause is a STRIDE, not a prefix: a prefix drops whole learners, and
    the claim tier 4 makes is about personalisation."""
    eps = _episodes(n_learners=6, menu=20)
    states = draw_states(eps, (3, 8, 16))
    strided = states[::2]
    assert len(strided) == (len(states) + 1) // 2
    assert {s['learner'] for s in strided} == {s['learner'] for s in states}
    assert {s['budget'] for s in strided} == {3, 8, 16}
    for ln in {s['learner'] for s in states}:
        for b in (3, 8, 16):
            assert any((s['learner'] == ln and s['budget'] == b for s in strided))
    prefix = states[:len(states) // 2]
    assert {s['learner'] for s in prefix} != {s['learner'] for s in states}, 'this is the property that makes a prefix the wrong lever'

def _warm_scenarios(n_learners: int=6, menu: int=20):
    """Scenario cells shaped like `_make_scenarios`: prefix length == the start."""
    scen = {}
    for start in (3, 5, 10):
        scen[f'train_start{start}'] = [{'learner': f'L{i}', 'prefix_events': [{'episode_id': f'p{i}_{j}', 'order': j} for j in range(start)], 'candidates': [{'episode_id': f'c{i}_{j}', 'problem_id': 1 + j % 5, 'order': start + j} for j in range(menu)]} for i in range(n_learners)]
    return scen

def test_warm_draw_never_starts_from_an_empty_belief():
    """The binding rule: minimum 3 real episodes, never 0. At an empty belief the observer
    state is bit-identical across learners, so the decision is person-blind by construction."""
    from ga_1_tier4_common import WARM_STARTS, draw_warm_states
    scen = _warm_scenarios()
    states, episodes = draw_warm_states(scen, 'train', {f'L{i}' for i in range(6)}, (3, 8, 16))
    assert states, 'draw produced no states'
    assert all((s['start'] in WARM_STARTS for s in states))
    assert min((s['start'] for s in states)) >= 3
    for s in states:
        history = len(episodes[s['episode_index']]['prefix_events']) + len(s['picks'])
        assert history >= 3, 'a training state carried fewer than 3 real revealed episodes'

def test_warm_draw_history_is_start_plus_step():
    from ga_1_tier4_common import draw_warm_states
    scen = _warm_scenarios(n_learners=2)
    states, episodes = draw_warm_states(scen, 'train', {'L0', 'L1'}, (8,))
    for s in states:
        assert len(episodes[s['episode_index']]['prefix_events']) == s['start']
        assert len(s['picks']) == s['step']

def test_warm_draw_is_deterministic_and_order_independent():
    from ga_1_tier4_common import draw_warm_states
    scen = _warm_scenarios()
    ids = {f'L{i}' for i in range(6)}
    a, _ = draw_warm_states(scen, 'train', ids, (3, 8, 16))
    b, _ = draw_warm_states(scen, 'train', set(reversed(sorted(ids))), (3, 8, 16))
    key = lambda s: (s['learner'], s['budget'], s['step'])
    assert {key(s): (s['start'], tuple(s['picks'])) for s in a} == {key(s): (s['start'], tuple(s['picks'])) for s in b}

def test_warm_draw_uses_the_whole_registered_start_range():
    from ga_1_tier4_common import WARM_STARTS, draw_warm_states
    scen = _warm_scenarios(n_learners=40)
    states, _ = draw_warm_states(scen, 'train', {f'L{i}' for i in range(40)}, (3, 8, 16))
    assert {s['start'] for s in states} == set(WARM_STARTS)

def test_warm_draw_falls_back_to_a_warm_start_never_to_zero():
    """A learner too short for its drawn start must fall back to another WARM start."""
    from ga_1_tier4_common import draw_warm_states
    scen = _warm_scenarios(n_learners=4)
    for start in (5, 10):
        scen[f'train_start{start}'] = [e for e in scen[f'train_start{start}'] if e['learner'] != 'L0']
    states, episodes = draw_warm_states(scen, 'train', {'L0'}, (3, 8, 16))
    assert states and all((s['start'] == 3 for s in states))

def test_warm_draw_rejects_a_missing_scenario_cell():
    from ga_1_tier4_common import draw_warm_states
    scen = _warm_scenarios()
    del scen['train_start5']
    with pytest.raises(KeyError):
        draw_warm_states(scen, 'train', {'L0'}, (3,))

def test_panel_starts_exclude_the_cut_start_zero_cell():
    from ga_1_tier4_panel_common import MODES, PANEL_STARTS
    assert PANEL_STARTS == (3, 5, 10)
    assert 0 not in PANEL_STARTS
    assert 'swapped_history' in MODES

def test_swap_partner_is_a_derangement_never_a_fixed_point():
    """A random permutation is not a derangement; a fixed point would silently make the
    swap arm identical to the history arm for that learner."""
    from atom.binding.derangement import assert_derangement, derangement_permutation
    for n in (2, 3, 11, 110):
        gen = torch.Generator().manual_seed(n)
        perm = derangement_permutation(n, generator=gen)
        assert_derangement(perm)
        assert all((int(perm[i]) != i for i in range(n)))
        assert sorted((int(x) for x in perm)) == list(range(n))

def test_realised_r_percentile_and_miss_are_computed_against_the_true_menu_r():
    """PRIMARY pointing scores against the training signal r, not against EPIG."""
    menu_pids = np.asarray([7, 11, 23, 31])
    menu_r = np.asarray([0.1, 0.4, 0.25, -0.05])
    for proj, want_pct, want_miss in ((11, 0.75, 0), (23, 0.5, 0), (31, 0.0, 0), (99, None, 1)):
        hit = np.flatnonzero(menu_pids == proj)
        if len(hit):
            r = float(menu_r[int(hit[0])])
            pct = float((menu_r < r).mean())
            assert pct == pytest.approx(want_pct)
            assert want_miss == 0
        else:
            assert want_miss == 1

def test_already_in_history_flag_uses_the_states_revealed_problem_ids():
    history_pids = {3, 12, 40}
    assert int(12 in history_pids) == 1
    assert int(41 in history_pids) == 0

def test_projection_misses_are_filtered_not_averaged_as_zero():
    """The landed panel reported NaN in every primary cell because a miss (NaN) was passed
    into the bootstrap. A miss is not a zero: aggregate over hits, report the miss rate."""
    vals = np.asarray([0.8, np.nan, 0.4, np.nan, 0.6])
    finite = vals[np.isfinite(vals)]
    assert len(finite) == 3
    assert finite.mean() == pytest.approx(0.6)
    assert not np.isfinite(vals.mean()), 'this is the bug: averaging NaN poisons the cell'

def test_contrastive_loss_penalises_a_corpus_copier_by_exactly_softplus_zero():
    """The design's load-bearing arithmetic: a model that scores target and foil alike
    earns softplus(0) = 0.693. That is what makes ignoring the student expensive."""
    import torch.nn.functional as F
    same = F.softplus(torch.tensor(0.0))
    assert float(same) == pytest.approx(0.6931, abs=0.0001)
    good = F.softplus(torch.tensor(-1.0))
    bad = F.softplus(torch.tensor(+1.0))
    assert float(good) < float(same) < float(bad)

def test_belief_vector_is_mu_concat_log_sigma_and_detached():
    """u must be 64-dim [mu || log_sigma] and must never carry gradient into the frozen
    substrate."""
    from ga_1_tier4_rungc import U_DIM
    mu, ls = (torch.randn(32, requires_grad=True), torch.randn(32, requires_grad=True))
    u = torch.cat((mu.detach().flatten(), ls.detach().flatten())).float()
    assert u.shape == (U_DIM,)
    assert not u.requires_grad

def test_hypernetwork_maps_belief_to_an_m_dim_mixing_vector():
    from ga_1_tier4_rungc import HYPER_HIDDEN, HYPER_KIND, LORA_M, U_DIM
    from atom.binding import Hypernetwork
    h = Hypernetwork(U_DIM, LORA_M, kind=HYPER_KIND, hidden=HYPER_HIDDEN, seed=0)
    c = h(torch.randn(5, U_DIM))
    assert c.shape == (5, LORA_M)
    assert float((c[0] - c[1]).detach().abs().max()) > 0

def test_belief_dependence_gate_fails_when_noise_beats_learner_signal():
    """Canary family #9. B-3's measured failure was 11.5% learner vs 135% epsilon; a run
    like that is VOID and this gate must say so rather than reporting a gain."""
    from ga_1_tier4_rungc import belief_dependence_gate

    class ConstantHyper(torch.nn.Module):
        """Ignores the belief entirely -> across-learner spread is exactly 0."""

        def forward(self, u):
            return torch.ones(u.shape[0], 8)
    rows = [{'u': torch.cat((torch.randn(32), torch.zeros(32)))} for _ in range(16)]
    out = belief_dependence_gate(ConstantHyper(), rows, 'cpu')
    assert out['across_learner_spread'] == pytest.approx(0.0, abs=1e-06)
    assert out['gate_pass'] is False
    assert 'VOID' in out['verdict']

def test_belief_dependence_gate_passes_a_genuinely_belief_reading_adapter():
    from ga_1_tier4_rungc import belief_dependence_gate

    class MuReader(torch.nn.Module):
        """Reads mu strongly; log_sigma is fixed at 0 so sigma = 1 drives the eps probe."""

        def forward(self, u):
            return u[..., :8] * 50.0
    rows = [{'u': torch.cat((torch.randn(32) * 10.0, torch.zeros(32)))} for _ in range(32)]
    out = belief_dependence_gate(MuReader(), rows, 'cpu')
    assert out['across_learner_spread'] > out['across_epsilon_spread']
    assert out['gate_pass'] is True

def test_rungc_gradient_reaches_hypernetwork_and_every_lora_basis():
    """Canary #6 shape: every trainable module must receive gradient AND move.

    Measured subtlety this pins: at step 1 the hypernetwork moves NOT AT ALL and only half
    the LoRA tensors move, because LoRA's B is zero-init (an untrained adapter is an exact
    no-op), so dL/dA flows through B=0 and the model output does not yet depend on c. Both
    start moving at step 2. Without this test that transient is indistinguishable from a
    permanently dead hypernetwork -- which is exactly defect #6 in this repo's canary list.
    """
    from ga_1_tier4_rungc import HYPER_HIDDEN, HYPER_KIND, LORA_ALPHA, LORA_M, LORA_R, U_DIM
    from atom.binding import Hypernetwork, attach_lora_basis
    from atom.binding.stub_causal_lm import CausalStubLM
    model = CausalStubLM(vocab_size=64, dim=32, n_layers=2, n_heads=4, pad_token_id=0, seed=0)
    bases, ctx = attach_lora_basis(model, m=LORA_M, r=LORA_R, alpha=LORA_ALPHA, seed=0)
    hyper = Hypernetwork(U_DIM, LORA_M, kind=HYPER_KIND, hidden=HYPER_HIDDEN, seed=0)
    named = [(f'hyper.{n}', p) for n, p in hyper.named_parameters()] + [(f'{k}.{n}', p) for k, b in bases.items() for n, p in b.named_parameters()]
    opt = torch.optim.Adam([p for _, p in named], lr=0.01)
    ids = torch.randint(1, 64, (2, 12))
    u = torch.randn(2, U_DIM)
    moved_per_step = []
    for _ in range(3):
        before = {n: p.detach().clone() for n, p in named}
        ctx.set(hyper(u))
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        loss = out.logits.float().log_softmax(-1)[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).mean().neg()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        moved_per_step.append([n for n, p in named if not torch.equal(p.detach(), before[n])])
    assert not [n for n in moved_per_step[0] if n.startswith('hyper')], 'step-1 transient changed'
    for step in (1, 2):
        assert len(moved_per_step[step]) == len(named), 'a trainable module stopped moving'
        assert sum((1 for n in moved_per_step[step] if n.startswith('hyper'))) == 4

def test_hypernetwork_stays_fp32_and_only_its_output_is_cast():
    """Measured on the real backbone: casting the hypernetwork to bf16 crashes with
    'mat1 and mat2 must have the same dtype' because the belief vector u is fp32. The basis
    keeps its low-rank branch in fp32 for the same numerical reason; only c crosses over."""
    from ga_1_tier4_rungc import HYPER_HIDDEN, HYPER_KIND, LORA_M, U_DIM
    from atom.binding import Hypernetwork
    h = Hypernetwork(U_DIM, LORA_M, kind=HYPER_KIND, hidden=HYPER_HIDDEN, seed=0)
    u = torch.randn(3, U_DIM)
    c = h(u)
    assert c.dtype == torch.float32
    assert c.to(torch.bfloat16).dtype == torch.bfloat16
    src = pathlib.Path(REPO / 'scripts' / 'ga_1_tier4_rungc.py').read_text()
    assert 'seed=SEED).to(args.device)' in src, 'hypernetwork must not be cast to bf16'
    assert 'context.set(c.to(dtype))' in src, 'c must be cast at the context boundary'

def test_rungc_panel_patch_targets_are_asserted():
    """The rung C panel must refuse to run if rung B's harness is not the shape it patches."""
    import ga_1_tier4_panel_common as panel
    import ga_1_tier4_rungc_panel as rungc_panel
    rungc_panel.assert_patch_targets_unchanged()
    original = panel.LORA_M
    try:
        panel.LORA_M = 8
        with pytest.raises(RuntimeError, match='LORA_M'):
            rungc_panel.assert_patch_targets_unchanged()
    finally:
        panel.LORA_M = original

def test_rungc_panel_belief_join_refuses_an_unrecorded_prompt():
    """A prompt that never went through the recording constructor must raise, not guess."""
    import ga_1_tier4_rungc_panel as rungc_panel
    with pytest.raises(RuntimeError, match='belief join'):
        rungc_panel.belief_for_prompt('a prompt the panel never built')

def test_rungc_mixing_vector_crosses_devices_exactly_once():
    """The hypernetwork stays on the CPU; only `c` crosses to the backbone.

    Pins the defect that killed the first tuned batch of the 6 Sep panel: `u` was moved to
    cuda while the net stayed on the cpu, so the FIRST tuned forward died with a device
    mismatch -- after the whole zero-shot arm had already been generated. A dtype/device
    contract that is only exercised on a GPU must still be testable at $0, so the crossing is
    isolated in one function and asserted here.
    """
    import ga_1_tier4_rungc_panel as rungc_panel

    class Net(torch.nn.Module):

        def forward(self, u):
            assert u.device.type == 'cpu', 'u must reach the hypernetwork on the cpu'
            return u[:, :8] * 2.0
    saved_hyper = rungc_panel._STATE.get('hyper')
    saved_prompts = dict(rungc_panel._PROMPT_EVENTS)
    saved_query = dict(rungc_panel._QUERY_BY_EVENTS)
    try:
        rungc_panel._STATE['hyper'] = Net()
        rungc_panel.belief_for_prompt = lambda text: torch.ones(64)
        c = rungc_panel.mixing_vector(['p1', 'p2'], 'cpu', torch.float16)
        assert c.shape == (2, 8)
        assert c.dtype is torch.float16
        assert c.device.type == 'cpu'
    finally:
        importlib.reload(rungc_panel)
        rungc_panel._STATE['hyper'] = saved_hyper
        rungc_panel._PROMPT_EVENTS.update(saved_prompts)
        rungc_panel._QUERY_BY_EVENTS.update(saved_query)
