from __future__ import annotations
import pathlib
import sys
import pytest
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from cw_sim_train_handoff import train_manifest
from imagination_core import Sim2RealConfig, evaluate_real_exam, train_pure_simulator

def _train_row():
    return {'source': 'simulator', 'generated_from': 'b_full', 'learner': 'L', 'problem_id': 1}

def _val_row():
    return {'source': 'simulated_validation', 'split': 'sim_val', 'generated_from': 'b_full', 'learner': 'V', 'problem_id': 1}

def test_exam_requires_all_blind_heuristics_and_real_rows():
    cells = {'sim_actor': {'s1': [0.2, 0.21]}, 'random': {'s1': [0.3, 0.31]}, 'greedy_epig': {'s1': [0.29, 0.3]}, 'max_entropy': {'s1': [0.28, 0.29]}, 'difficulty_matched': {'s1': [0.27, 0.28]}}
    with pytest.raises(AssertionError):
        evaluate_real_exam(cells=cells, training_rows=[{**_train_row(), 'source': 'real_replay'}], selection_rows=[_val_row()], exam_rows=[{'source': 'real_heldout_replay'}])
    result = evaluate_real_exam(cells=cells, training_rows=[_train_row()], selection_rows=[_val_row()], exam_rows=[{'source': 'real_heldout_replay'}], config=Sim2RealConfig(n_boot=10))
    assert result['bar_pass']

def test_train_handoff_membership_is_by_value_and_excludes_val_lockbox(tmp_path):
    split = tmp_path / 'split.csv'
    split.write_text('learner,split\nT0,train\nV0,val\nL0,lockbox\n')
    attempts = __import__('pandas').DataFrame([{'learner': 'T0', 'episode_id': 'e0', 'release': 'S19'}, {'learner': 'V0', 'episode_id': 'e1', 'release': 'F19'}])
    episodes = attempts.copy()
    manifest = {'train_learners': {'T0'}, 'attempts': attempts, 'episodes': episodes}
    out, meta = train_manifest(manifest, split)
    assert set(out['attempts']['learner']) == {'T0'}
    assert set(out['episodes']['learner']) == {'T0'}
    assert meta['train_set_equals_split_by_value']
    assert meta['val_handoff_inputs_read'] is False
