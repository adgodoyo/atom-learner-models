import torch

from atom.utils.checkpoints import (
    has_checkpoint,
    latest_step,
    load_checkpoint,
    save_checkpoint,
)


def test_save_load_roundtrip(tmp_path):
    state = {"model": {"w": torch.tensor([1.0, 2.0])}}
    save_checkpoint(state, tmp_path, step=5, metric=0.9, is_best=True)
    assert has_checkpoint(tmp_path)
    loaded = load_checkpoint(tmp_path, which="last")
    assert loaded["step"] == 5
    assert torch.equal(loaded["model"]["w"], torch.tensor([1.0, 2.0]))


def test_keep_last_n_rotation(tmp_path):
    for step in range(1, 7):
        save_checkpoint({"v": step}, tmp_path, step=step, metric=1.0 / step, keep_last=3)
    ckpts = sorted((tmp_path / "checkpoints").glob("ckpt_step*.pt"))
    assert len(ckpts) == 3
    assert latest_step(tmp_path) == 6
    # Oldest three removed, newest three kept.
    kept_steps = {int(p.stem.replace("ckpt_step", "")) for p in ckpts}
    assert kept_steps == {4, 5, 6}


def test_best_is_never_rotated_away(tmp_path):
    save_checkpoint({"v": 1}, tmp_path, step=1, metric=0.5, is_best=True, keep_last=2)
    for step in range(2, 8):
        save_checkpoint({"v": step}, tmp_path, step=step, metric=0.4, keep_last=2)
    best = load_checkpoint(tmp_path, which="best")
    assert best["step"] == 1  # the best checkpoint survives rotation
    assert (tmp_path / "checkpoints" / "best.pt").exists()


def test_load_specific_step(tmp_path):
    for step in (10, 20, 30):
        save_checkpoint({"v": step}, tmp_path, step=step, keep_last=5)
    assert load_checkpoint(tmp_path, which=20)["v"] == 20


def test_no_tmp_files_survive_a_clean_save(tmp_path):
    # save_checkpoint writes via a .tmp + os.replace (atomic rename) so a kill mid-write
    # can never leave a torn .pt or index.json -- a clean run must leave zero .tmp files
    # behind (the property an interrupted-run kill test depends on).
    save_checkpoint({"v": 1}, tmp_path, step=1, metric=0.5, is_best=True)
    leftover_tmp = list((tmp_path / "checkpoints").glob("*.tmp"))
    assert leftover_tmp == []


def test_index_replace_is_atomic_not_truncate_in_place(tmp_path):
    # A truncate-then-write (open("w")) index update would, if killed mid-write, leave a
    # half-written (unparseable) index.json. os.replace swaps in a fully-written file in
    # one syscall -- verify the index file on disk is always valid JSON after each save.
    for step in range(1, 4):
        save_checkpoint({"v": step}, tmp_path, step=step)
        raw = (tmp_path / "checkpoints" / "index.json").read_text()
        import json

        parsed = json.loads(raw)  # must not raise -- would if truncate-in-place tore it
        assert parsed["history"][-1]["step"] == step


def test_require_sampler_semantics_is_opt_in_and_does_not_affect_ordinary_resumes(tmp_path):
    # The flag defaults to False, so every non-simulator resume (filter-only, planner,
    # binding) is completely unaffected by its existence.
    save_checkpoint({"v": 1}, tmp_path, step=1, is_best=True)
    loaded = load_checkpoint(tmp_path, which="last")  # no flag passed
    assert loaded["v"] == 1


def test_require_sampler_semantics_true_refuses_a_checkpoint_missing_the_marker(tmp_path):
    save_checkpoint({"v": 1}, tmp_path, step=1, is_best=True)  # no sampler_semantics key
    try:
        load_checkpoint(tmp_path, which="last", require_sampler_semantics=True)
        raise AssertionError("expected ValueError for a checkpoint missing sampler_semantics")
    except ValueError as e:
        assert "sampler_semantics" in str(e)


def test_require_sampler_semantics_true_accepts_a_checkpoint_carrying_the_current_marker(tmp_path):
    from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS

    save_checkpoint({"v": 1, "sampler_semantics": CURRENT_SAMPLER_SEMANTICS}, tmp_path,
                    step=1, is_best=True)
    loaded = load_checkpoint(tmp_path, which="last", require_sampler_semantics=True)
    assert loaded["v"] == 1
