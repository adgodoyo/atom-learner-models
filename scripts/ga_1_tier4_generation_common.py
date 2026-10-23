"""Shared helpers for `ga_1_tier4_rungc` (writing the question, Figure 13): advantage-weighted
target sampling, prompt/target encoding with suffix-only loss, left-padded collation with a
windowed vocabulary projection, token-budget batching, plain LoRA attachment, and an
instantaneous-rate overrun trigger for long training stages.
"""
from __future__ import annotations
import pathlib
import sys
import time
from typing import Any, Sequence
import numpy as np
import torch
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from ga_1_common import (  # noqa: E402
    EXPECTED_HF_ID,
    LocalGenerationStore,
    NumericProgress,
    build_tier1_prompt,
    load_local_gemma,
    load_problem_statements,
    load_raw_episode_text,
    stable_hash,
    stable_uint64,
    stop_instance_if_requested,
    validate_no_external_api_surface,
)
TAU = 0.3
LORA_R = 16
LORA_ALPHA = 32
LORA_M = 1                      # m=1 + constant c == standard LoRA on q/v
LR = 1e-4
GRAD_CLIP = 1.0
SFT_SEEDS = (20260902, 20260903, 20260904)
SINGLE_RUN_STAMP = ("single training run — no seed-level clustering is possible "
                    "and none is claimed")
INTERNAL_HOLDOUT_LEARNERS = 40
MAX_LEN = 4096
OVERRUN_BURN_IN_BATCHES = 200
OVERRUN_WINDOW_BATCHES = 100
MIN_COMPLETION_TOKENS = 8
HISTORY_TOKENS = 3000


def sample_target(advantage: np.ndarray, learner: str, budget: int, step: int, seed: int) -> int:
    """j* ~ p*(j|s) proportional to exp(A(s,j)/tau).  Content-addressed, fixed across epochs.

    Sampling the SFT target from p* IS the advantage weighting -- it removes the read-out
    entirely.  The draw depends only on (state, seed), so epoch 2 is a genuine second pass
    over the same corpus rather than a resample.
    """
    a = np.asarray(advantage, dtype=np.float64) / TAU
    p = np.exp(a - a.max())
    p /= p.sum()
    rng = np.random.default_rng(stable_uint64(f"ga1-tier4-target|{seed}|{learner}|B{budget}|k{step}")
                                % (2 ** 63))
    return int(rng.choice(len(p), p=p))


def encode_row(prompt: str, target: str, tok: Any, max_len: int) -> dict[str, Any]:
    """Prompt masked to -100; EOS appended ONLY when the target fits.

    An EOS at an arbitrary truncation cutoff teaches premature termination; a censored
    target keeps its tokens and loses its stop symbol instead.
    """
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    room_for_prompt = max_len - MIN_COMPLETION_TOKENS
    if len(p_ids) > room_for_prompt:
        p_ids = p_ids[-room_for_prompt:]          # keep the tail: the instruction lives there
    t_ids = tok(" " + target, add_special_tokens=False)["input_ids"]
    room = max(0, max_len - len(p_ids))
    censored = (len(t_ids) + 1) > room
    c_ids = t_ids[:room] if censored else t_ids + [tok.eos_token_id]
    return {"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids,
            "n_prompt": len(p_ids), "n_target": len(c_ids), "target_censored": censored}


def collate(rows: Sequence[dict], pad_id: int, device: str) -> dict[str, torch.Tensor]:
    """LEFT-pad, and report the target window.

    Left padding is a MEMORY decision with a pre-launch arithmetic behind it.  This
    backbone's vocabulary makes the full-sequence logits tensor the dominant term long
    before parameters do: at 8,192 padded tokens it is 8192 x 262,144 x 2 B = 4.3 GB in
    bf16 and 8.6 GB more if the loss casts it to fp32, enough to run out of memory.
    Every row's supervised span is its SUFFIX, so left padding puts all of
    them flush against the right edge and ``logits_to_keep = max(n_target) + 1`` restricts
    the vocabulary projection to that window (~200 positions), cutting the term by ~40x.

    Left padding shifts absolute positions, so ``position_ids`` is supplied explicitly from
    the attention mask (HF's own left-padded-generation convention) rather than left to the
    default ``arange`` -- RoPE is relative, but an assumption that convenient should be
    asserted, not relied upon.  ``assert_windowed_loss_matches_full`` is that assertion.
    """
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, width - n:] = torch.tensor(r["input_ids"])
        labels[i, width - n:] = torch.tensor(r["labels"])
        attn[i, width - n:] = 1
    position_ids = (attn.cumsum(-1) - 1).clamp_min(0)
    window = min(width, max(r["n_target"] for r in rows) + 1)
    return {"input_ids": ids.to(device), "labels": labels.to(device),
            "attention_mask": attn.to(device), "position_ids": position_ids.to(device),
            "window": window}


def _forward(model: Any, batch: dict[str, torch.Tensor], logits_to_keep: int) -> Any:
    import inspect

    kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    if "position_ids" in inspect.signature(model.forward).parameters:
        kwargs["position_ids"] = batch["position_ids"]
    if logits_to_keep:
        kwargs["logits_to_keep"] = logits_to_keep
    return model(**kwargs)


def windowed_nll(model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Mean token NLL over the supervised suffix only, plus the token count.

    Returns the loss as a graph-carrying tensor so the training step and the held-out
    evaluation share one arithmetic path (a second implementation of the same loss is how
    a train/eval mismatch hides).
    """
    k = int(batch["window"])
    out = _forward(model, batch, k)
    logits = out.logits[:, :-1]
    labels = batch["labels"][:, -k:][:, 1:]
    keep = labels != -100
    n = int(keep.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), ignore_index=-100)
    return loss, n


def token_budget_batches(rows: Sequence[dict], token_budget: int, max_rows: int) -> list[list[int]]:
    """Bound batches on PADDED token cost, never on row count (as in `ga_1_tier0_readout`).

    Rows are grouped by length first, so a batch's padding waste is small and its peak
    activation memory is bounded by ``rows x longest-in-batch``.
    """
    order = sorted(range(len(rows)), key=lambda i: (len(rows[i]["input_ids"]), i))
    out: list[list[int]] = []
    cur: list[int] = []
    longest = 0
    for i in order:
        n = len(rows[i]["input_ids"])
        nxt = max(longest, n)
        if cur and ((len(cur) + 1) * nxt > token_budget or len(cur) + 1 > max_rows):
            out.append(cur)
            cur, longest = [i], n
        else:
            cur.append(i)
            longest = nxt
    if cur:
        out.append(cur)
    return out


def attach_plain_lora(model: Any, seed: int) -> tuple[dict, Any]:
    from atom.binding import attach_lora_basis

    bases, context = attach_lora_basis(model, m=LORA_M, r=LORA_R, alpha=LORA_ALPHA, seed=seed)
    return bases, context


class InstantaneousOverrun:
    """The live-overrun trigger, measured on the INSTANTANEOUS rate.

    The shared ``NumericProgress`` trigger arms 60 s into a stage and projects from the
    stage's CUMULATIVE rate.  For a multi-hour epoch that is structurally most pessimistic
    exactly when it first arms -- CUDA warmup, cuDNN autotuning and allocator growth are
    still in the average -- so it can stop a healthy run (e.g. 254.3 min projected against
    a 250 min threshold after 65 batches).  Resuming inflates the cumulative projection in
    the same way; the instantaneous rate is the reliable one.

    So: burn in for ``OVERRUN_BURN_IN_BATCHES``, then project from a trailing window.  The
    trigger stays a broken-run detector -- it still fires on a real slowdown -- it simply
    stops firing on warmup.  ``ga_1_common`` is left unchanged; the shared heartbeat is
    called with ``enforce_overrun=False`` for training stages instead.
    """

    def __init__(self, stage: str, total: int, threshold_minutes: int):
        self.stage, self.total, self.threshold = stage, total, threshold_minutes
        self.times: list[tuple[int, float]] = []

    def observe(self, done: int) -> float | None:
        self.times.append((done, time.time()))
        if self.threshold <= 0 or done >= self.total:
            return None
        # The rate window must contain NO warmup batch, or the trigger simply fires at the
        # burn-in boundary instead of at 60 s.  So the window is drawn only from batches after
        # burn-in, and the trigger effectively arms at burn-in + window.
        post = [t for t in self.times if t[0] > OVERRUN_BURN_IN_BATCHES]
        if len(post) < OVERRUN_WINDOW_BATCHES:
            return None
        window = post[-OVERRUN_WINDOW_BATCHES:]
        span_done, span_s = done - window[0][0], time.time() - window[0][1]
        if span_done <= 0 or span_s <= 0:
            return None
        projected_s = (self.total - done) * span_s / span_done + (time.time() - self.times[0][1])
        if projected_s > self.threshold * 60:
            raise RuntimeError(
                f"{self.stage} projected {projected_s / 60:.1f} minutes from its "
                f"instantaneous rate ({span_s / span_done:.2f} s/batch over the last "
                f"{span_done} batches), exceeding the {self.threshold}-minute trigger")
        return projected_s
